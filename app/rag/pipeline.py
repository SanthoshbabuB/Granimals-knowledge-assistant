import uuid
from pathlib import Path

from langchain_core.messages import BaseMessage
from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import ChunkStorage
from app.llm.ollama_client import OllamaClient
from app.rag.confidence import ConfidenceChecker
from app.rag.context_builder import ContextBuilder
from app.rag.prompts import build_prompt
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.query_expander import QueryExpander
from app.retrieval.reranker import Reranker


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Pipeline:

        User Question
              |
              v
        Generic Query Expansion
              |
              v
        Dense Retrieval + BM25
              |
              v
        Reciprocal Rank Fusion
              |
              v
        Deterministic Reranking
              |
              v
        Confidence Check
              |
              v
        Context Construction
              |
              v
        Local Ollama LLM
              |
              v
        Grounded Answer + Sources

    The retrieval layer is document-independent and can work
    across multiple PDFs.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.dense_top_k = settings.dense_top_k
        self.sparse_top_k = settings.sparse_top_k
        self.rerank_top_k = settings.rerank_top_k
        self.max_context_chunks = settings.max_context_chunks
        self.temperature = settings.temperature

        # ---------------------------------------------------------
        # Retrieval services
        # ---------------------------------------------------------

        self.embedding_service = EmbeddingService()

        self.qdrant = QdrantStore()

        self.bm25 = BM25Store()

        self.query_expander = QueryExpander()

        self.hybrid = HybridRetriever(
            rrf_k=settings.rrf_k
        )

        self.reranker = Reranker()

        # ---------------------------------------------------------
        # RAG services
        # ---------------------------------------------------------

        self.context_builder = ContextBuilder()

        self.confidence_checker = ConfidenceChecker(
            minimum_score=0.10,
            minimum_keyword_coverage=0.30,
        )

        # ---------------------------------------------------------
        # Local LLM
        # ---------------------------------------------------------

        self.llm = OllamaClient()

        # ---------------------------------------------------------
        # Document ingestion
        # ---------------------------------------------------------

        self.ingestion_pipeline = IngestionPipeline(
            asset_root=Path("data/assets"),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            enable_ocr=settings.enable_ocr,
            enable_vision=settings.enable_vision,
        )

        self.chunk_storage = ChunkStorage()

        # ---------------------------------------------------------
        # Build BM25 index from ALL processed documents
        # ---------------------------------------------------------

        self._build_bm25_index()

    # ============================================================
    # DOCUMENT INGESTION
    # ============================================================

    def ingest_document(self, pdf_path: Path) -> dict:
        """
        Process one PDF document.

        The ingestion pipeline handles:
        - page-level text
        - tables
        - embedded images
        - rendered pages
        - OCR
        - text chunking

        The processed chunks are saved to JSONL.

        Returns:
            Dictionary containing document information and
            number of generated chunks.
        """

        print("\n" + "=" * 70)
        print("DOCUMENT INGESTION")
        print("=" * 70)

        print(
            f"Processing document: {pdf_path.name}"
        )

        # ---------------------------------------------------------
        # 1. Process PDF
        # ---------------------------------------------------------

        document, chunks = (
            self.ingestion_pipeline.process(
                pdf_path
            )
        )

        if not chunks:
            raise ValueError(
                f"No chunks were created from "
                f"{pdf_path.name}"
            )

        # ---------------------------------------------------------
        # 2. Save processed chunks
        # ---------------------------------------------------------

        processed_dir = Path(
            "data/processed"
        )

        processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed_path = (
            processed_dir
            / f"{document.document_id}_chunks.jsonl"
        )

        self.chunk_storage.save(
            chunks=chunks,
            output_path=processed_path,
        )

        print(
            f"Saved {len(chunks)} chunks to "
            f"{processed_path}"
        )

        # ---------------------------------------------------------
        # 3. Create embeddings
        # ---------------------------------------------------------

        batch_size = 32

        total = len(chunks)

        for start in range(
            0,
            total,
            batch_size,
        ):
            batch = chunks[
                start:start + batch_size
            ]

            print(
                f"\nEmbedding upload batch "
                f"{start + 1}-"
                f"{min(start + batch_size, total)}"
                f"/{total}"
            )

            texts = [
                chunk.text
                for chunk in batch
            ]

            embeddings = (
                self.embedding_service.embed(
                    texts
                )
            )

            # -----------------------------------------------------
            # 4. Create Qdrant points
            # -----------------------------------------------------

            points: list[PointStruct] = []

            for chunk, vector in zip(
                batch,
                embeddings,
            ):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "content_type": chunk.content_type,
                    "image_id": chunk.image_id,
                    "image_path": chunk.image_path,
                    "table_id": chunk.table_id,
                    "page_image_path": chunk.page_image_path,
                }

                point = PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            chunk.chunk_id,
                        )
                    ),
                    vector=vector,
                    payload=payload,
                )

                points.append(point)

            # -----------------------------------------------------
            # 5. Upload batch to Qdrant
            # -----------------------------------------------------

            self.qdrant.upsert(
                points
            )

        # ---------------------------------------------------------
        # 6. Rebuild BM25 index
        # ---------------------------------------------------------

        self._build_bm25_index()

        print("\n" + "=" * 70)
        print("DOCUMENT INGESTION COMPLETE")
        print("=" * 70)

        print(
            f"Document: {document.document_name}"
        )

        print(
            f"Chunks: {len(chunks)}"
        )

        print("=" * 70)

        return {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "chunks": len(chunks),
            "processed_file": str(
                processed_path
            ),
        }

    # ============================================================
    # BM25 INDEX
    # ============================================================

    def _build_bm25_index(self) -> None:
        """
        Build one BM25 index across all processed documents.

        Any file matching:

            data/processed/*_chunks.jsonl

        is automatically included.

        This makes the retrieval layer scalable to multiple PDFs
        without hardcoding document names.
        """

        processed_dir = Path(
            "data/processed"
        )

        # ---------------------------------------------------------
        # Validate processed directory
        # ---------------------------------------------------------

        if not processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory not found: "
                f"{processed_dir}"
            )

        # ---------------------------------------------------------
        # Discover all chunk files
        # ---------------------------------------------------------

        chunk_files = sorted(
            processed_dir.glob(
                "*_chunks.jsonl"
            )
        )

        if not chunk_files:
            raise FileNotFoundError(
                "No processed chunk files were found "
                f"in {processed_dir}. "
                "Run the ingestion pipeline first."
            )

        print("\n" + "=" * 70)
        print("BUILDING BM25 INDEX")
        print("=" * 70)

        print(
            f"Found {len(chunk_files)} "
            "processed document(s):"
        )

        for chunk_file in chunk_files:
            print(
                f"  - {chunk_file.name}"
            )

        # ---------------------------------------------------------
        # Load all chunks
        # ---------------------------------------------------------

        storage = ChunkStorage()

        all_chunks = []

        for chunk_file in chunk_files:

            print(
                f"\nLoading: "
                f"{chunk_file.name}"
            )

            chunks = storage.load(
                chunk_file
            )

            print(
                f"Loaded {len(chunks)} chunks"
            )

            all_chunks.extend(
                chunks
            )

        # ---------------------------------------------------------
        # Validate chunks
        # ---------------------------------------------------------

        if not all_chunks:
            raise RuntimeError(
                "Processed chunk files were found, "
                "but no chunks were loaded."
            )

        # ---------------------------------------------------------
        # Build one global BM25 index
        # ---------------------------------------------------------

        self.bm25.build(
            all_chunks
        )

        # ---------------------------------------------------------
        # Print summary
        # ---------------------------------------------------------

        document_names = sorted(
            {
                chunk.document_name
                for chunk in all_chunks
            }
        )

        print("\n" + "=" * 70)
        print("BM25 INDEX READY")
        print("=" * 70)

        print(
            f"Total chunks: "
            f"{len(all_chunks)}"
        )

        print(
            f"Documents: "
            f"{len(document_names)}"
        )

        for document_name in document_names:
            document_chunk_count = sum(
                1
                for chunk in all_chunks
                if chunk.document_name
                == document_name
            )

            print(
                f"  - {document_name}: "
                f"{document_chunk_count} chunks"
            )

        print("=" * 70)

    # ============================================================
    # DENSE RETRIEVAL
    # ============================================================

    def _dense_retrieve(
        self,
        query: str,
    ) -> list:
        """
        Perform semantic retrieval using Qdrant.
        """

        query_vector = (
            self.embedding_service.embed(
                [query]
            )[0]
        )

        results = self.qdrant.search(
            query_vector=query_vector,
            limit=self.dense_top_k,
        )

        return results

    # ============================================================
    # SPARSE RETRIEVAL
    # ============================================================

    def _sparse_retrieve(
        self,
        query: str,
    ) -> list:
        """
        Perform lexical retrieval using BM25.
        """

        return self.bm25.search(
            query=query,
            limit=self.sparse_top_k,
        )

    # ============================================================
    # HYBRID RETRIEVAL
    # ============================================================

    def _hybrid_retrieve(
        self,
        question: str,
    ) -> list:
        """
        Perform dense + sparse hybrid retrieval.

        Query expansion is generic and document-independent.

        The expanded query is used only for retrieval.

        The original question is preserved for:
        - reranking
        - confidence checking
        - prompt construction
        - answer generation
        """

        expanded_query = (
            self.query_expander.expand(
                question
            )
        )

        # ---------------------------------------------------------
        # Debug information
        # ---------------------------------------------------------

        print("\n" + "-" * 70)

        print("ORIGINAL QUERY:")
        print(question)

        print("\nEXPANDED QUERY:")
        print(
            expanded_query.expanded
        )

        print("\nKEYWORDS:")
        print(
            expanded_query.keywords
        )

        print("\nYEARS:")
        print(
            expanded_query.years
        )

        print("\nNUMBERS:")
        print(
            expanded_query.numbers
        )

        print("\nENTITIES:")
        print(
            expanded_query.entities
        )

        print("\nUNITS:")
        print(
            expanded_query.units
        )

        print("-" * 70)

        # ---------------------------------------------------------
        # Dense retrieval
        # ---------------------------------------------------------

        dense_results = (
            self._dense_retrieve(
                expanded_query.expanded
            )
        )

        # ---------------------------------------------------------
        # Sparse retrieval
        # ---------------------------------------------------------

        sparse_results = (
            self._sparse_retrieve(
                expanded_query.expanded
            )
        )

        # ---------------------------------------------------------
        # RRF fusion
        # ---------------------------------------------------------

        hybrid_results = (
            self.hybrid.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
            )
        )

        return hybrid_results

    # ============================================================
    # RERANKING
    # ============================================================

    def _rerank(
        self,
        question: str,
        candidates: list,
    ) -> list:
        """
        Rerank candidates against the original user question.

        The original question is deliberately used rather
        than the expanded query.
        """

        return self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=self.rerank_top_k,
        )

    # ============================================================
    # CONFIDENCE
    # ============================================================

    def _check_confidence(
        self,
        question: str,
        reranked_results: list,
    ):
        """
        Check whether retrieved evidence appears sufficient.
        """

        return self.confidence_checker.check(
            question=question,
            reranked_results=reranked_results,
        )

    # ============================================================
    # CONTEXT
    # ============================================================

    def _build_context(
        self,
        reranked_results: list,
    ):
        """
        Build the final context supplied to the LLM.
        """

        selected_results = (
            reranked_results[
                : self.max_context_chunks
            ]
        )

        context_items = (
            self.context_builder.build(
                selected_results
            )
        )

        context = (
            self.context_builder.format_for_llm(
                context_items
            )
        )

        return context_items, context

    # ============================================================
    # SOURCES
    # ============================================================

    def _build_sources(
        self,
        context_items: list,
    ) -> list[dict]:
        """
        Build citation/source metadata.

        Duplicate document/page combinations are removed.
        """

        sources = []

        seen_sources = set()

        for item in context_items:

            source_key = (
                item.document_name,
                item.page_number,
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            sources.append(
                {
                    "document": item.document_name,
                    "page": item.page_number,
                    "content_type": item.content_type,
                    "chunk_id": item.chunk_id,
                    "image_path": item.image_path,
                    "page_image_path": item.page_image_path,
                }
            )

        return sources

    # ============================================================
    # DEBUG RESULTS
    # ============================================================

    def _print_reranked_results(
        self,
        reranked_results: list,
    ) -> None:
        """
        Print reranked results for development/debugging.
        """

        print("\n" + "=" * 70)
        print("RERANKED RESULTS")
        print("=" * 70)

        for index, result in enumerate(
            reranked_results,
            start=1,
        ):

            original_result = result[
                "result"
            ]

            # -----------------------------------------------------
            # BM25 result
            # -----------------------------------------------------

            if isinstance(
                original_result,
                dict,
            ):

                chunk = original_result[
                    "chunk"
                ]

                document_name = (
                    chunk.document_name
                )

                page_number = (
                    chunk.page_number
                )

                content_type = (
                    chunk.content_type
                )

                text = chunk.text

            # -----------------------------------------------------
            # Qdrant result
            # -----------------------------------------------------

            else:

                payload = (
                    original_result.payload
                )

                document_name = (
                    payload[
                        "document_name"
                    ]
                )

                page_number = (
                    payload[
                        "page_number"
                    ]
                )

                content_type = (
                    payload[
                        "content_type"
                    ]
                )

                text = payload.get(
                    "text",
                    "",
                )

            print(
                f"\nRank {index}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
            )

            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"Document: "
                f"{document_name}"
            )

            print(
                f"Page: "
                f"{page_number}"
            )

            print(
                f"Content Type: "
                f"{content_type}"
            )

            print(
                f"Dense Score: "
                f"{result.get('dense_score', 0.0):.4f}"
            )

            print(
                f"BM25 Score: "
                f"{result.get('bm25_score', 0.0):.4f}"
            )

            print(
                f"RRF Score: "
                f"{result.get('rrf_score', 0.0):.4f}"
            )

            print(
                f"Text: "
                f"{text[:500]}"
            )

    # ============================================================
    # DEBUG CONFIDENCE
    # ============================================================

    def _print_confidence(
        self,
        confidence,
    ) -> None:
        """
        Print confidence information.
        """

        print("\n" + "=" * 70)
        print("CONFIDENCE")
        print("=" * 70)

        print(
            f"Confident: "
            f"{confidence.confident}"
        )

        print(
            f"Score: "
            f"{confidence.score:.4f}"
        )

        print(
            f"Reason: "
            f"{confidence.reason}"
        )

    # ============================================================
    # PUBLIC ANSWER METHOD
    # ============================================================

    def answer(
        self,
        question: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> dict:
        """
        Execute the complete RAG pipeline.

        Args:
            question: Current user question.
            conversation_history: Previous messages in the conversation.

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "confidence": 0.91,
                "retrieved_chunks": 5
            }
        """

        question = question.strip()

        # ---------------------------------------------------------
        # Empty question
        # ---------------------------------------------------------

        if not question:

            return {
                "answer": "Please provide a question.",
                "sources": [],
                "confidence": 0.0,
                "retrieved_chunks": 0,
            }

        # ---------------------------------------------------------
        # STEP 1
        # Hybrid retrieval
        # ---------------------------------------------------------

        hybrid_results = (
            self._hybrid_retrieve(
                question
            )
        )

        # ---------------------------------------------------------
        # STEP 2
        # Reranking
        # ---------------------------------------------------------

        reranked_results = (
            self._rerank(
                question=question,
                candidates=hybrid_results,
            )
        )

        # ---------------------------------------------------------
        # Debug results
        # ---------------------------------------------------------

        self._print_reranked_results(
            reranked_results
        )

        # ---------------------------------------------------------
        # STEP 3
        # Confidence
        # ---------------------------------------------------------

        confidence = (
            self._check_confidence(
                question=question,
                reranked_results=reranked_results,
            )
        )

        # ---------------------------------------------------------
        # Debug confidence
        # ---------------------------------------------------------

        self._print_confidence(
            confidence
        )

        # ---------------------------------------------------------
        # STEP 4
        # Safe refusal
        # ---------------------------------------------------------

        if not confidence.confident:

            return {
                "answer": (
                    "I could not find enough information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "sources": [],
                "confidence": confidence.score,
                "retrieved_chunks": 0,
            }

        # ---------------------------------------------------------
        # STEP 5
        # Context
        # ---------------------------------------------------------

        context_items, context = (
            self._build_context(
                reranked_results
            )
        )

        # ---------------------------------------------------------
        # STEP 6
        # Grounded prompt
        # ---------------------------------------------------------

        prompt = build_prompt(
            question=question,
            context=context,
            conversation_history=conversation_history,
        )

        # ---------------------------------------------------------
        # STEP 7
        # Local LLM generation
        # ---------------------------------------------------------

        answer = self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
        )

        # ---------------------------------------------------------
        # STEP 8
        # Sources
        # ---------------------------------------------------------

        sources = self._build_sources(
            context_items
        )

        # ---------------------------------------------------------
        # STEP 9
        # Final result
        # ---------------------------------------------------------

        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": confidence.score,
            "retrieved_chunks": len(
                context_items
            ),
        }