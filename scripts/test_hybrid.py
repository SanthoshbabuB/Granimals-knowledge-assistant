import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from app.ingestion.storage import ChunkStorage
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.qdrant_store import QdrantStore


CHUNK_FILE = Path(
    "data/processed/atlas_copco_2024_chunks.jsonl"
)


def main() -> None:
    storage = ChunkStorage()

    chunks = storage.load(CHUNK_FILE)

    print(f"Loaded {len(chunks)} chunks")

    bm25 = BM25Store()

    bm25.build(chunks)

    query = "What were Atlas Copco Group's revenues in 2024 in MSEK?"

    print("\nQuery:")
    print(query)

    embedding_service = EmbeddingService()

    qdrant = QdrantStore()

    query_vector = embedding_service.embed(
        [query]
    )[0]

    dense_results = qdrant.search(
        query_vector=query_vector,
        limit=15,
    )

    sparse_results = bm25.search(
        query=query,
        limit=15,
    )

    hybrid = HybridRetriever(
        rrf_k=60
    )

    hybrid_results = hybrid.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
    )

    print("\n========== HYBRID RESULTS ==========")

    for rank, result in enumerate(
        hybrid_results[:10],
        start=1,
    ):
        original_result = result["result"]

        if isinstance(original_result, dict):
            chunk = original_result["chunk"]
            page_number = chunk.page_number
            content_type = chunk.content_type
            text = chunk.text
        else:
            payload = original_result.payload
            page_number = payload.get("page_number")
            content_type = payload.get("content_type")
            text = payload.get("text", "")

        print(f"\nRank: {rank}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Page: {page_number}")
        print(f"Type: {content_type}")
        print(f"Text: {text[:500]}")


if __name__ == "__main__":
    main()