import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from app.ingestion.storage import ChunkStorage
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.reranker import Reranker


CHUNK_FILE = Path(
    "data/processed/atlas_copco_2024_chunks.jsonl"
)


def get_result_details(
    candidate,
):
    """Extract page, content type, and text."""

    original_result = candidate[
        "result"
    ]

    # BM25 result
    if isinstance(
        original_result,
        dict,
    ):

        chunk = original_result[
            "chunk"
        ]

        return (
            chunk.page_number,
            chunk.content_type,
            chunk.text,
        )

    # Qdrant result
    payload = (
        original_result.payload
    )

    return (
        payload.get(
            "page_number"
        ),
        payload.get(
            "content_type"
        ),
        payload.get(
            "text",
            "",
        ),
    )


def main() -> None:

    # ==================================================
    # 1. Load chunks
    # ==================================================

    print(
        "Loading chunks..."
    )

    storage = ChunkStorage()

    chunks = storage.load(
        CHUNK_FILE
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )

    # ==================================================
    # 2. Build BM25 index
    # ==================================================

    print(
        "\nBuilding BM25 index..."
    )

    bm25 = BM25Store()

    bm25.build(
        chunks
    )

    print(
        "BM25 index ready."
    )

    # ==================================================
    # 3. Query
    # ==================================================

    query = (
        "What were Atlas Copco Group's "
        "revenues in 2024 in MSEK?"
    )

    print(
        "\n========== QUERY =========="
    )

    print(
        query
    )

    # ==================================================
    # 4. Dense retrieval
    # ==================================================

    print(
        "\nRunning dense retrieval..."
    )

    embedding_service = (
        EmbeddingService()
    )

    qdrant = QdrantStore()

    query_vector = (
        embedding_service.embed(
            [query]
        )[0]
    )

    dense_results = (
        qdrant.search(
            query_vector=query_vector,
            limit=15,
        )
    )

    print(
        f"Dense results: "
        f"{len(dense_results)}"
    )

    # ==================================================
    # 5. BM25 retrieval
    # ==================================================

    print(
        "\nRunning BM25 retrieval..."
    )

    sparse_results = (
        bm25.search(
            query=query,
            limit=15,
        )
    )

    print(
        f"BM25 results: "
        f"{len(sparse_results)}"
    )

    # ==================================================
    # 6. Hybrid RRF
    # ==================================================

    print(
        "\nRunning hybrid RRF..."
    )

    hybrid = HybridRetriever(
        rrf_k=60
    )

    hybrid_results = (
        hybrid.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )
    )

    print(
        f"Hybrid candidates: "
        f"{len(hybrid_results)}"
    )

    # ==================================================
    # 7. Display hybrid results
    # ==================================================

    print(
        "\n========== HYBRID TOP 10 =========="
    )

    for rank, candidate in enumerate(
        hybrid_results[:10],
        start=1,
    ):

        (
            page_number,
            content_type,
            text,
        ) = get_result_details(
            candidate
        )

        print(
            f"\nRank: {rank}"
        )

        print(
            f"RRF Score: "
            f"{candidate['rrf_score']:.6f}"
        )

        print(
            f"Dense Score: "
            f"{candidate['dense_score']:.6f}"
        )

        print(
            f"BM25 Score: "
            f"{candidate['bm25_score']:.6f}"
        )

        print(
            f"Page: "
            f"{page_number}"
        )

        print(
            f"Type: "
            f"{content_type}"
        )

        print(
            f"Text: "
            f"{text[:400]}"
        )

    # ==================================================
    # 8. Initialize reranker
    # ==================================================

    print(
        "\nInitializing fast local reranker..."
    )

    reranker = Reranker()

    # ==================================================
    # 9. Rerank
    # ==================================================

    print(
        "\nRunning fast reranking..."
    )

    reranked_results = (
        reranker.rerank(
            query=query,
            candidates=hybrid_results[:15],
            top_k=5,
        )
    )

    # ==================================================
    # 10. Display reranked results
    # ==================================================

    print(
        "\n========== RERANKED RESULTS =========="
    )

    for rank, candidate in enumerate(
        reranked_results,
        start=1,
    ):

        (
            page_number,
            content_type,
            text,
        ) = get_result_details(
            candidate
        )

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Reranker Score: "
            f"{candidate['score']:.4f}"
        )

        print(
            f"RRF Score: "
            f"{candidate['rrf_score']:.6f}"
        )

        print(
            f"Dense Score: "
            f"{candidate['dense_score']:.6f}"
        )

        print(
            f"BM25 Score: "
            f"{candidate['bm25_score']:.6f}"
        )

        print(
            f"Page: "
            f"{page_number}"
        )

        print(
            f"Type: "
            f"{content_type}"
        )

        print(
            f"Text: "
            f"{text[:500]}"
        )

    # ==================================================
    # 11. Completion
    # ==================================================

    print(
        "\n========== RERANKING COMPLETE =========="
    )

    print(
        f"Returned top "
        f"{len(reranked_results)} "
        f"results."
    )


if __name__ == "__main__":
    main()