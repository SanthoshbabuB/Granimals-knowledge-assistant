from pathlib import Path

from app.ingestion.storage import ChunkStorage
from app.retrieval.bm25_store import BM25Store


CHUNK_FILE = Path(
    "data/processed/atlas_copco_2024_chunks.jsonl"
)


def main() -> None:

    storage = ChunkStorage()

    chunks = storage.load(
        CHUNK_FILE
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )

    bm25 = BM25Store()

    bm25.build(
        chunks
    )

    query = (
        "What were Atlas Copco Group's "
        "revenues in 2024 in MSEK?"
    )

    results = bm25.search(
        query=query,
        limit=5,
    )

    print(
        "\n========== BM25 RESULTS =========="
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]
        score = result["score"]

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Page: {chunk.page_number}"
        )

        print(
            f"Type: {chunk.content_type}"
        )

        print(
            f"Text: {chunk.text[:500]}"
        )


if __name__ == "__main__":
    main()