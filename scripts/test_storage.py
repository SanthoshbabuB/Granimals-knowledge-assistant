from pathlib import Path

from app.ingestion.storage import ChunkStorage


CHUNK_FILE = Path(
    "data/processed/atlas_copco_2024_chunks.jsonl"
)


def main() -> None:

    storage = ChunkStorage()

    chunks = storage.load(
        CHUNK_FILE
    )

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    print("\nFirst chunk:")

    first_chunk = chunks[0]

    print(
        f"ID: {first_chunk.chunk_id}"
    )

    print(
        f"Document: {first_chunk.document_name}"
    )

    print(
        f"Page: {first_chunk.page_number}"
    )

    print(
        f"Type: {first_chunk.content_type}"
    )

    print(
        f"Text: {first_chunk.text[:300]}"
    )


if __name__ == "__main__":
    main()