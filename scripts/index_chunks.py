from pathlib import Path
import uuid

from qdrant_client.models import PointStruct

from app.ingestion.storage import ChunkStorage
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.qdrant_store import QdrantStore


CHUNK_FILE = Path(
    "data/processed/atlas_copco_2024_chunks.jsonl"
)

BATCH_SIZE = 32


def main() -> None:

    print("Loading chunks...")

    storage = ChunkStorage()

    chunks = storage.load(
        CHUNK_FILE
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )

    embedding_service = (
        EmbeddingService()
    )

    qdrant = QdrantStore()

    qdrant.create_collection(
        vector_size=768
    )

    total = len(chunks)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        print(
            f"\nEmbedding batch "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, total)}"
            f"/{total}"
        )

        texts = [
            chunk.text
            for chunk in batch
        ]

        embeddings = (
            embedding_service.embed(
                texts
            )
        )

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
                id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk.chunk_id,
    )),
            vector=vector,
            payload=payload,
)

            points.append(point)

        qdrant.upsert(
            points
        )

    print(
        "\n========== INDEXING COMPLETE =========="
    )

    print(
        f"Indexed {total} chunks"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()