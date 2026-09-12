from app.retrieval.qdrant_store import QdrantStore


def main() -> None:
    store = QdrantStore()

    store.create_collection(
        vector_size=768
    )


if __name__ == "__main__":
    main()