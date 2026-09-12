from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings


class QdrantStore:
    """Manage the Qdrant vector collection."""

    def __init__(self) -> None:
        settings = get_settings()

        self.client = QdrantClient(
            url=settings.qdrant_url
        )

        self.collection_name = (
            settings.qdrant_collection
        )

    def create_collection(
        self,
        vector_size: int,
    ) -> None:

        collections = (
            self.client.get_collections()
        )

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name in collection_names:
            print(
                f"Collection already exists: "
                f"{self.collection_name}"
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{self.collection_name}"
        )

    def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        """Create the collection if it does not already exist."""

        collections = (
            self.client.get_collections()
        )

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name not in collection_names:
            self.create_collection(
                vector_size=vector_size
            )

    def upsert(
        self,
        points: list[PointStruct],
    ) -> None:

        if not points:
            return

        vector_size = len(
            points[0].vector
        )

        self._ensure_collection(
            vector_size=vector_size
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            f"Uploaded {len(points)} points"
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list:
        """Search for the most similar chunks."""

        self._ensure_collection(
            vector_size=len(query_vector)
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return results.points