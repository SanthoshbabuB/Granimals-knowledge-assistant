from app.retrieval.embeddings import EmbeddingService
from app.retrieval.qdrant_store import QdrantStore


class RetrievalService:
    """Retrieve relevant chunks for a user query."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.qdrant = QdrantStore()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list:
        """Retrieve the most relevant chunks."""

        query_vector = self.embedding_service.embed(
            [query]
        )[0]

        results = self.qdrant.search(
            query_vector=query_vector,
            limit=limit,
        )

        return results