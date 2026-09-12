from ollama import Client

from app.core.config import get_settings


class EmbeddingService:
    """Generate embeddings using the configured local embedding model."""

    def __init__(self) -> None:
        settings = get_settings()

        self.client = Client(
            host=settings.ollama_base_url
        )

        self.model = settings.ollama_embedding_model

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        return response["embeddings"]   