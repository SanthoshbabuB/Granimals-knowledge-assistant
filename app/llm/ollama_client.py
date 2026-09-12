import logging

from ollama import Client

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama models."""

    def __init__(self) -> None:

        settings = get_settings()

        self.client = Client(
            host=settings.ollama_base_url
        )

        self.model = settings.ollama_model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate a deterministic response."""

        logger.info(
            "Generating response using model=%s",
            self.model,
        )

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": temperature,
                "top_p": 0.1,
                "seed": 42,
            },
        )

        return response["message"]["content"].strip()