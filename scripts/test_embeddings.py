from ollama import Client

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    client = Client(
        host=settings.ollama_base_url
    )

    texts = [
        """
        Atlas Copco is a global industrial company
        providing compressors, vacuum solutions,
        power techniques and industrial tools.
        """,

        """
        Table: Financial performance

        Year | Revenue | Operating profit
        2024 | 176,864 | 34,500
        2023 | 170,000 | 32,000
        """,
    ]

    response = client.embed(
        model=settings.ollama_embedding_model,
        input=texts,
    )

    embeddings = response["embeddings"]

    print("========== EMBEDDING TEST ==========")

    print(
        f"Number of inputs: {len(texts)}"
    )

    print(
        f"Number of embeddings: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: {len(embeddings[0])}"
    )

    print("\nText embedding:")
    print(
        embeddings[0][:10]
    )

    print("\nTable embedding:")
    print(
        embeddings[1][:10]
    )

    print("\n====================================")


if __name__ == "__main__":
    main()