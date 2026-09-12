import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from app.rag.pipeline import RAGPipeline


def main() -> None:
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    pipeline = RAGPipeline()

    questions = [
        "What were Atlas Copco Group's revenues in 2024 in MSEK?",
        "What was the operating margin in 2024?",
        "What was Atlas Copco Group's revenue in 2020?",
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = pipeline.answer(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nCONFIDENCE:")
        print(result["confidence"])

        print("\nRETRIEVED CHUNKS:")
        print(result["retrieved_chunks"])

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                f"- {source['document']}, "
                f"Page {source['page']} "
                f"({source['content_type']})"
            )


if __name__ == "__main__":
    main()
    