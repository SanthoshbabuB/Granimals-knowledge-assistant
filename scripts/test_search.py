from app.retrieval.service import RetrievalService


def main() -> None:

    query = "What were Atlas Copco Group's revenues in 2024 in MSEK?"

    print("Query:")
    print(query)

    retrieval_service = RetrievalService()

    results = retrieval_service.retrieve(
        query=query,
        limit=5,
    )

    print("\n========== SEARCH RESULTS ==========")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = result.payload

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Document: "
            f"{payload.get('document_name')}"
        )

        print(
            f"Page: "
            f"{payload.get('page_number')}"
        )

        print(
            f"Type: "
            f"{payload.get('content_type')}"
        )

        print(
            f"Text: "
            f"{payload.get('text', '')[:500]}"
        )


if __name__ == "__main__":
    main()