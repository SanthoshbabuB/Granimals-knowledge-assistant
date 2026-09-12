from rank_bm25 import BM25Okapi


class BM25Store:
    """In-memory BM25 sparse retrieval index."""

    def __init__(self) -> None:
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list) -> None:
        self.chunks = chunks

        tokenized_documents = [
            chunk.text.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list:

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indexes[:limit]:
            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(scores[index]),
                }
            )

        return results