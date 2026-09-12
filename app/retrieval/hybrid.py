class HybridRetriever:
    """Combine dense and BM25 retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        rrf_k: int = 60,
    ) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        dense_results: list,
        sparse_results: list,
    ) -> list:
        """Fuse dense and sparse results using RRF."""

        scores: dict[str, float] = {}
        items: dict[str, object] = {}

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        # ==================================================
        # 1. Process dense results
        # ==================================================

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result.payload[
                "chunk_id"
            ]

            dense_score = float(
                result.score
            )

            dense_scores[chunk_id] = (
                dense_score
            )

            rrf_contribution = (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

            scores[chunk_id] = (
                scores.get(
                    chunk_id,
                    0.0,
                )
                + rrf_contribution
            )

            items[chunk_id] = result

        # ==================================================
        # 2. Process BM25 results
        # ==================================================

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            chunk = result["chunk"]

            chunk_id = chunk.chunk_id

            bm25_score = float(
                result["score"]
            )

            bm25_scores[chunk_id] = (
                bm25_score
            )

            rrf_contribution = (
                1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

            scores[chunk_id] = (
                scores.get(
                    chunk_id,
                    0.0,
                )
                + rrf_contribution
            )

            items[chunk_id] = result

        # ==================================================
        # 3. Sort by RRF score
        # ==================================================

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # ==================================================
        # 4. Build final hybrid results
        # ==================================================

        results = []

        for chunk_id, rrf_score in ranked:

            results.append(
                {
                    "chunk_id": chunk_id,
                    "rrf_score": rrf_score,
                    "dense_score": dense_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    "bm25_score": bm25_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    "result": items[chunk_id],
                }
            )

        return results