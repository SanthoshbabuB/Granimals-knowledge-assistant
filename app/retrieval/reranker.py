import re


class Reranker:
    """
    Fast deterministic reranker.

    This reranker does not call an external API
    or an LLM. It combines multiple relevance
    signals to reorder hybrid search results.

    Signals:

    - Exact phrase matching
    - Number matching
    - Year matching
    - Important keyword overlap
    - Dense similarity
    - RRF score
    """

    def __init__(self) -> None:
        pass

    # ==================================================
    # Tokenization
    # ==================================================

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        """Convert text into normalized tokens."""

        return re.findall(
            r"\b[a-zA-Z0-9]+(?:\.[0-9]+)?\b",
            text.lower(),
        )

    # ==================================================
    # Important query words
    # ==================================================

    def _important_words(
        self,
        query: str,
    ) -> set[str]:
        """
        Remove common question words.

        Example:

        What were Atlas Copco Group's
        revenues in 2024 in MSEK?

        becomes approximately:

        atlas
        copco
        group
        revenues
        2024
        mseK
        """

        stop_words = {
            "what",
            "was",
            "were",
            "is",
            "are",
            "the",
            "a",
            "an",
            "in",
            "on",
            "of",
            "for",
            "to",
            "from",
            "how",
            "many",
            "much",
            "does",
            "did",
            "do",
            "and",
            "or",
            "with",
            "about",
            "which",
            "where",
            "when",
            "who",
        }

        tokens = self._tokenize(
            query
        )

        return {
            token
            for token in tokens
            if token not in stop_words
        }

    # ==================================================
    # Keyword score
    # ==================================================

    def _keyword_score(
        self,
        query: str,
        text: str,
    ) -> float:
        """Calculate important keyword overlap."""

        query_words = self._important_words(
            query
        )

        text_words = set(
            self._tokenize(text)
        )

        if not query_words:
            return 0.0

        overlap = query_words.intersection(
            text_words
        )

        return (
            len(overlap)
            / len(query_words)
        )

    # ==================================================
    # Exact phrase score
    # ==================================================

    def _phrase_score(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Detect exact query phrases.

        Exact phrase matches receive a strong boost.
        """

        query_lower = query.lower()
        text_lower = text.lower()

        score = 0.0

        # --------------------------------------------------
        # Full query match
        # --------------------------------------------------

        if query_lower in text_lower:
            score = 1.0

        # --------------------------------------------------
        # Multi-word phrase matching
        # --------------------------------------------------

        query_words = self._tokenize(
            query
        )

        if len(query_words) >= 2:

            for size in (2, 3):

                for index in range(
                    len(query_words) - size + 1
                ):

                    phrase = " ".join(
                        query_words[
                            index:index + size
                        ]
                    )

                    if phrase in text_lower:
                        score += 0.20

        return min(
            score,
            1.0,
        )

    # ==================================================
    # Number score
    # ==================================================

    def _number_score(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Compare numbers present in the query
        with numbers present in the document.

        Useful for:

        - Financial values
        - Years
        - Percentages
        - Quantities
        """

        query_numbers = set(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\b",
                query,
            )
        )

        if not query_numbers:
            return 0.0

        text_numbers = set(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\b",
                text,
            )
        )

        matched_numbers = (
            query_numbers.intersection(
                text_numbers
            )
        )

        return (
            len(matched_numbers)
            / len(query_numbers)
        )

    # ==================================================
    # Year score
    # ==================================================

    def _year_score(
        self,
        query: str,
        text: str,
    ) -> float:
        """Give additional weight to requested years."""

        query_years = set(
            re.findall(
                r"\b20\d{2}\b",
                query,
            )
        )

        if not query_years:
            return 0.0

        text_years = set(
            re.findall(
                r"\b20\d{2}\b",
                text,
            )
        )

        matched_years = (
            query_years.intersection(
                text_years
            )
        )

        return (
            len(matched_years)
            / len(query_years)
        )

    # ==================================================
    # Dense similarity normalization
    # ==================================================

    def _dense_score(
        self,
        score: float,
    ) -> float:
        """Normalize dense cosine similarity."""

        return max(
            0.0,
            min(
                float(score),
                1.0,
            ),
        )

    # ==================================================
    # RRF normalization
    # ==================================================

    def _rrf_score(
        self,
        score: float,
    ) -> float:
        """Normalize RRF score."""

        return max(
            0.0,
            min(
                float(score) * 30,
                1.0,
            ),
        )

    # ==================================================
    # Final scoring
    # ==================================================

    def _calculate_score(
        self,
        query: str,
        text: str,
        rrf_score: float,
        dense_score: float,
    ) -> float:
        """Calculate the final reranking score."""

        keyword_score = (
            self._keyword_score(
                query,
                text,
            )
        )

        phrase_score = (
            self._phrase_score(
                query,
                text,
            )
        )

        number_score = (
            self._number_score(
                query,
                text,
            )
        )

        year_score = (
            self._year_score(
                query,
                text,
            )
        )

        normalized_dense = (
            self._dense_score(
                dense_score
            )
        )

        normalized_rrf = (
            self._rrf_score(
                rrf_score
            )
        )

        # --------------------------------------------------
        # Weighted final score
        # --------------------------------------------------

        final_score = (
            0.15 * phrase_score
            + 0.10 * number_score
            + 0.10 * year_score
            + 0.15 * keyword_score
            + 0.45 * normalized_dense
            + 0.05 * normalized_rrf
)

        return final_score

    # ==================================================
    # Rerank
    # ==================================================

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ) -> list:
        """Rerank hybrid candidates."""

        if not candidates:
            return []

        reranked = []

        for candidate in candidates:

            original_result = candidate[
                "result"
            ]

            # --------------------------------------------------
            # Extract text
            # --------------------------------------------------

            if isinstance(
                original_result,
                dict,
            ):

                chunk = original_result[
                    "chunk"
                ]

                text = chunk.text

            else:

                payload = (
                    original_result.payload
                )

                text = payload.get(
                    "text",
                    "",
                )

            # --------------------------------------------------
            # Extract retrieval scores
            # --------------------------------------------------

            rrf_score = candidate.get(
                "rrf_score",
                candidate.get(
                    "score",
                    0.0,
                ),
            )

            dense_score = candidate.get(
                "dense_score",
                0.0,
            )

            bm25_score = candidate.get(
                "bm25_score",
                0.0,
            )

            # --------------------------------------------------
            # Calculate final score
            # --------------------------------------------------

            score = self._calculate_score(
                query=query,
                text=text,
                rrf_score=rrf_score,
                dense_score=dense_score,
            )

            reranked.append(
                {
                    "chunk_id": candidate[
                        "chunk_id"
                    ],
                    "score": score,
                    "rrf_score": rrf_score,
                    "dense_score": dense_score,
                    "bm25_score": bm25_score,
                    "result": original_result,
                }
            )

        # --------------------------------------------------
        # Sort by final score
        # --------------------------------------------------

        reranked.sort(
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        return reranked[:top_k]