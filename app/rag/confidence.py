import re

from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    confident: bool
    score: float
    reason: str


class ConfidenceChecker:
    """
    Validate whether retrieved evidence is sufficiently relevant
    to the user's question.

    This is a heuristic retrieval-confidence mechanism,
    not a calibrated probability.

    The checker deliberately avoids being overly strict because
    the reranker score is a relevance score rather than a
    probability.
    """

    STOP_WORDS = {
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
        "please",
        "tell",
        "me",
    }

    def __init__(
        self,
        minimum_score: float = 0.15,
        minimum_keyword_coverage: float = 0.30,
    ) -> None:

        self.minimum_score = minimum_score

        self.minimum_keyword_coverage = (
            minimum_keyword_coverage
        )

    # ==========================================================
    # TOKENIZATION
    # ==========================================================

    def _tokenize(
        self,
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+(?:\.[0-9]+)?\b",
                text.lower(),
            )
        )

    # ==========================================================
    # IMPORTANT QUESTION WORDS
    # ==========================================================

    def _important_words(
        self,
        question: str,
    ) -> set[str]:

        tokens = self._tokenize(
            question
        )

        return {
            token
            for token in tokens
            if token not in self.STOP_WORDS
        }

    # ==========================================================
    # EXTRACT YEARS
    # ==========================================================

    def _extract_years(
        self,
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"\b20\d{2}\b",
                text,
            )
        )

    # ==========================================================
    # KEYWORD COVERAGE
    # ==========================================================

    def _keyword_coverage(
        self,
        question: str,
        texts: list[str],
    ) -> float:

        question_words = (
            self._important_words(
                question
            )
        )

        if not question_words:
            return 1.0

        evidence_words = self._tokenize(
            " ".join(texts)
        )

        matched = (
            question_words.intersection(
                evidence_words
            )
        )

        return (
            len(matched)
            / len(question_words)
        )

    # ==========================================================
    # YEAR SUPPORT
    # ==========================================================

    def _year_supported(
        self,
        question: str,
        texts: list[str],
    ) -> bool:

        requested_years = (
            self._extract_years(
                question
            )
        )

        # No explicit year in question.
        if not requested_years:
            return True

        evidence_text = " ".join(
            texts
        )

        evidence_years = (
            self._extract_years(
                evidence_text
            )
        )

        return requested_years.issubset(
            evidence_years
        )

    # ==========================================================
    # MAIN CONFIDENCE CHECK
    # ==========================================================

    def check(
        self,
        question: str,
        reranked_results: list,
    ) -> ConfidenceResult:

        # ------------------------------------------------------
        # No retrieval results
        # ------------------------------------------------------

        if not reranked_results:

            return ConfidenceResult(
                confident=False,
                score=0.0,
                reason=(
                    "No relevant documents "
                    "were retrieved."
                ),
            )

        # ------------------------------------------------------
        # Top reranker score
        # ------------------------------------------------------

        top_score = float(
            reranked_results[0].get(
                "score",
                0.0,
            )
        )

        # ------------------------------------------------------
        # Collect evidence from top results
        # ------------------------------------------------------

        top_results = (
            reranked_results[:5]
        )

        texts: list[str] = []

        for candidate in top_results:

            original_result = candidate[
                "result"
            ]

            # --------------------------------------------------
            # BM25 result
            # --------------------------------------------------

            if isinstance(
                original_result,
                dict,
            ):

                chunk = original_result[
                    "chunk"
                ]

                texts.append(
                    chunk.text
                )

            # --------------------------------------------------
            # Qdrant result
            # --------------------------------------------------

            else:

                payload = (
                    original_result.payload
                )

                texts.append(
                    payload.get(
                        "text",
                        "",
                    )
                )

        # ------------------------------------------------------
        # Keyword coverage
        # ------------------------------------------------------

        keyword_coverage = (
            self._keyword_coverage(
                question=question,
                texts=texts,
            )
        )

        # ------------------------------------------------------
        # Year validation
        # ------------------------------------------------------

        year_supported = (
            self._year_supported(
                question=question,
                texts=texts,
            )
        )

        # ------------------------------------------------------
        # DEBUG INFORMATION
        # ------------------------------------------------------

        print("\n" + "=" * 70)
        print("CONFIDENCE CHECK")
        print("=" * 70)

        print(
            f"Top reranker score: "
            f"{top_score:.4f}"
        )

        print(
            f"Keyword coverage: "
            f"{keyword_coverage:.4f}"
        )

        print(
            f"Year supported: "
            f"{year_supported}"
        )

        print(
            f"Minimum score: "
            f"{self.minimum_score}"
        )

        print(
            f"Minimum keyword coverage: "
            f"{self.minimum_keyword_coverage}"
        )

        # ------------------------------------------------------
        # Score check
        # ------------------------------------------------------

        if top_score < self.minimum_score:

            return ConfidenceResult(
                confident=False,
                score=top_score,
                reason=(
                    "Top retrieval score is below "
                    "the confidence threshold."
                ),
            )

        # ------------------------------------------------------
        # Keyword check
        # ------------------------------------------------------

        if (
            keyword_coverage
            < self.minimum_keyword_coverage
        ):

            return ConfidenceResult(
                confident=False,
                score=top_score,
                reason=(
                    "Retrieved evidence does not "
                    "contain enough important terms "
                    "from the question."
                ),
            )

        # ------------------------------------------------------
        # Year check
        # ------------------------------------------------------

        if not year_supported:

            return ConfidenceResult(
                confident=False,
                score=top_score,
                reason=(
                    "The requested year is not "
                    "supported by the retrieved evidence."
                ),
            )

        # ------------------------------------------------------
        # Everything passed
        # ------------------------------------------------------

        return ConfidenceResult(
            confident=True,
            score=top_score,
            reason=(
                "Retrieved evidence satisfies "
                "the relevance, keyword, and "
                "year checks."
            ),
        )