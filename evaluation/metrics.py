from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(
    retrieved_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K =
        relevant retrieved items / total retrieved items at K
    """
    if k <= 0:
        return 0.0

    retrieved = list(retrieved_items)[:k]
    relevant = set(relevant_items)

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        1
        for item in retrieved
        if item in relevant
    )

    return relevant_retrieved / len(retrieved)


def recall_at_k(
    retrieved_items: Sequence[str],
    relevant_items: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K =
        relevant retrieved items / total relevant items
    """
    if k <= 0:
        return 0.0

    relevant = set(relevant_items)

    if not relevant:
        return 0.0

    retrieved = set(
        list(retrieved_items)[:k]
    )

    relevant_retrieved = len(
        retrieved.intersection(relevant)
    )

    return relevant_retrieved / len(relevant)


def f1_score(
    precision: float,
    recall: float,
) -> float:
    """
    Calculate F1 score from precision and recall.
    """
    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / (precision + recall)


def groundedness(
    answer: str,
    context: str,
) -> float:
    """
    Estimate how much of the answer is supported
    by the retrieved context.

    This is a lightweight lexical baseline, not
    an LLM-based faithfulness evaluator.
    """
    if not answer.strip():
        return 0.0

    answer_words = set(
        answer.lower().split()
    )

    context_words = set(
        context.lower().split()
    )

    if not answer_words:
        return 0.0

    grounded_words = answer_words.intersection(
        context_words
    )

    return len(grounded_words) / len(answer_words)


def overall_score(
    retrieval_f1: float,
    groundedness_score: float,
) -> float:
    """
    Calculate a combined evaluation score.

    Retrieval quality has a higher weight because
    retrieval is the foundation of the RAG pipeline.
    """
    return (
        0.6 * retrieval_f1
        + 0.4 * groundedness_score
    )