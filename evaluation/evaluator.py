from evaluation.dataset import (
    load_evaluation_dataset,
)
from evaluation.metrics import (
    f1_score,
    groundedness,
    overall_score,
    precision_at_k,
    recall_at_k,
)


class RAGEvaluator:
    """Evaluate the RAG pipeline against a test dataset."""

    def __init__(
        self,
        rag_pipeline,
        k: int = 5,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.k = k

    def _extract_retrieved_pages(
        self,
        sources: list[dict],
    ) -> list[str]:
        """Extract page numbers from retrieved sources."""

        pages = []

        for source in sources:
            page = source.get("page")

            if page is not None:
                pages.append(str(page))

        return pages

    def evaluate_case(
        self,
        case,
    ) -> dict:
        """Evaluate a single RAG evaluation case."""

        result = self.rag_pipeline.answer(
            case.question
        )

        actual_answer = result["answer"]

        sources = result.get(
            "sources",
            [],
        )

        retrieved_pages = (
            self._extract_retrieved_pages(
                sources
            )
        )

        expected_pages = [
            str(page)
            for page in case.relevant_pages
        ]

        precision = precision_at_k(
            retrieved_items=retrieved_pages,
            relevant_items=expected_pages,
            k=self.k,
        )

        recall = recall_at_k(
            retrieved_items=retrieved_pages,
            relevant_items=expected_pages,
            k=self.k,
        )

        retrieval_f1 = f1_score(
            precision,
            recall,
        )

        context = "\n".join(
            str(source)
            for source in sources
        )

        groundedness_score = groundedness(
            answer=actual_answer,
            context=context,
        )

        final_score = overall_score(
            retrieval_f1=retrieval_f1,
            groundedness_score=groundedness_score,
        )

        return {
            "case_id": case.case_id,
            "question": case.question,
            "category": case.category,
            "retrieved_pages": retrieved_pages[
                :self.k
            ],
            "relevant_pages": expected_pages,
            "precision_at_k": round(
                precision,
                3,
            ),
            "recall_at_k": round(
                recall,
                3,
            ),
            "f1_score": round(
                retrieval_f1,
                3,
            ),
            "groundedness": round(
                groundedness_score,
                3,
            ),
            "overall_score": round(
                final_score,
                3,
            ),
            "answer": actual_answer,
            "sources": sources,
        }

    def evaluate(
        self,
        dataset_path: str,
    ) -> list[dict]:
        """Evaluate all cases in the dataset."""

        cases = load_evaluation_dataset(
            dataset_path
        )

        results = []

        for case in cases:
            result = self.evaluate_case(
                case
            )

            results.append(result)

        return results