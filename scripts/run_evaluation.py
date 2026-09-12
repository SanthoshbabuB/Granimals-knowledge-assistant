import json

from evaluation.evaluator import RAGEvaluator
from app.rag.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    evaluator = RAGEvaluator(
        rag_pipeline=pipeline,
        k=5,
    )

    results = evaluator.evaluate(
        "evaluation/rag_eval.jsonl"
    )

    print("\n" + "=" * 70)
    print("RAG EVALUATION RESULTS")
    print("=" * 70)

    for result in results:
        print(
            f"\n[{result['case_id']}] "
            f"{result['category']}"
        )

        print(
            f"Question: "
            f"{result['question']}"
        )

        print(
            f"Retrieved Pages: "
            f"{result['retrieved_pages']}"
        )

        print(
            f"Relevant Pages: "
            f"{result['relevant_pages']}"
        )

        print(
            f"Precision@5: "
            f"{result['precision_at_k']}"
        )

        print(
            f"Recall@5: "
            f"{result['recall_at_k']}"
        )

        print(
            f"F1@5: "
            f"{result['f1_score']}"
        )

        print(
            f"Groundedness: "
            f"{result['groundedness']}"
        )

        print(
            f"Overall Score: "
            f"{result['overall_score']}"
        )

    with open(
        "evaluation/results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to:")
    print("evaluation/results.json")


if __name__ == "__main__":
    main()