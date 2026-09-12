from dataclasses import dataclass


@dataclass
class EvaluationCase:
    case_id: str
    question: str
    expected_answer: str
    relevant_pages: list[int]
    category: str


def load_evaluation_dataset(
    file_path: str,
) -> list[EvaluationCase]:
    import json

    cases = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            cases.append(
                EvaluationCase(
                    case_id=data["case_id"],
                    question=data["question"],
                    expected_answer=data[
                        "expected_answer"
                    ],
                    relevant_pages=data.get(
                        "relevant_pages",
                        [],
                    ),
                    category=data["category"],
                )
            )

    return cases