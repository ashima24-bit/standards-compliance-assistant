import json
from pathlib import Path
import rag_pipeline

EVAL_FILE = Path(__file__).parent.parent / "eval_questions.json"


def run_evaluation():
    if not EVAL_FILE.exists():
        print(f"No eval file found at {EVAL_FILE}.")
        return

    test_cases = json.loads(EVAL_FILE.read_text())
    total = len(test_cases)
    retrieval_hits = 0
    verified_count = 0

    print(f"Running {total} evaluation question(s)...\n")
    for case in test_cases:
        result = rag_pipeline.answer_question(case["question"])
        expected_keyword = case.get("expected_keyword", "").lower()

        retrieval_hit = any(
            expected_keyword in s["text"].lower() for s in result["sources"]
        ) if expected_keyword else False
        if retrieval_hit:
            retrieval_hits += 1
        if result["status"] == "Verified":
            verified_count += 1

        print(f"Q: {case['question']}")
        print(f"  Retrieval contained expected content: {retrieval_hit}")
        print(f"  Status: {result['status']}")
        print(f"  Answer: {result['answer'][:150]}...")
        print()

    print("=" * 50)
    print(f"Retrieval precision (expected content found): {retrieval_hits}/{total} "
          f"({100*retrieval_hits/total:.1f}%)")
    print(f"Answers marked Verified (passed groundedness check): {verified_count}/{total} "
          f"({100*verified_count/total:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
    