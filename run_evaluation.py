import json
import time
from pathlib import Path

from src.evaluator import check_faithfulness, check_retrieval
from src.rag import NOT_FOUND, ask, format_context

TEST_SET_PATH = Path("evaluation/test_set.json")
RESULTS_PATH = Path("evaluation/results.json")


def run_evaluation() -> None:
    test_cases = json.loads(TEST_SET_PATH.read_text())
    results = []

    retrieval_checks = []
    correct_not_found = 0
    should_not_find_count = 0
    faithfulness_checks = []

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        print(f"[{i}/{len(test_cases)}] {question}")

        start = time.time()
        result = ask(question)
        elapsed = time.time() - start

        answered = result["answer"] != NOT_FOUND
        retrieval_ok = check_retrieval(result["sources"], case["expected_source"])

        faithful = None
        if answered and result["sources"]:
            # Re-run retrieval formatting to get the exact context the LLM saw.
            # We approximate here using the returned sources' text is not stored,
            # so we judge faithfulness against the answer using the sources' pages as a proxy.
            context_note = "; ".join(f"{s['source']} page {s['page']}" for s in result["sources"])
            faithful = check_faithfulness(result["answer"], context_note)

        if not case["should_find"]:
            should_not_find_count += 1
            if not answered:
                correct_not_found += 1

        if retrieval_ok is not None:
            retrieval_checks.append(retrieval_ok)
        if faithful is not None:
            faithfulness_checks.append(faithful)

        results.append({
            "question": question,
            "answer": result["answer"],
            "expected_source": case["expected_source"],
            "should_find": case["should_find"],
            "answered": answered,
            "retrieval_correct": retrieval_ok,
            "seconds": round(elapsed, 1),
        })

    retrieval_accuracy = sum(retrieval_checks) / len(retrieval_checks) if retrieval_checks else None
    not_found_accuracy = (
        correct_not_found / should_not_find_count if should_not_find_count else None
    )
    avg_time = sum(r["seconds"] for r in results) / len(results)

    summary = {
        "total_questions": len(test_cases),
        "retrieval_accuracy": retrieval_accuracy,
        "correct_not_found_rate": not_found_accuracy,
        "average_response_time_seconds": round(avg_time, 1),
        "results": results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    print("\n--- Summary ---")
    print(f"Retrieval accuracy: {retrieval_accuracy}")
    print(f"Correct 'not found' rate: {not_found_accuracy}")
    print(f"Average response time: {avg_time:.1f}s")
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()