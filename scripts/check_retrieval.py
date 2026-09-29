"""Step 7: test retrieval BEFORE the LLM. Prints top chunks and checks the expected file.
    python -m scripts.check_retrieval
"""
import sys

from app import embeddings, vector_store
from app.config import get_settings

CASES = [  # (question, file that MUST appear in the results)
    ("How many annual leave days are available?", "leave_policy.txt"),
    ("How many vacation days do I get?", "leave_policy.txt"),
    ("How many days can I work from home?", "remote_work_policy.md"),
    ("What is the hotel limit for London?", "expense_policy.txt"),
    ("How often must I change my password?", "it_security_policy.txt"),
    ("Does the company pay for my health insurance?", "benefits.md"),
]

def main() -> int:
    failed = 0
    for question, expected in CASES:
        # raw search (no MIN_SCORE filter) so you can SEE the scores and tune the threshold
        hits = vector_store.search(embeddings.embed_query(question), 3)
        ok = any(h.payload["source"] == expected for h in hits)
        failed += not ok
        print(f"\nQ: {question}\n   expected: {expected}  ->  {'PASS' if ok else 'FAIL'}")
        for h in hits:
            keep = "keep" if h.score >= get_settings().min_score else "drop"
            print(f"   {h.score:.3f} [{keep}] {h.payload['source']}: "
                  f"{h.payload['text'][:60].replace(chr(10), ' ')}...")

    print(f"\n{len(CASES) - failed}/{len(CASES)} retrieval checks passed "
          f"(MIN_SCORE={get_settings().min_score})")
    return failed


if __name__ == "__main__":
    sys.exit(main())
