"""Positive / filtered / no-evidence checks.  python -m evaluation.rag_evaluation
Also prints the best score per query so you can tune MIN_SCORE in config.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.retriever import retrieve_context

# (query, filters, expect_found, expected_doc_substring, expected_text_substring)
CASES = [
 ("What is the attendance requirement?", {}, True, "regulations", "75%"),
 ("How is SGPA calculated?", {}, True, "regulations", "credit-weighted"),
 ("What are my fifth semester subjects?", {"department": "CSE", "semester": 5}, True, "curriculum__CSE", "Machine Learning"),
 ("What are my fifth semester subjects?", {"department": "ECE", "semester": 5}, True, "curriculum__ECE", "VLSI"),
 ("When do end semester exams start?", {}, True, "calendar", "2 December"),
 ("What CGPA do I need for placements?", {}, True, "placement", "6.5"),
 ("Merit scholarship tuition waiver", {}, True, "scholarship", "50%"),
 ("Does the university give a free laptop to every student?", {}, False, None, None),
 ("What is the recipe for chocolate cake?", {}, False, None, None),
 ("Who won the football world cup?", {}, False, None, None),
]


def main():
    passed = 0
    for q, f, expect, doc, txt in CASES:
        r = retrieve_context(q, **f)
        top = r["sources"][0] if r["sources"] else None
        ok = r["found"] == expect
        if expect and ok:
            ok = any(doc in s["document"] for s in r["sources"]) and txt in r["context"]
        passed += ok
        print(f'{"PASS" if ok else "FAIL"} | found={r["found"]!s:5} | top={top and (top["document"], top["page"], top["score"])} | {q}')
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
