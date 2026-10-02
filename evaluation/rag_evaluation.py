"""RAG checks.  python -m evaluation.rag_evaluation
Prints the top score per query so you can tune MIN_SCORE in config.py.

Case keys: q, filters, found (expected), doc (substring expected among sources),
           text (substring expected in context), not_doc (substring that must NOT appear in sources).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIN_SCORE
from rag.retriever import retrieve_context

POS = lambda q, doc, text, **f: dict(q=q, filters=f, found=True, doc=doc, text=text)
NEG = lambda q, **f: dict(q=q, filters=f, found=False)

CASES = [
 # --- core positives ---
 POS("What is the attendance requirement?", "regulations", "75%"),
 POS("How is SGPA calculated?", "regulations", "credit-weighted"),
 POS("When do end semester exams start?", "calendar", "2 December"),
 POS("What CGPA do I need for placements?", "placement", "6.5"),
 POS("Merit scholarship tuition waiver", "scholarship", "50%"),
 POS("How many days do I have to apply for re-evaluation?", "examination", "7 days"),
 POS("How long is the mandatory internship?", "internship", "6 weeks"),
 POS("What are the library timings?", "faq", "8:30"),
 # --- department / semester / subject filters ---
 POS("What are my fifth semester subjects?", "curriculum__CSE", "Machine Learning", department="CSE", semester=5),
 POS("What are my fifth semester subjects?", "curriculum__ECE", "VLSI", department="ECE", semester=5),
 dict(q="What are my fifth semester subjects?", filters=dict(department="ECE", semester=5),
      found=True, not_doc="CSE"),                                   # ECE student must never see CSE docs
 POS("How is Machine Learning evaluated?", "handbook__CSE", "20%", department="CSE", semester=5, subject="Machine Learning"),
 POS("What is the attendance requirement?", "regulations", "75%", department="CSE", semester=5),  # ALL docs still reachable
 # --- messy input ---
 POS("attendence requirment", "regulations", "75%"),                 # typos
 POS("attendance", "regulations", "75%"),                            # one word
 POS("   WHAT   IS THE   ATTENDANCE REQUIREMENT???  ", "regulations", "75%"),
 # --- needs two documents ---
 POS("What CGPA is needed for placement and for a merit scholarship?", "placement", "6.5"),
 # --- no evidence: must say unavailable ---
 NEG("Does the university give a free laptop to every student?"),
 NEG("What is the recipe for chocolate cake?"),
 NEG("Who won the football world cup?"),
 NEG("What is the hostel mess menu on Sunday?"),
 NEG("How do I apply for a visa to Germany?"),
 NEG(""),                                                            # empty query
 NEG("?"),
]


def check(case) -> tuple[bool, dict]:
    r = retrieve_context(case["q"], **case["filters"])
    ok = r["found"] == case["found"]
    docs = [s["document"] for s in r["sources"]]
    if ok and case["found"]:
        if "doc" in case:
            ok = any(case["doc"] in d for d in docs)
        if ok and "text" in case:
            ok = case["text"] in " ".join(r["context"].split())   # ignore PDF line wraps
        if ok and "not_doc" in case:
            ok = not any(case["not_doc"] in d for d in docs)
    return ok, r


def store_check() -> bool:
    """Store must hold exactly the chunks the documents produce (no duplicates / stale chunks)."""
    from ingestion.ingest import build_chunks
    from rag.vectorstore import get_collection
    n_store, n_docs = get_collection().count(), len(build_chunks())
    ok = n_store == n_docs
    print(f'{"PASS" if ok else "FAIL"} | store has {n_store} chunks, documents produce {n_docs}'
          + ("" if ok else "  -> run: python -m ingestion.ingest --rebuild"))
    return ok


def main():
    print(f"MIN_SCORE = {MIN_SCORE}\n")
    passed = 0
    failed = []
    for c in CASES:
        ok, r = check(c)
        passed += ok
        top = (r["sources"][0]["document"][:28], r["sources"][0]["page"], r["sources"][0]["score"]) if r["sources"] else f'best={r["best_score"]}'
        label = f'{c["q"].strip()[:55]!r} {c["filters"] or ""}'
        print(f'{"PASS" if ok else "FAIL"} | found={r["found"]!s:5} | top={top} | {label}')
        if not ok:
            failed.append(c["q"])
    store_ok = store_check()
    total = len(CASES) + 1
    print(f"\n{passed + store_ok}/{total} passed")
    if failed:
        print("Failed queries:", *failed, sep="\n  - ")
        print("\nTip: a no-evidence query that returned found=True -> raise MIN_SCORE; "
              "a real query that returned found=False -> lower it.")


if __name__ == "__main__":
    main()
