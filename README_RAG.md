# CampusAI - RAG module (Member 1)

## Setup
    pip install -r requirements_rag.txt
    python scripts/make_synthetic_docs.py      # creates data/documents/*.pdf (or drop in your own PDFs)
    python -m ingestion.ingest --rebuild       # chunk + embed + persist to data/chroma_db
    python -m evaluation.rag_evaluation        # sanity checks

## PDF naming (drives metadata)
    <category>__<department>__<semester>__<subject>__<year>.pdf
    ALL = applies to everyone, semester 0 = all semesters.

## Contract for Member 2
    from rag.retriever import retrieve_context
    r = retrieve_context("attendance requirement", department="CSE", semester=5)
    r["found"]    # False -> reply "information unavailable in the provided documents"
    r["context"]  # text to put in the LLM prompt
    r["sources"]  # [{document, category, page, score}] for citations

Rebuild the store any time documents change: `python -m ingestion.ingest --rebuild`.
Tune `MIN_SCORE` in `config.py` using the scores printed by the evaluation script.
