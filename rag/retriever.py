"""Public interface for Member 2:  retrieve_context(query, department=None, semester=None, subject=None)

Always returns the SAME shape:
{
  "found":   bool,          # False => agent must answer "information unavailable"
  "context": str,           # chunks joined, each prefixed with [document p.N]
  "sources": [{"document","category","page","score"}],
  "query":   str,
  "filters": {...}
}
"""
from config import TOP_K, MIN_SCORE, ALL
from rag.embeddings import embed
from rag.vectorstore import get_collection
from rag.citations import build_sources


def _build_where(department, semester, subject):
    conds = []
    if department:
        conds.append({"department": {"$in": [department.upper(), ALL]}})
    if semester:
        conds.append({"semester": {"$in": [int(semester), 0]}})
    if subject:
        conds.append({"subject": {"$in": [subject, ALL]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _empty(query, filters):
    return {"found": False, "context": "", "sources": [], "query": query, "filters": filters}


def retrieve_context(query: str, department: str | None = None,
                     semester: int | None = None, subject: str | None = None,
                     k: int = TOP_K) -> dict:
    filters = {"department": department, "semester": semester, "subject": subject}
    if not query or not query.strip():
        return _empty(query, filters)

    col = get_collection()
    if col.count() == 0:
        return _empty(query, filters)

    res = col.query(
        query_embeddings=embed([query]),
        n_results=k,
        where=_build_where(department, semester, subject),
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {"text": d, "metadata": m, "score": 1.0 - dist}   # cosine distance -> similarity
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    if not hits or hits[0]["score"] < MIN_SCORE:           # grounding rule: weak evidence = none
        return _empty(query, filters)

    hits = [h for h in hits if h["score"] >= MIN_SCORE]
    context = "\n\n".join(f'[{h["metadata"]["document"]} p.{h["metadata"]["page"]}]\n{h["text"]}' for h in hits)
    return {"found": True, "context": context, "sources": build_sources(hits),
            "query": query, "filters": filters}
