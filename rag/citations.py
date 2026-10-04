"""Turn raw hits into de-duplicated, display-ready sources."""


def build_sources(hits: list[dict]) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        m = h["metadata"]
        key = (m["document"], m["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "document": m["document"],
            "category": m["category"],
            "page": m["page"],
            "score": round(h["score"], 3),
        })
    return out


def format_source(s: dict) -> str:
    return f'{s["document"]} (p. {s["page"]})'
