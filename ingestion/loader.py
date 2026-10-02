"""PDF loading + cleaning (PyMuPDF)."""
import re
from collections import Counter
from pathlib import Path
import pymupdf  # PyMuPDF


# Always-strip lines (covers 1-2 page docs where repetition can't be detected).
STRIP = [re.compile(r"^\s*(.*[-|]\s*)?page\s+\d+(\s+of\s+\d+)?\s*$", re.I),
         re.compile(r"SYNTHETIC DEMO DATA", re.I)]


def _norm(line: str) -> str:
    # normalise digits so "Page 1" and "Page 2" count as the same header/footer
    return re.sub(r"\d+", "#", line.strip().lower())


def load_pdf(path: Path) -> list[dict]:
    """Return [{'page': 1-based int, 'text': cleaned str}, ...]."""
    with pymupdf.open(path) as pdf:
        raw = [p.get_text("text") for p in pdf]

    # Repeated header/footer lines: present on >=50% of pages (only when 3+ pages).
    counts = Counter()
    for t in raw:
        counts.update({_norm(l) for l in t.splitlines() if l.strip()})
    threshold = max(2, len(raw) // 2) if len(raw) >= 3 else 10**9
    repeated = {l for l, c in counts.items() if c >= threshold}

    pages = []
    for i, t in enumerate(raw, start=1):
        lines = [l for l in t.splitlines()
                 if _norm(l) not in repeated and not any(r.search(l) for r in STRIP)]
        text = "\n".join(lines)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages
