"""Metadata from filenames.

Convention:  <category>__<department>__<sem>__<subject>__<year>.pdf
Use ALL for "applies to everyone", 0 for no semester. Trailing parts optional.
Examples:
  regulations__ALL__0.pdf
  curriculum__CSE__5.pdf
  handbook__CSE__5__Machine-Learning.pdf
  calendar__ALL__0__ALL__2026-27.pdf
"""
from pathlib import Path
from config import ALL


def metadata_from_filename(path: Path) -> dict:
    parts = path.stem.split("__")
    get = lambda i, default: parts[i] if len(parts) > i and parts[i] else default
    try:
        semester = int(get(2, 0))
    except ValueError:
        semester = 0
    dept = get(1, ALL)
    return {
        "document": path.name,
        "category": get(0, "general").lower(),
        "department": ALL if dept.upper() == "ALL" else dept.upper(),
        "semester": semester,                       # 0 = all semesters
        "subject": get(3, ALL).replace("-", " "),
        "academic_year": get(4, "2026-27"),
    }
