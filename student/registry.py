"""Student registry (Member 3). The CSV is the source of truth for a student's profile.

Contract:  get_student_profile(student_id) -> dict   (JSON-friendly, safe to put in LangGraph state)
All data in data/students/students.csv is SYNTHETIC / demo data.
"""
from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# subjects / exam_dates (and the study-preference columns) are optional, so a minimal 6-column CSV also works.
REQUIRED = ["student_id", "name", "department", "semester", "year", "cgpa"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class RegistryError(ValueError):
    """The registry file or one of its rows is malformed."""


class StudentNotFoundError(LookupError):
    """No such student ID in the registry (no fake profile is ever created)."""


def csv_path() -> Path:
    return Path(os.environ.get("CAMPUSAI_STUDENTS_CSV", ROOT / "data" / "students" / "students.csv"))


def normalize_id(student_id: str) -> str:
    return str(student_id or "").strip().upper()


def _rows() -> list[dict[str, str]]:
    path = csv_path()
    if not path.is_file():
        raise RegistryError(f"Student registry not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:  # utf-8-sig: tolerate the BOM Excel adds
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise RegistryError(f"students.csv is missing columns: {', '.join(missing)}")
        return list(reader)


def _parse(row: dict[str, str]) -> dict:
    sid = row["student_id"].strip().upper()
    try:
        semester, year = int(row["semester"]), int(row["year"])
        cgpa = float(row["cgpa"])
        hours = float(row.get("available_hours_per_day") or 3)
        block = int(row.get("block_minutes") or 60)
        subjects = [s.strip() for s in (row.get("subjects") or "").split("|") if s.strip()]
        exams: dict[str, str] = {}
        for pair in filter(None, (p.strip() for p in (row.get("exam_dates") or "").split("|"))):
            subject, _, iso = pair.rpartition(":")
            exams[subject.strip()] = date.fromisoformat(iso.strip()).isoformat()
    except ValueError as e:
        raise RegistryError(f"Bad data for {sid}: {e}") from e
    if not 1 <= semester <= 8 or not 0 <= cgpa <= 10 or not 0 < hours <= 12 or not 15 <= block <= 180:
        raise RegistryError(f"Out-of-range value for {sid}")
    unknown = set(exams) - set(subjects)
    if unknown:
        raise RegistryError(f"{sid}: exam date given for unknown subject(s): {', '.join(sorted(unknown))}")
    rest = (row.get("rest_day") or "").strip().lower() or None
    if rest and rest not in WEEKDAYS:
        raise RegistryError(f"{sid}: invalid rest_day '{rest}'")
    return {
        "student_id": sid,
        "name": row["name"].strip(),
        "department": row["department"].strip(),
        "semester": semester,
        "year": year,
        "cgpa": cgpa,
        "subjects": subjects,
        "exam_dates": exams,  # {subject: "YYYY-MM-DD"}
        "available_hours_per_day": hours,
        "study_start": (row.get("study_start") or "18:00").strip(),
        "rest_day": rest,
        "block_minutes": block,
    }


def get_student_profile(student_id: str) -> dict:
    """Return the student's profile dict, or raise StudentNotFoundError."""
    sid = normalize_id(student_id)
    for row in _rows():
        if row["student_id"].strip().upper() == sid:
            return _parse(row)
    raise StudentNotFoundError(f"Unknown student ID: {student_id!r}")


def list_student_ids() -> list[str]:
    return [r["student_id"].strip().upper() for r in _rows()]


def validate_registry() -> int:
    """Parse every row; raises RegistryError on the first bad one. Returns the row count."""
    rows = _rows()
    for r in rows:
        _parse(r)
    return len(rows)
