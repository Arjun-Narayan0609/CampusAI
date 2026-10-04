"""
Student Registry Module
Owned by: Member 3

Responsibilities:
- Manage access to synthetic student registry data (data/students/students.csv)
- Public contract:
    get_student_profile(student_id) -> dict | None
"""

import csv
import os
from typing import Optional

_STUDENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "students", "students.csv"
)


def get_student_profile(student_id: str) -> Optional[dict]:
    """Look up a student profile by ID from the synthetic registry CSV.

    Args:
        student_id: The student identifier (e.g. "STU001").

    Returns:
        A dict with keys ``student_id``, ``name``, ``department``,
        ``semester`` (int), ``year`` (int), ``cgpa`` (float) if found,
        or ``None`` if the student is not in the registry or the CSV
        cannot be read.
    """
    if not student_id:
        return None
    try:
        with open(_STUDENTS_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("student_id") == student_id:
                    return {
                        "student_id": row["student_id"],
                        "name": row.get("name", ""),
                        "department": row.get("department"),
                        "semester": int(row["semester"]) if row.get("semester") else None,
                        "year": int(row["year"]) if row.get("year") else None,
                        "cgpa": float(row["cgpa"]) if row.get("cgpa") else None,
                    }
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return None
