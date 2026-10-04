"""Deterministic SGPA / CGPA calculation (Member 3). No LLM, no floats in the arithmetic.

SGPA = sum(credits x grade_points) / sum(credits)           (failed courses count in the denominator)
CGPA = sum(all credit points)      / sum(all credits)        (credit-weighted across semesters)
Results are rounded to 2 decimals, ROUND_HALF_UP (6.125 -> 6.13, not banker's rounding).

The grade scale below is a SYNTHETIC demo scale. Edit GRADE_SCALES to match your real regulations.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

GRADE_SCALES: dict[str, dict[str, int]] = {
    "synthetic_10": {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "P": 4, "F": 0, "AB": 0},
    "vtu_10": {"S": 10, "A": 9, "B": 8, "C": 7, "D": 6, "E": 5, "F": 0},
}
DEFAULT_SCALE = "synthetic_10"


class GradeError(ValueError):
    """Invalid grade, credits, or input shape."""


def _q(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _s(x: Decimal) -> str:
    return format(x.normalize(), "f")


def _dec(value: Any, what: str) -> Decimal:
    try:
        d = Decimal(str(value))
    except Exception as e:
        raise GradeError(f"{what} must be a number, got {value!r}") from e
    if not d.is_finite():
        raise GradeError(f"{what} must be a finite number")
    return d


def _scale(name: str) -> dict[str, int]:
    try:
        return GRADE_SCALES[name]
    except KeyError:
        raise GradeError(f"Unknown grade scale {name!r}. Available: {', '.join(GRADE_SCALES)}") from None


def _course(item: Any) -> tuple[str, str, Any]:
    if isinstance(item, dict):
        if "grade" not in item or "credits" not in item:
            raise GradeError("Each course needs 'grade' and 'credits'")
        return str(item.get("subject", "")), item["grade"], item["credits"]
    if isinstance(item, (tuple, list)) and len(item) == 3:
        return str(item[0]), item[1], item[2]
    raise GradeError("Each course must be a dict {subject, grade, credits} or a (subject, grade, credits) tuple")


def _points(courses: list, scale: str) -> tuple[Decimal, Decimal, list[dict]]:
    table = _scale(scale)
    if not courses:
        raise GradeError("At least one course is required")
    total_pts = total_cr = Decimal(0)
    rows = []
    for item in courses:
        subject, grade, credits = _course(item)
        g = str(grade).strip().upper()
        if g not in table:
            raise GradeError(f"Unknown grade {grade!r} for scale '{scale}'. Valid: {', '.join(table)}")
        cr = _dec(credits, "credits")
        if cr <= 0:
            raise GradeError(f"Credits must be positive (got {credits!r})")
        pts = cr * table[g]
        total_pts, total_cr = total_pts + pts, total_cr + cr
        rows.append({"subject": subject, "grade": g, "credits": float(cr), "grade_points": table[g], "credit_points": float(pts)})
    return total_pts, total_cr, rows


def calculate_sgpa(courses: list, scale: str = DEFAULT_SCALE) -> dict:
    """courses: [{"subject": "DBMS", "grade": "A", "credits": 4}, ...]"""
    pts, cr, rows = _points(courses, scale)
    sgpa = _q(pts / cr)
    return {
        "tool": "calculate_sgpa",
        "sgpa": float(sgpa),
        "total_credits": float(cr),
        "credit_points": float(pts),
        "scale": scale,
        "courses": rows,
        "formula": f"SGPA = sum(credits x grade points) / sum(credits) = {_s(pts)} / {_s(cr)} = {sgpa}",
    }


def calculate_cgpa(semesters: list[dict], scale: str = DEFAULT_SCALE) -> dict:
    """Each semester: {"semester": 1, "courses": [...]}  OR  {"semester": 1, "sgpa": 8.2, "credits": 22}.
    If NO semester has credits, a plain average of SGPAs is used (weighting='equal')."""
    if not semesters:
        raise GradeError("At least one semester is required")
    items: list[dict] = []
    for i, sem in enumerate(semesters, start=1):
        if not isinstance(sem, dict):
            raise GradeError("Each semester must be a dict")
        label = sem.get("semester", i)
        if sem.get("courses"):
            pts, cr, _ = _points(sem["courses"], scale)
        elif "sgpa" in sem:
            sg = _dec(sem["sgpa"], "sgpa")
            if not 0 <= sg <= max(_scale(scale).values()):
                raise GradeError(f"SGPA {sem['sgpa']} is out of range for semester {label}")
            cr = _dec(sem["credits"], "credits") if sem.get("credits") is not None else None
            if cr is not None and cr <= 0:
                raise GradeError("Credits must be positive")
            pts = sg * cr if cr is not None else sg
        else:
            raise GradeError(f"Semester {label} needs 'courses' or 'sgpa'")
        items.append({"semester": label, "pts": pts, "credits": cr, "sgpa_given": sem.get("sgpa")})

    with_credits = [x for x in items if x["credits"] is not None]
    if with_credits and len(with_credits) != len(items):
        raise GradeError("Give credits for every semester, or for none of them")

    if with_credits:
        total_pts = sum((x["pts"] for x in items), Decimal(0))
        total_cr = sum((x["credits"] for x in items), Decimal(0))
        cgpa, weighting = _q(total_pts / total_cr), "credit"
        formula = f"CGPA = total credit points / total credits = {_s(total_pts)} / {_s(total_cr)} = {cgpa}"
        out_sems = [{"semester": x["semester"], "sgpa": float(_q(x["pts"] / x["credits"])), "credits": float(x["credits"])} for x in items]
        credits_out: float | None = float(total_cr)
    else:
        total = sum((x["pts"] for x in items), Decimal(0))
        cgpa, weighting = _q(total / len(items)), "equal"
        formula = f"CGPA = mean of SGPAs = {_s(total)} / {len(items)} = {cgpa}"
        out_sems = [{"semester": x["semester"], "sgpa": float(x["pts"]), "credits": None} for x in items]
        credits_out = None
    return {"tool": "calculate_cgpa", "cgpa": float(cgpa), "total_credits": credits_out, "weighting": weighting, "scale": scale, "semesters": out_sems, "formula": formula}
