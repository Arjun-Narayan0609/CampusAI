"""LangChain wrappers so Member 2 (LangGraph) can bind Member 3's functions as tools:

    from tools.langchain_tools import get_member3_tools
    llm_with_tools = llm.bind_tools(get_member3_tools())

student_id is a normal argument here, but the graph should INJECT it from AgentState rather than let the
LLM guess it. Every tool returns a dict: {"ok": True, ...result} or {"ok": False, "error": "..."} so the
reviewer node can always tell whether a real tool result exists.
"""
from __future__ import annotations

import functools
from datetime import date
from typing import Any

from student.registry import RegistryError, StudentNotFoundError
from tools import cgpa as _cgpa
from tools import planner as _planner
from tools import scheduler as _sch
from student import registry as _registry

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:  # keeps the rest of Member 3's code usable without LangChain installed
    tool = None

_EXPECTED = (_cgpa.GradeError, _sch.ScheduleError, StudentNotFoundError, RegistryError)


def _safe(fn):
    @functools.wraps(fn)  # keep docstring + signature: LangChain builds the tool schema from them
    def wrapper(*a, **k) -> dict[str, Any]:
        try:
            out = fn(*a, **k)
            return {"ok": True, **(out if isinstance(out, dict) else {"result": out})}
        except _EXPECTED as e:
            return {"ok": False, "error": str(e.args[0] if e.args else e)}
    return wrapper


if tool is not None:

    class CourseGrade(BaseModel):
        subject: str = Field("", description="Course name")
        grade: str = Field(description="Letter grade, e.g. O, A+, A, B+, B, C, P, F")
        credits: float = Field(gt=0, description="Course credits")

    class SemesterResult(BaseModel):
        semester: int | None = None
        sgpa: float | None = Field(None, description="Use with credits when courses are not available")
        credits: float | None = None
        courses: list[CourseGrade] | None = None

    @tool
    @_safe
    def sgpa_tool(courses: list[CourseGrade], scale: str = "synthetic_10") -> dict:
        """Calculate SGPA exactly from a list of courses (grade + credits). Always use this for SGPA; never do the arithmetic yourself."""
        return _cgpa.calculate_sgpa([c.model_dump() if hasattr(c, "model_dump") else c for c in courses], scale)

    @tool
    @_safe
    def cgpa_tool(semesters: list[SemesterResult], scale: str = "synthetic_10") -> dict:
        """Calculate CGPA exactly across semesters. Each semester has courses, or sgpa + credits."""
        return _cgpa.calculate_cgpa([s.model_dump(exclude_none=True) if hasattr(s, "model_dump") else s for s in semesters], scale)

    @tool
    @_safe
    def student_profile_tool(student_id: str) -> dict:
        """Get the student's department, semester, CGPA, subjects, exam dates and study preferences."""
        return {"profile": _registry.get_student_profile(student_id)}

    @tool
    @_safe
    def study_plan_tool(student_id: str) -> dict:
        """Create (and save) a study plan from the student's subjects, exam dates and daily hours. Deterministic."""
        plan = _planner.create_study_plan(student_id)
        return {**plan, "issues": _planner.validate_plan(plan)}

    @tool
    @_safe
    def schedule_tool(student_id: str, start_date: str | None = None, end_date: str | None = None) -> dict:
        """List the student's study sessions (optionally between two YYYY-MM-DD dates)."""
        return {"sessions": _sch.get_schedule(student_id, start_date, end_date)}

    @tool
    @_safe
    def reschedule_day_tool(student_id: str, day: str, move_to_date: str | None = None) -> dict:
        """The student cannot study on `day` (YYYY-MM-DD or a weekday name like 'Wednesday'). Moves that day's sessions."""
        try:
            d: date | str = date.fromisoformat(day)
        except ValueError:
            d = _sch.resolve_weekday(day)
        if move_to_date:
            return _sch.modify_schedule(student_id, d, move_to_date)
        return _planner.reschedule_unavailable_day(student_id, d)


def get_member3_tools() -> list:
    if tool is None:
        raise ImportError("langchain-core is not installed. Run: pip install langchain-core")
    return [sgpa_tool, cgpa_tool, student_profile_tool, study_plan_tool, schedule_tool, reschedule_day_tool]
