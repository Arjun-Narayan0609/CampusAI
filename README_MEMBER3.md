# CampusAI - Member 3 (Academic Intelligence & Tools)

Student profile/state, deterministic SGPA/CGPA, study planner, internal calendar. **No LLM is used anywhere in this
package** - the LLM only decides *which* function to call and phrases the result.
All data in `data/students/students.csv` is synthetic demo data.

## Drop it into the repo
Copy these into the root of the shared repo (they match the agreed project structure; `student/` and `tools/` replace
the empty placeholder files, and `students.csv` replaces the 4-row placeholder - see 'Heads-up' below):
`student/`, `tools/`, `data/students/students.csv`, `tests/member3/`, `demo_member3.py`. Add `data/*.db` to `.gitignore` (the scheduler creates `data/campusai.db` on first use).

## Try it / test it
```
python demo_member3.py        # login -> SGPA/CGPA -> study plan -> "I cannot study Wednesday"
pip install pytest langchain-core
pytest tests/member3 -q       # 54 tests
```

## Interface for Member 2 (LangGraph) and Member 4 (Streamlit)
```python
from tools import (get_student_profile, calculate_sgpa, calculate_cgpa, create_study_plan, validate_plan,
                   get_schedule, add_session, complete_session, miss_session, modify_session,
                   modify_schedule, reschedule_unavailable_day, resolve_weekday)
```
| Function | Returns |
|---|---|
| `get_student_profile(student_id)` | dict: name, department, semester, year, cgpa, subjects, exam_dates, available_hours_per_day, study_start, rest_day. Raises `StudentNotFoundError` for unknown IDs |
| `calculate_sgpa(courses)` | `{"sgpa", "total_credits", "credit_points", "courses", "formula"}`; course = `{"subject","grade","credits"}` |
| `calculate_cgpa(semesters)` | `{"cgpa", "weighting", "semesters", "formula"}`; semester = `{"courses":[...]}` or `{"sgpa","credits"}` |
| `create_study_plan(student_id, today=None, persist=True, blocked_dates=None)` | `{"plan":[sessions], "per_subject", "feasible", "warnings", ...}`; saves to the calendar by default |
| `validate_plan(plan)` | list of problems (empty = valid) - **use this in the reviewer node** |
| `get_schedule(student_id, start_date=None, end_date=None, status=None)` | list of sessions |
| `reschedule_unavailable_day(student_id, date)` | "I cannot study Wednesday": moves sessions if there is room, otherwise re-plans around that day. Check `strategy` and `sessions_dropped` |
| `resolve_weekday("Wednesday")` | `date` of the next Wednesday |

Errors you can catch: `GradeError`, `ScheduleError` (`ScheduleConflictError`, `SessionNotFoundError`), `StudentNotFoundError`.

### LangChain tools (for Arjun)
```python
from tools.langchain_tools import get_member3_tools
llm_with_tools = llm.bind_tools(get_member3_tools())
```
Six tools: `sgpa_tool`, `cgpa_tool`, `student_profile_tool`, `study_plan_tool`, `schedule_tool`, `reschedule_day_tool`.
Each returns `{"ok": True, ...}` or `{"ok": False, "error": "..."}` - **never raises**, so the reviewer can always check
that a real tool result exists. **Inject `student_id` from `AgentState`**; don't let the LLM supply it.

## Rules the code enforces
- SGPA/CGPA use `Decimal` and round half-up to 2 places (6.125 -> 6.13). F grades count in the denominator.
- Grade scale is a synthetic demo scale (`GRADE_SCALES` in `tools/cgpa.py`); edit it to match your regulations.
- Sessions: 15-240 min, between 05:00 and 23:00, no overlaps, per-day limit = the student's available hours.
- Planner: study days are tomorrow up to the day before the last exam; skips the rest day; nearer exams get more
  blocks, every subject keeps getting covered; warns if a subject gets under 4 h. Same input gives the same plan.
- Re-planning replaces still-`planned` sessions only; `completed`/`missed` history is kept.

## Config
`CAMPUSAI_DB` (scheduler SQLite path) and `CAMPUSAI_STUDENTS_CSV` (registry path) override the defaults.

## Heads-up for the team
- `data/students/students.csv` is replaced by a richer synthetic file (24CSE001 etc., with subjects, exam dates and
  study hours - the plan's demo logs in as 24CSE001). The registry also still works with the old 6-column file
  (STU001...), subjects/exam dates simply come back empty. Members 1 and 4 should agree on which IDs the demo uses.
- `create_study_plan` is importable from both `tools.planner` and `tools.scheduler` (the skeleton's docstring put it
  in `scheduler.py`).
- Not built: persisting *dynamic* student preferences (`student/state.py` is still the empty placeholder).
- Add `data/*.db` to `.gitignore` (the scheduler creates `data/campusai.db`).
