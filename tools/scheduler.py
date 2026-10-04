"""Internal calendar backed by SQLite (Member 3). No Google Calendar / OAuth needed.

Contracts for the agent / UI:
  add_session, get_schedule, complete_session, miss_session, modify_session   (single-session ops)
  modify_schedule(student_id, unavailable_date, ...)  -> moves a whole day ("I cannot study Wednesday")
  resolve_weekday("wednesday") -> date                (turn a weekday name from the user into a date)

Rules enforced on every write: duration 15-240 min, inside 05:00-23:00, no overlapping sessions,
and (optionally) a per-day time limit. Missed sessions don't block time; they can be re-planned.
DB location: env CAMPUSAI_DB (default data/campusai.db).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
MIN_DURATION_MIN, MAX_DURATION_MIN = 15, 240
DAY_START_MIN, DAY_END_MIN = 5 * 60, 23 * 60
STATUSES = ("planned", "completed", "missed")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id   TEXT NOT NULL,
  subject      TEXT NOT NULL,
  session_date TEXT NOT NULL,
  start_min    INTEGER NOT NULL,
  duration_min INTEGER NOT NULL,
  status       TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned','completed','missed')),
  note         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_student_date ON sessions (student_id, session_date);
"""


class ScheduleError(ValueError):
    """A schedule rule was violated (bad duration, outside hours, daily limit...)."""


class ScheduleConflictError(ScheduleError):
    """The session overlaps an existing one."""


class SessionNotFoundError(ScheduleError):
    """No such session (or it belongs to another student)."""


# ------------------------------------------------------------------ helpers
def db_path() -> Path:
    return Path(os.environ.get("CAMPUSAI_DB", ROOT / "data" / "campusai.db"))


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _d(value: date | str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as e:
        raise ScheduleError(f"Invalid date {value!r}; use YYYY-MM-DD") from e


def to_minutes(value: str | int) -> int:
    """'18:30' -> 1110. Ints pass through."""
    if isinstance(value, int):
        return value
    try:
        h, m = str(value).split(":")
        h, m = int(h), int(m)
    except ValueError as e:
        raise ScheduleError(f"Invalid time {value!r}; use HH:MM") from e
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ScheduleError(f"Invalid time {value!r}")
    return h * 60 + m


def fmt_minutes(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _out(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"], "student_id": r["student_id"], "subject": r["subject"], "date": r["session_date"],
        "start_time": fmt_minutes(r["start_min"]), "end_time": fmt_minutes(r["start_min"] + r["duration_min"]),
        "duration_min": r["duration_min"], "status": r["status"], "note": r["note"],
    }


def resolve_weekday(name: str, today: date | None = None, include_today: bool = True) -> date:
    """'Wednesday' -> the next Wednesday (today counts when include_today)."""
    key = name.strip().lower()
    if key not in WEEKDAYS:
        raise ScheduleError(f"Unknown weekday {name!r}")
    today = today or date.today()
    ahead = (WEEKDAYS.index(key) - today.weekday()) % 7
    if ahead == 0 and not include_today:
        ahead = 7
    return today + timedelta(days=ahead)


def _validate(con: sqlite3.Connection, student_id: str, day: date, start: int, dur: int,
              daily_limit_min: int | None, exclude_id: int | None = None) -> None:
    if not MIN_DURATION_MIN <= dur <= MAX_DURATION_MIN:
        raise ScheduleError(f"Duration must be {MIN_DURATION_MIN}-{MAX_DURATION_MIN} minutes (got {dur})")
    if start < DAY_START_MIN or start + dur > DAY_END_MIN:
        raise ScheduleError(f"Session must lie between {fmt_minutes(DAY_START_MIN)} and {fmt_minutes(DAY_END_MIN)}")
    rows = con.execute(
        "SELECT * FROM sessions WHERE student_id=? AND session_date=? AND status!='missed' AND id IS NOT ?",
        (student_id, day.isoformat(), exclude_id),
    ).fetchall()
    for r in rows:
        if start < r["start_min"] + r["duration_min"] and r["start_min"] < start + dur:
            raise ScheduleConflictError(
                f"Overlaps '{r['subject']}' on {day} ({fmt_minutes(r['start_min'])}-{fmt_minutes(r['start_min'] + r['duration_min'])})"
            )
    if daily_limit_min is not None and sum(r["duration_min"] for r in rows) + dur > daily_limit_min:
        raise ScheduleError(f"Daily study limit of {daily_limit_min} minutes would be exceeded on {day}")


def _fetch(con: sqlite3.Connection, session_id: int, student_id: str | None) -> sqlite3.Row:
    r = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if r is None or (student_id is not None and r["student_id"] != student_id.strip().upper()):
        raise SessionNotFoundError(f"Session {session_id} not found")
    return r


# --------------------------------------------------------------- public API
def add_session(student_id: str, subject: str, session_date: date | str, start_time: str | int,
                duration_min: int, note: str = "", daily_limit_min: int | None = None) -> dict:
    sid = student_id.strip().upper()
    if not sid or not subject.strip():
        raise ScheduleError("student_id and subject are required")
    day, start = _d(session_date), to_minutes(start_time)
    with _conn() as con:
        _validate(con, sid, day, start, int(duration_min), daily_limit_min)
        cur = con.execute(
            "INSERT INTO sessions (student_id, subject, session_date, start_min, duration_min, note) VALUES (?,?,?,?,?,?)",
            (sid, subject.strip(), day.isoformat(), start, int(duration_min), note),
        )
        return _out(_fetch(con, cur.lastrowid, sid))


def get_schedule(student_id: str, start_date: date | str | None = None, end_date: date | str | None = None,
                 status: str | None = None) -> list[dict]:
    if status is not None and status not in STATUSES:
        raise ScheduleError(f"status must be one of {STATUSES}")
    sql, args = "SELECT * FROM sessions WHERE student_id=?", [student_id.strip().upper()]
    if start_date:
        sql, args = sql + " AND session_date>=?", args + [_d(start_date).isoformat()]
    if end_date:
        sql, args = sql + " AND session_date<=?", args + [_d(end_date).isoformat()]
    if status:
        sql, args = sql + " AND status=?", args + [status]
    with _conn() as con:
        return [_out(r) for r in con.execute(sql + " ORDER BY session_date, start_min, id", args)]


def _set_status(session_id: int, status: str, student_id: str | None) -> dict:
    with _conn() as con:
        _fetch(con, session_id, student_id)
        con.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
        return _out(_fetch(con, session_id, None))


def complete_session(session_id: int, student_id: str | None = None) -> dict:
    return _set_status(session_id, "completed", student_id)


def miss_session(session_id: int, student_id: str | None = None) -> dict:
    return _set_status(session_id, "missed", student_id)


def modify_session(session_id: int, new_date: date | str | None = None, new_start_time: str | int | None = None,
                   new_duration_min: int | None = None, student_id: str | None = None,
                   daily_limit_min: int | None = None) -> dict:
    if new_date is None and new_start_time is None and new_duration_min is None:
        raise ScheduleError("Nothing to change")
    with _conn() as con:
        r = _fetch(con, session_id, student_id)
        if r["status"] == "completed":
            raise ScheduleError("Completed sessions can't be modified")
        day = _d(new_date) if new_date else date.fromisoformat(r["session_date"])
        start = to_minutes(new_start_time) if new_start_time is not None else r["start_min"]
        dur = int(new_duration_min) if new_duration_min is not None else r["duration_min"]
        _validate(con, r["student_id"], day, start, dur, daily_limit_min, exclude_id=session_id)
        # Re-planning a missed session puts it back on the calendar as planned.
        con.execute("UPDATE sessions SET session_date=?, start_min=?, duration_min=?, status='planned' WHERE id=?",
                    (day.isoformat(), start, dur, session_id))
        return _out(_fetch(con, session_id, None))


def delete_session(session_id: int, student_id: str | None = None) -> None:
    with _conn() as con:
        _fetch(con, session_id, student_id)
        con.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def clear_planned(student_id: str, from_date: date | str | None = None) -> int:
    """Delete still-'planned' sessions (history of completed/missed is kept). Used when re-planning."""
    sql, args = "DELETE FROM sessions WHERE student_id=? AND status='planned'", [student_id.strip().upper()]
    if from_date:
        sql, args = sql + " AND session_date>=?", args + [_d(from_date).isoformat()]
    with _conn() as con:
        return con.execute(sql, args).rowcount


def find_free_slot(student_id: str, day: date | str, duration_min: int, preferred_start: str | int = "18:00",
                   daily_limit_min: int | None = None) -> int | None:
    """First start minute (preferred time first, then every 15 min) where the session fits, else None."""
    sid, d = student_id.strip().upper(), _d(day)
    preferred = to_minutes(preferred_start)
    with _conn() as con:
        for start in [preferred, *range(DAY_START_MIN, DAY_END_MIN - duration_min + 1, 15)]:
            try:
                _validate(con, sid, d, start, duration_min, daily_limit_min)
                return start
            except ScheduleConflictError:
                continue
            except ScheduleError:
                if start == preferred:
                    continue  # preferred time outside hours; keep scanning
                return None  # daily limit reached
    return None


def _relocate(s: dict, prof: dict, day: date, move_to_date: date | str | None, limit: int) -> tuple[dict | None, str]:
    """Try to move one session off `day`. Returns (moved_session | None, reason_if_failed)."""
    from student.registry import WEEKDAYS as _WD

    exam = prof["exam_dates"].get(s["subject"])
    deadline = date.fromisoformat(exam) - timedelta(days=1) if exam else None
    candidates = [_d(move_to_date)] if move_to_date else [day + timedelta(days=i) for i in range(1, 15)]
    reason = "No free slot found in the next 14 days"
    for cand in candidates:
        if cand == day:
            continue
        if deadline and cand > deadline:
            return None, f"No free day before the {s['subject']} exam on {exam}"
        if not move_to_date and prof["rest_day"] and _WD[cand.weekday()] == prof["rest_day"]:
            continue
        start = find_free_slot(prof["student_id"], cand, s["duration_min"], s["start_time"], limit)
        if start is None:
            reason = f"{cand} has no free slot"
            continue
        return modify_session(s["id"], cand, start, daily_limit_min=limit), ""
    return None, reason


def modify_schedule(student_id: str, unavailable_date: date | str, move_to_date: date | str | None = None,
                    daily_limit_min: int | None = None) -> dict:
    """'I cannot study on <date>': move every planned session that day to another day.

    With move_to_date, all sessions go there. Otherwise each session goes to the first day within the next
    14 that has room, skipping the student's rest day and never landing on/after that subject's exam date.
    Returns {"moved": [{"from", "to"}], "failed": [{"session", "reason"}], "schedule": [planned sessions]}.
    """
    from student.registry import get_student_profile

    prof = get_student_profile(student_id)
    limit = daily_limit_min or int(prof["available_hours_per_day"] * 60)
    day = _d(unavailable_date)
    moved, failed = [], []
    for s in get_schedule(prof["student_id"], day, day, status="planned"):
        new, reason = _relocate(s, prof, day, move_to_date, limit)
        if new:
            moved.append({"from": s, "to": new})
        else:
            failed.append({"session": s, "reason": reason})
    return {"tool": "modify_schedule", "unavailable_date": day.isoformat(), "moved": moved, "failed": failed,
            "schedule": get_schedule(prof["student_id"], status="planned")}


def create_study_plan(*args, **kwargs) -> dict:
    """Compatibility alias: the repo skeleton lists create_study_plan under tools/scheduler.py.
    The implementation lives in tools/planner.py (imported lazily to avoid a circular import)."""
    from tools.planner import create_study_plan as _create

    return _create(*args, **kwargs)
