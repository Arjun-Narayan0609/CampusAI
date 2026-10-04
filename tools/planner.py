"""Deterministic study planner (Member 3). The LLM may *phrase* the result; it never decides the allocation.

Algorithm (same input -> same plan):
  1. Study days = tomorrow .. day before the last exam (max 28 days), skipping the student's rest day.
  2. Each day gets floor(available_minutes / block) blocks, laid out from the student's study_start.
  3. Every block goes to the eligible subject (exam still ahead) with the highest score
        score = weight * sqrt(1 / days_left) / (1 + blocks_already_given)
     so nearer exams get more time but every subject keeps getting covered. Ties: earlier exam, then name.
  4. Warn (feasible=False) if a subject ends up with less than MIN_HOURS_PER_SUBJECT before its exam.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import sqrt

from student.registry import get_student_profile
from tools import scheduler as sch

BREAK_MIN = 10
MIN_HOURS_PER_SUBJECT = 4
MAX_HORIZON_DAYS = 28


def create_study_plan(student_id: str, today: date | None = None, persist: bool = True,
                      weights: dict[str, float] | None = None,
                      blocked_dates: list[date | str] | None = None) -> dict:
    prof = get_student_profile(student_id)
    sid, today = prof["student_id"], today or date.today()
    block = prof["block_minutes"]
    limit = int(prof["available_hours_per_day"] * 60)
    start0 = sch.to_minutes(prof["study_start"])
    rest = prof["rest_day"]
    weights = weights or {}
    blocked = {sch._d(d) for d in blocked_dates or []}  # days the student cannot study
    warnings: list[str] = []

    horizon_end = today + timedelta(days=MAX_HORIZON_DAYS)
    exams: dict[str, date] = {}
    for subj in prof["subjects"]:
        iso = prof["exam_dates"].get(subj)
        if iso is None:
            exams[subj] = horizon_end + timedelta(days=1)
            warnings.append(f"No exam date for {subj}; planning up to {MAX_HORIZON_DAYS} days ahead.")
        elif date.fromisoformat(iso) <= today:
            warnings.append(f"The {subj} exam ({iso}) has already passed; skipped.")
        else:
            exams[subj] = date.fromisoformat(iso)

    result = {"tool": "create_study_plan", "student_id": sid, "generated_for": today.isoformat(), "feasible": False,
              "daily_limit_min": limit, "rest_day": rest, "blocked_dates": sorted(d.isoformat() for d in blocked), "plan": [], "per_subject": [], "warnings": warnings}
    if not exams:
        warnings.append("No upcoming exams, so there is nothing to plan.")
        return result

    last_day = min(max(exams.values()) - timedelta(days=1), horizon_end)
    blocks_given = {s: 0 for s in exams}
    sessions: list[dict] = []
    day = today + timedelta(days=1)
    while day <= last_day:
        if (rest is None or sch.WEEKDAYS[day.weekday()] != rest) and day not in blocked:
            n_blocks = limit // block
            # never run past the end of the allowed day
            while n_blocks and start0 + n_blocks * block + (n_blocks - 1) * BREAK_MIN > sch.DAY_END_MIN:
                n_blocks -= 1
            for i in range(n_blocks):
                eligible = [s for s, ex in exams.items() if ex > day]
                if not eligible:
                    break
                def score(s: str) -> tuple:
                    days_left = (exams[s] - day).days
                    return (-weights.get(s, 1.0) * sqrt(1 / days_left) / (1 + blocks_given[s]), exams[s], s)
                subj = min(eligible, key=score)
                blocks_given[subj] += 1
                start = start0 + i * (block + BREAK_MIN)
                sessions.append({"subject": subj, "date": day.isoformat(), "start_time": sch.fmt_minutes(start), "duration_min": block})
        day += timedelta(days=1)

    if persist:
        sch.clear_planned(sid, from_date=today + timedelta(days=1))
        saved = []
        for s in sessions:
            try:
                saved.append(sch.add_session(sid, s["subject"], s["date"], s["start_time"], s["duration_min"], daily_limit_min=limit))
            except sch.ScheduleError as e:  # e.g. a completed session already occupies the slot
                warnings.append(f"Skipped {s['subject']} on {s['date']} {s['start_time']}: {e}")
        sessions = [{k: v for k, v in x.items() if k in ("id", "subject", "date", "start_time", "duration_min")} for x in saved]

    feasible = True
    for subj, ex in exams.items():
        mine = [s for s in sessions if s["subject"] == subj]
        total = sum(s["duration_min"] for s in mine)
        result["per_subject"].append({"subject": subj, "exam_date": ex.isoformat() if subj in prof["exam_dates"] else None,
                                      "sessions": len(mine), "total_minutes": total})
        if total < MIN_HOURS_PER_SUBJECT * 60:
            feasible = False
            warnings.append(f"Only {total / 60:.1f}h available for {subj} before its exam (recommended minimum {MIN_HOURS_PER_SUBJECT}h). Consider more daily hours.")
    result.update(plan=sessions, feasible=feasible)
    return result


def validate_plan(plan: dict) -> list[str]:
    """Checks for the reviewer node. Returns a list of problems (empty list = plan is valid)."""
    issues: list[str] = []
    exams = {p["subject"]: p["exam_date"] for p in plan.get("per_subject", []) if p.get("exam_date")}
    limit = plan.get("daily_limit_min")
    by_day: dict[str, list[dict]] = {}
    for s in plan.get("plan", []):
        by_day.setdefault(s["date"], []).append(s)
        if s["subject"] in exams and s["date"] >= exams[s["subject"]]:
            issues.append(f"{s['subject']} session on {s['date']} is not before the exam ({exams[s['subject']]})")
        if not sch.MIN_DURATION_MIN <= s["duration_min"] <= sch.MAX_DURATION_MIN:
            issues.append(f"Unrealistic duration {s['duration_min']} min for {s['subject']} on {s['date']}")
    for d, items in by_day.items():
        if limit and sum(i["duration_min"] for i in items) > limit:
            issues.append(f"{d}: planned time exceeds the daily limit of {limit} minutes")
        spans = sorted((sch.to_minutes(i["start_time"]), sch.to_minutes(i["start_time"]) + i["duration_min"]) for i in items)
        for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
            if b0 < a1:
                issues.append(f"{d}: overlapping sessions")
    for d in plan.get("blocked_dates", []):
        if d in by_day:
            issues.append(f"{d} has sessions but the student marked it unavailable")
    if plan.get("rest_day"):
        for d in by_day:
            if sch.WEEKDAYS[date.fromisoformat(d).weekday()] == plan["rest_day"]:
                issues.append(f"{d} falls on the student's rest day")
    return issues


def reschedule_unavailable_day(student_id: str, unavailable_date: date | str, today: date | None = None) -> dict:
    """Use this for "I cannot study on Wednesday".

    1. Try to MOVE that day's sessions into free time on other days (strategy="moved").
    2. If the timetable is too full for that, RE-PLAN the remaining days with that date blocked
       (strategy="replanned"); the result reports how many sessions had to be dropped.
    """
    day = sch._d(unavailable_date)
    sid = get_student_profile(student_id)["student_id"]
    before = len(sch.get_schedule(sid, status="planned"))
    attempt = sch.modify_schedule(sid, day)
    if not attempt["failed"]:
        return {**attempt, "tool": "reschedule_unavailable_day", "strategy": "moved", "sessions_dropped": 0}
    plan = create_study_plan(sid, today=today, blocked_dates=[day])
    for f in attempt["failed"]:  # sessions on today/past dates aren't covered by the re-plan; drop them explicitly
        try:
            sch.delete_session(f["session"]["id"], sid)
        except sch.SessionNotFoundError:
            pass
    schedule = sch.get_schedule(sid, status="planned")
    return {"tool": "reschedule_unavailable_day", "strategy": "replanned", "unavailable_date": day.isoformat(),
            "moved": attempt["moved"], "failed": attempt["failed"], "replan": plan, "issues": validate_plan(plan),
            "sessions_dropped": max(0, before - len(schedule)), "schedule": schedule}
