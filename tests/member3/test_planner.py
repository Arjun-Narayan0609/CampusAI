from datetime import date

import pytest

from student.registry import StudentNotFoundError, get_student_profile
from tools import scheduler as sch
from tools.planner import create_study_plan, validate_plan

TODAY = date(2026, 10, 5)


@pytest.mark.parametrize("sid", ["24CSE001", "24CSE002", "24ISE003", "24ECE004"])
def test_plan_is_valid_and_covers_every_subject(sid):
    prof = get_student_profile(sid)
    plan = create_study_plan(sid, today=TODAY)
    assert plan["plan"] and validate_plan(plan) == []
    assert {s["subject"] for s in plan["plan"]} == set(prof["subjects"])
    for s in plan["plan"]:  # all before that subject's exam, none today
        assert TODAY.isoformat() < s["date"] < prof["exam_dates"][s["subject"]]


def test_respects_daily_hours_and_rest_day():
    plan = create_study_plan("24CSE002", today=TODAY)  # 2h/day, rest on Saturday
    per_day: dict[str, int] = {}
    for s in plan["plan"]:
        per_day[s["date"]] = per_day.get(s["date"], 0) + s["duration_min"]
        assert date.fromisoformat(s["date"]).strftime("%A") != "Saturday"
    assert max(per_day.values()) <= 120


def test_deterministic():
    a = create_study_plan("24CSE001", today=TODAY, persist=False)["plan"]
    b = create_study_plan("24CSE001", today=TODAY, persist=False)["plan"]
    assert a == b


def test_nearer_exams_get_more_time():
    per = {p["subject"]: p["total_minutes"] for p in create_study_plan("24CSE001", today=TODAY)["per_subject"]}
    assert per["Database Management Systems"] > per["Software Engineering"]  # exam Nov 16 vs Nov 25... fewer days, so more focus per day


def test_persisted_and_replanning_does_not_duplicate():
    first = create_study_plan("24CSE001", today=TODAY)
    n = len(sch.get_schedule("24CSE001"))
    assert n == len(first["plan"]) and all("id" in s for s in first["plan"])
    create_study_plan("24CSE001", today=TODAY)
    assert len(sch.get_schedule("24CSE001")) == n


def test_replanning_keeps_completed_history():
    first = create_study_plan("24CSE001", today=TODAY)
    sch.complete_session(first["plan"][0]["id"], "24CSE001")
    create_study_plan("24CSE001", today=TODAY)
    assert len(sch.get_schedule("24CSE001", status="completed")) == 1


def test_tight_timetable_is_flagged_infeasible():
    plan = create_study_plan("24CSE005", today=TODAY, persist=False)  # 1.5h/day, exams in 7-11 days
    assert plan["feasible"] is False and any("Consider more daily hours" in w for w in plan["warnings"])


def test_unknown_student_and_past_exams():
    with pytest.raises(StudentNotFoundError):
        create_study_plan("NOPE")
    plan = create_study_plan("24CSE005", today=date(2026, 12, 1), persist=False)
    assert plan["plan"] == [] and any("already passed" in w for w in plan["warnings"])


def test_validate_plan_catches_problems():
    plan = create_study_plan("24CSE001", today=TODAY, persist=False)
    plan["plan"].append({"subject": "Database Management Systems", "date": "2026-11-16", "start_time": "18:00", "duration_min": 60})
    assert any("not before the exam" in i for i in validate_plan(plan))


def test_blocked_dates_are_skipped():
    plan = create_study_plan("24CSE001", today=TODAY, persist=False, blocked_dates=["2026-10-07"])
    assert all(s["date"] != "2026-10-07" for s in plan["plan"]) and validate_plan(plan) == []


def test_cannot_study_wednesday_on_a_full_timetable_replans():
    from tools.planner import reschedule_unavailable_day
    first = create_study_plan("24CSE001", today=TODAY)
    wed = "2026-10-07"
    assert sch.get_schedule("24CSE001", wed, wed)  # there is something to move
    r = reschedule_unavailable_day("24CSE001", wed, today=TODAY)
    assert r["strategy"] == "replanned" and r["issues"] == []
    assert sch.get_schedule("24CSE001", wed, wed) == []  # persisted
    assert r["sessions_dropped"] == len(first["plan"]) - len(r["schedule"]) > 0


def test_cannot_study_wednesday_with_slack_just_moves():
    from tools.planner import reschedule_unavailable_day
    sch.add_session("24CSE001", "Computer Networks", "2026-10-07", "18:00", 60)
    r = reschedule_unavailable_day("24CSE001", "2026-10-07", today=TODAY)
    assert r["strategy"] == "moved" and r["sessions_dropped"] == 0 and not r["failed"]
