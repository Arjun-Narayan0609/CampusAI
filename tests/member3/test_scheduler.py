from datetime import date

import pytest

from tools import scheduler as s

SID = "24CSE001"
MON, TUE, WED, THU = "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"


def test_add_and_list_sorted():
    s.add_session(SID, "CN", MON, "19:30", 60)
    s.add_session(SID, "DBMS", MON, "18:00", 60)
    assert [x["subject"] for x in s.get_schedule(SID)] == ["DBMS", "CN"]
    assert s.get_schedule(SID)[0]["end_time"] == "19:00"


def test_persists_across_calls_and_isolated_per_student():
    s.add_session(SID, "DBMS", MON, "18:00", 60)
    assert s.get_schedule("24CSE002") == []
    assert len(s.get_schedule(SID.lower())) == 1


def test_overlap_and_duplicate_rejected():
    s.add_session(SID, "DBMS", MON, "18:00", 60)
    with pytest.raises(s.ScheduleConflictError):
        s.add_session(SID, "DBMS", MON, "18:00", 60)  # duplicate
    with pytest.raises(s.ScheduleConflictError):
        s.add_session(SID, "CN", MON, "18:30", 60)  # overlap
    s.add_session(SID, "CN", MON, "19:00", 60)  # back-to-back is fine


@pytest.mark.parametrize("start,dur", [("18:00", 10), ("18:00", 300), ("04:00", 60), ("22:30", 60)])
def test_unrealistic_sessions_rejected(start, dur):
    with pytest.raises(s.ScheduleError):
        s.add_session(SID, "DBMS", MON, start, dur)


def test_daily_limit():
    s.add_session(SID, "A", MON, "18:00", 120, daily_limit_min=180)
    with pytest.raises(s.ScheduleError, match="limit"):
        s.add_session(SID, "B", MON, "20:30", 90, daily_limit_min=180)


def test_complete_miss_modify():
    a = s.add_session(SID, "DBMS", MON, "18:00", 60)
    assert s.complete_session(a["id"], SID)["status"] == "completed"
    with pytest.raises(s.ScheduleError):
        s.modify_session(a["id"], THU)  # completed sessions are frozen
    b = s.add_session(SID, "CN", MON, "19:00", 60)
    assert s.miss_session(b["id"])["status"] == "missed"
    assert s.add_session(SID, "OS", MON, "19:00", 60)  # missed sessions free their slot
    moved = s.modify_session(b["id"], THU, "20:00")
    assert (moved["date"], moved["start_time"], moved["status"]) == (THU, "20:00", "planned")


def test_other_students_session_not_touchable():
    a = s.add_session(SID, "DBMS", MON, "18:00", 60)
    with pytest.raises(s.SessionNotFoundError):
        s.complete_session(a["id"], "24CSE002")


def test_resolve_weekday():
    assert s.resolve_weekday("Wednesday", today=date(2026, 10, 5)) == date(2026, 10, 7)
    assert s.resolve_weekday("monday", today=date(2026, 10, 5)) == date(2026, 10, 5)
    assert s.resolve_weekday("monday", today=date(2026, 10, 5), include_today=False) == date(2026, 10, 12)


def test_cannot_study_wednesday_moves_sessions_to_thursday():
    s.add_session(SID, "DBMS", WED, "18:00", 60)
    s.add_session(SID, "CN", WED, "19:10", 60)
    r = s.modify_schedule(SID, WED, move_to_date=THU)
    assert len(r["moved"]) == 2 and not r["failed"]
    assert {x["date"] for x in r["schedule"]} == {THU}
    assert s.get_schedule(SID, WED, WED) == []  # and it persists


def test_auto_reschedule_skips_rest_day_and_respects_exam():
    sat = "2026-10-10"  # next day (Sunday) is 24CSE001's rest day
    s.add_session(SID, "DBMS", sat, "18:00", 60)
    r = s.modify_schedule(SID, sat)
    assert r["moved"][0]["to"]["date"] == "2026-10-12"  # Monday
    # exam guard: DBMS exam is 2026-11-16, so a session can't move onto/after it
    s.add_session(SID, "Database Management Systems", "2026-11-15", "18:00", 60)
    r2 = s.modify_schedule(SID, "2026-11-15")
    assert r2["failed"] and "exam" in r2["failed"][0]["reason"]


def test_create_study_plan_importable_from_scheduler_module():
    assert s.create_study_plan("24CSE001", today=date(2026, 10, 5), persist=False)["plan"]
