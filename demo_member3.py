"""Quick demo of everything Member 3 owns:  python demo_member3.py"""
import os
import tempfile
from datetime import date

os.environ.setdefault("CAMPUSAI_DB", os.path.join(tempfile.mkdtemp(), "demo.db"))  # throwaway DB for the demo

from tools import (calculate_cgpa, calculate_sgpa, create_study_plan, get_schedule, get_student_profile,
                   reschedule_unavailable_day, resolve_weekday, validate_plan)

SID = "24CSE001"
p = get_student_profile(SID)
print(f"1. LOGIN      {p['name']} | {p['department']} sem {p['semester']} | CGPA {p['cgpa']}")

r = calculate_sgpa([("Maths", "A", 4), ("Physics", "B+", 3), ("Chemistry", "O", 3), ("Workshop", "C", 2)])
print(f"2. SGPA       {r['formula']}")
print(f"   CGPA       {calculate_cgpa([{'semester': 1, 'sgpa': 8.0, 'credits': 20}, {'semester': 2, 'sgpa': 9.0, 'credits': 10}])['formula']}")

plan = create_study_plan(SID, today=date(2026, 10, 5))
print(f"3. STUDY PLAN {len(plan['plan'])} sessions, feasible={plan['feasible']}, problems={validate_plan(plan) or 'none'}")
for s in plan["per_subject"]:
    print(f"   {s['subject']:<36} {s['total_minutes'] / 60:>4.1f}h  (exam {s['exam_date']})")
for s in plan["plan"][:4]:
    print(f"   {s['date']} {s['start_time']}  {s['duration_min']}min  {s['subject']}")

wed = resolve_weekday("Wednesday", today=date(2026, 10, 5))
out = reschedule_unavailable_day(SID, wed, today=date(2026, 10, 5))
print(f"4. 'I CANNOT STUDY WEDNESDAY' ({wed})  strategy={out['strategy']}, sessions dropped={out['sessions_dropped']}")
print(f"   (timetable was full, so the remaining days were re-planned around {wed}; problems={out['issues'] or 'none'})")
print(f"   sessions left on {wed}: {len(get_schedule(SID, wed, wed))}")
