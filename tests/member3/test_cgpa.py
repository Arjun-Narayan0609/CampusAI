import pytest

from tools.cgpa import GradeError, calculate_cgpa, calculate_sgpa

SEM1 = [
    {"subject": "Maths", "grade": "A", "credits": 4},   # 32
    {"subject": "Physics", "grade": "B+", "credits": 3},  # 21
    {"subject": "Chemistry", "grade": "O", "credits": 3},  # 30
    {"subject": "Workshop", "grade": "C", "credits": 2},  # 10   -> 93 / 12 = 7.75
]


def test_sgpa_hand_verified():
    r = calculate_sgpa(SEM1)
    assert (r["sgpa"], r["total_credits"], r["credit_points"]) == (7.75, 12, 93)
    assert "93 / 12 = 7.75" in r["formula"]


def test_sgpa_rounds_half_up_not_bankers():
    # (3*8 + 5*5) / 8 = 49 / 8 = 6.125 -> 6.13
    assert calculate_sgpa([("a", "A", 3), ("b", "C", 5)])["sgpa"] == 6.13


def test_failed_course_counts_in_denominator():
    assert calculate_sgpa([("a", "O", 3), ("b", "F", 3)])["sgpa"] == 5.0


def test_lowercase_grades_and_tuples_accepted():
    assert calculate_sgpa([("a", "b+", 4)])["sgpa"] == 7.0


def test_alternate_scale():
    assert calculate_sgpa([("a", "S", 4), ("b", "B", 4)], scale="vtu_10")["sgpa"] == 9.0


@pytest.mark.parametrize("bad", [[], [{"grade": "Z", "credits": 3}], [{"grade": "A", "credits": 0}],
                                 [{"grade": "A", "credits": -2}], [{"grade": "A"}], [{"grade": "A", "credits": "x"}]])
def test_sgpa_rejects_bad_input(bad):
    with pytest.raises(GradeError):
        calculate_sgpa(bad)


def test_cgpa_credit_weighted_from_courses():
    sem2 = [("a", "A", 4), ("b", "B", 4)]  # 32+24 = 56 over 8 credits
    r = calculate_cgpa([{"semester": 1, "courses": SEM1}, {"semester": 2, "courses": sem2}])
    assert r["cgpa"] == 7.45  # (93 + 56) / 20 = 7.45
    assert r["weighting"] == "credit" and r["semesters"][0]["sgpa"] == 7.75


def test_cgpa_from_sgpa_and_credits():
    r = calculate_cgpa([{"sgpa": 8.0, "credits": 20}, {"sgpa": 9.0, "credits": 10}])
    assert r["cgpa"] == 8.33  # (160 + 90) / 30


def test_cgpa_equal_weight_when_no_credits():
    r = calculate_cgpa([{"sgpa": 8.0}, {"sgpa": 9.0}])
    assert (r["cgpa"], r["weighting"]) == (8.5, "equal")


def test_cgpa_rejects_mixed_credits_and_bad_sgpa():
    with pytest.raises(GradeError):
        calculate_cgpa([{"sgpa": 8.0, "credits": 20}, {"sgpa": 9.0}])
    with pytest.raises(GradeError):
        calculate_cgpa([{"sgpa": 11, "credits": 20}])
