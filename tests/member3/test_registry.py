import pytest

from student.registry import RegistryError, StudentNotFoundError, get_student_profile, list_student_ids, validate_registry


def test_valid_login_loads_profile():
    p = get_student_profile("24CSE001")
    assert (p["name"], p["department"], p["semester"]) == ("Aarav Shetty", "CSE", 5)
    assert "Database Management Systems" in p["subjects"] and p["exam_dates"]["Computer Networks"] == "2026-11-18"


def test_id_is_normalised():
    assert get_student_profile("  24cse001 ")["student_id"] == "24CSE001"


def test_unknown_id_raises_and_creates_nothing():
    with pytest.raises(StudentNotFoundError):
        get_student_profile("99XYZ999")
    assert "99XYZ999" not in list_student_ids()


def test_whole_registry_is_valid():
    assert validate_registry() >= 3


def test_bad_row_is_reported(tmp_path, monkeypatch):
    f = tmp_path / "s.csv"
    f.write_text("student_id,name,department,semester,year,cgpa,subjects,exam_dates\nX1,A,CSE,9,1,8,Maths,Maths:2026-11-01\n")
    monkeypatch.setenv("CAMPUSAI_STUDENTS_CSV", str(f))
    with pytest.raises(RegistryError):
        get_student_profile("X1")


def test_minimal_csv_with_only_core_columns_works(tmp_path, monkeypatch):
    f = tmp_path / "s.csv"
    f.write_text("student_id,name,department,semester,year,cgpa\nSTU001,Alex Rivera,Computer Science,4,2,8.45\n")
    monkeypatch.setenv("CAMPUSAI_STUDENTS_CSV", str(f))
    p = get_student_profile("stu001")
    assert p["department"] == "Computer Science" and p["subjects"] == [] and p["available_hours_per_day"] == 3.0


def test_csv_saved_by_excel_with_bom_works(tmp_path, monkeypatch):
    f = tmp_path / "s.csv"
    f.write_bytes(b"\xef\xbb\xbfstudent_id,name,department,semester,year,cgpa\r\nSTU001,Alex,CSE,4,2,8.45\r\n")
    monkeypatch.setenv("CAMPUSAI_STUDENTS_CSV", str(f))
    assert get_student_profile("STU001")["name"] == "Alex"
