import pytest

pytest.importorskip("langchain_core")
from tools.langchain_tools import get_member3_tools  # noqa: E402


def _tool(name):
    return next(t for t in get_member3_tools() if t.name == name)


def test_six_tools_exposed():
    assert len(get_member3_tools()) == 6


def test_sgpa_tool_returns_deterministic_result():
    r = _tool("sgpa_tool").invoke({"courses": [{"subject": "M", "grade": "A", "credits": 4}, {"subject": "P", "grade": "B", "credits": 4}]})
    assert r["ok"] and r["sgpa"] == 7.0


def test_tool_errors_are_returned_not_raised():
    assert _tool("sgpa_tool").invoke({"courses": [{"subject": "M", "grade": "Z", "credits": 4}]})["ok"] is False
    assert _tool("student_profile_tool").invoke({"student_id": "NOPE"})["ok"] is False
