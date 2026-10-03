"""
Unit Tests for the Response Reviewer Module
Owned by: Member 2

Tests cover:
- Valid responses pass review unchanged (all intents)
- Missing evidence triggers bounded fallback (ACADEMIC_QUERY)
- Missing tool results trigger bounded fallback (CGPA, STUDY_PLAN, SCHEDULE)
- Missing/empty answer handling
- State preservation across review
- No-invention rule enforcement

No real Groq API calls, ChromaDB, or external tools required.
All tests use plain AgentState dictionaries.
"""

import unittest
from typing import Dict

from agents.reviewer import (
    FALLBACK_ACADEMIC,
    FALLBACK_CGPA,
    FALLBACK_GENERAL,
    FALLBACK_SCHEDULE,
    FALLBACK_STUDY_PLAN,
    FALLBACK_UNKNOWN_INTENT,
    review_response,
)
from agents.state import AgentState


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _academic_state(**overrides) -> Dict:
    """Create a baseline valid ACADEMIC_QUERY state."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "What is the attendance policy?",
        "intent": "ACADEMIC_QUERY",
        "context": "Minimum 75% attendance is required as per college policy.",
        "sources": [{"document": "handbook.pdf", "page": 12}],
        "answer": "The college requires a minimum of 75% attendance.",
    }
    state.update(overrides)
    return state


def _cgpa_state(**overrides) -> Dict:
    """Create a baseline valid CGPA state."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "Calculate my CGPA.",
        "intent": "CGPA",
        "tool_result": {"cgpa": 8.75, "sgpa": 9.0},
        "answer": "Your CGPA is 8.75 and current SGPA is 9.0.",
    }
    state.update(overrides)
    return state


def _study_plan_state(**overrides) -> Dict:
    """Create a baseline valid STUDY_PLAN state."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "Make me a study plan for finals.",
        "intent": "STUDY_PLAN",
        "tool_result": {"sessions": [{"subject": "Math", "hours": 2}]},
        "answer": "Here is your study plan: Math - 2 hours.",
    }
    state.update(overrides)
    return state


def _schedule_state(**overrides) -> Dict:
    """Create a baseline valid SCHEDULE state."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "What classes do I have tomorrow?",
        "intent": "SCHEDULE",
        "tool_result": {"classes": ["Math 101", "Physics 201"]},
        "answer": "Tomorrow you have Math 101 and Physics 201.",
    }
    state.update(overrides)
    return state


def _general_state(**overrides) -> Dict:
    """Create a baseline valid GENERAL state."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "Hello!",
        "intent": "GENERAL",
        "answer": "Hello! How can I help you today?",
    }
    state.update(overrides)
    return state


# ===========================================================================
# 1. Valid academic response — answer preserved
# ===========================================================================


class TestValidAcademicResponse(unittest.TestCase):
    """Valid ACADEMIC_QUERY with context, sources, and answer should pass."""

    def test_valid_academic_answer_unchanged(self) -> None:
        state = _academic_state()
        original_answer = state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], original_answer)


# ===========================================================================
# 2. Academic response with missing context
# ===========================================================================


class TestAcademicMissingContext(unittest.TestCase):
    """ACADEMIC_QUERY without context must be replaced with fallback."""

    def test_empty_context_triggers_fallback(self) -> None:
        state = _academic_state(context="")
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)
        self.assertNotEqual(result["answer"], "The college requires a minimum of 75% attendance.")

    def test_none_context_triggers_fallback(self) -> None:
        state = _academic_state(context=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_whitespace_context_triggers_fallback(self) -> None:
        state = _academic_state(context="   ")
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)


# ===========================================================================
# 3. Academic response with missing sources
# ===========================================================================


class TestAcademicMissingSources(unittest.TestCase):
    """ACADEMIC_QUERY without sources must not pass as fully grounded."""

    def test_empty_sources_list_triggers_fallback(self) -> None:
        state = _academic_state(sources=[])
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_none_sources_triggers_fallback(self) -> None:
        state = _academic_state(sources=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)


# ===========================================================================
# 4. Valid CGPA tool result — answer preserved
# ===========================================================================


class TestValidCgpaResponse(unittest.TestCase):
    """Valid CGPA with tool_result and answer should pass unchanged."""

    def test_valid_cgpa_answer_unchanged(self) -> None:
        state = _cgpa_state()
        original_answer = state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], original_answer)


# ===========================================================================
# 5. Missing CGPA tool result
# ===========================================================================


class TestCgpaMissingToolResult(unittest.TestCase):
    """CGPA without tool_result must replace unsupported answer."""

    def test_none_tool_result_triggers_fallback(self) -> None:
        state = _cgpa_state(tool_result=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_CGPA)
        # The original unsupported CGPA claim must NOT survive
        self.assertNotIn("8.75", result["answer"])

    def test_missing_tool_result_key_triggers_fallback(self) -> None:
        state = _cgpa_state()
        del state["tool_result"]
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_CGPA)


# ===========================================================================
# 6. Valid study-plan result — answer preserved
# ===========================================================================


class TestValidStudyPlanResponse(unittest.TestCase):
    """Valid STUDY_PLAN with tool_result and answer should pass unchanged."""

    def test_valid_study_plan_answer_unchanged(self) -> None:
        state = _study_plan_state()
        original_answer = state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], original_answer)


# ===========================================================================
# 7. Missing study-plan result
# ===========================================================================


class TestStudyPlanMissingToolResult(unittest.TestCase):
    """STUDY_PLAN without tool_result must return bounded fallback."""

    def test_none_tool_result_triggers_fallback(self) -> None:
        state = _study_plan_state(tool_result=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_STUDY_PLAN)

    def test_empty_tool_result_triggers_fallback(self) -> None:
        state = _study_plan_state(tool_result={})
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_STUDY_PLAN)


# ===========================================================================
# 8. Valid schedule result — answer preserved
# ===========================================================================


class TestValidScheduleResponse(unittest.TestCase):
    """Valid SCHEDULE with tool_result and answer should pass unchanged."""

    def test_valid_schedule_answer_unchanged(self) -> None:
        state = _schedule_state()
        original_answer = state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], original_answer)


# ===========================================================================
# 9. Missing schedule result
# ===========================================================================


class TestScheduleMissingToolResult(unittest.TestCase):
    """SCHEDULE without tool_result must return bounded fallback."""

    def test_none_tool_result_triggers_fallback(self) -> None:
        state = _schedule_state(tool_result=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_SCHEDULE)

    def test_empty_list_tool_result_triggers_fallback(self) -> None:
        state = _schedule_state(tool_result=[])
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_SCHEDULE)


# ===========================================================================
# 10. GENERAL answer — preserved when present
# ===========================================================================


class TestGeneralResponse(unittest.TestCase):
    """GENERAL with a normal answer should pass unchanged."""

    def test_valid_general_answer_unchanged(self) -> None:
        state = _general_state()
        original_answer = state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], original_answer)


# ===========================================================================
# 11. Missing / empty answer handling
# ===========================================================================


class TestMissingAnswer(unittest.TestCase):
    """Reviewer must safely handle missing, None, or empty answers."""

    def test_none_answer_academic_triggers_fallback(self) -> None:
        state = _academic_state(answer=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_empty_answer_academic_triggers_fallback(self) -> None:
        state = _academic_state(answer="")
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_none_answer_cgpa_triggers_fallback(self) -> None:
        state = _cgpa_state(answer=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_CGPA)

    def test_none_answer_general_triggers_fallback(self) -> None:
        state = _general_state(answer=None)
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_GENERAL)

    def test_whitespace_answer_general_triggers_fallback(self) -> None:
        state = _general_state(answer="   ")
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_GENERAL)

    def test_missing_answer_key_schedule_triggers_fallback(self) -> None:
        state = _schedule_state()
        del state["answer"]
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_SCHEDULE)


# ===========================================================================
# 12. State preservation
# ===========================================================================


class TestStatePreservation(unittest.TestCase):
    """Reviewer must preserve all unrelated state fields."""

    def test_academic_preserves_fields_on_valid(self) -> None:
        state = _academic_state()
        result = review_response(state)
        self.assertEqual(result["student_id"], "24CSE001")
        self.assertEqual(result["question"], "What is the attendance policy?")
        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["context"], "Minimum 75% attendance is required as per college policy.")
        self.assertEqual(result["sources"], [{"document": "handbook.pdf", "page": 12}])

    def test_academic_preserves_fields_on_fallback(self) -> None:
        """Even when answer is replaced, other fields must survive."""
        state = _academic_state(context="")
        result = review_response(state)
        self.assertEqual(result["student_id"], "24CSE001")
        self.assertEqual(result["question"], "What is the attendance policy?")
        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["sources"], [{"document": "handbook.pdf", "page": 12}])
        # Answer was replaced
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_cgpa_preserves_tool_result_on_valid(self) -> None:
        state = _cgpa_state()
        result = review_response(state)
        self.assertEqual(result["tool_result"], {"cgpa": 8.75, "sgpa": 9.0})
        self.assertEqual(result["student_id"], "24CSE001")
        self.assertEqual(result["intent"], "CGPA")

    def test_cgpa_preserves_fields_on_fallback(self) -> None:
        state = _cgpa_state(tool_result=None)
        result = review_response(state)
        self.assertEqual(result["student_id"], "24CSE001")
        self.assertEqual(result["question"], "Calculate my CGPA.")
        self.assertEqual(result["intent"], "CGPA")
        # tool_result is None but key should still exist
        self.assertIsNone(result.get("tool_result"))

    def test_general_preserves_student_id(self) -> None:
        state = _general_state(student_id="24CSE099")
        result = review_response(state)
        self.assertEqual(result["student_id"], "24CSE099")


# ===========================================================================
# Additional edge cases
# ===========================================================================


class TestUnknownIntent(unittest.TestCase):
    """Unknown or missing intents should produce a safe fallback."""

    def test_unknown_intent_triggers_fallback(self) -> None:
        state: Dict = {
            "student_id": "24CSE001",
            "question": "Something unusual",
            "intent": "UNKNOWN_INTENT",
            "answer": "Some answer",
        }
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_UNKNOWN_INTENT)

    def test_missing_intent_triggers_fallback(self) -> None:
        state: Dict = {
            "student_id": "24CSE001",
            "question": "Something",
        }
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_UNKNOWN_INTENT)

    def test_none_intent_triggers_fallback(self) -> None:
        state: Dict = {
            "student_id": "24CSE001",
            "question": "Something",
            "intent": None,
            "answer": "Some answer",
        }
        result = review_response(state)
        self.assertEqual(result["answer"], FALLBACK_UNKNOWN_INTENT)


class TestNoInventionRule(unittest.TestCase):
    """The reviewer must NEVER invent institutional information."""

    def test_academic_fallback_does_not_contain_policy_facts(self) -> None:
        """Fallback for academic should not contain fabricated details."""
        state = _academic_state(context=None, sources=None)
        result = review_response(state)
        answer = result["answer"]
        # Must not contain any specific institutional claim
        self.assertNotIn("75%", answer)
        self.assertNotIn("attendance", answer.lower())

    def test_cgpa_fallback_does_not_contain_numbers(self) -> None:
        """Fallback for CGPA should not contain fabricated grades."""
        state = _cgpa_state(tool_result=None)
        result = review_response(state)
        answer = result["answer"]
        self.assertNotIn("8.75", answer)
        self.assertNotIn("9.0", answer)

    def test_study_plan_fallback_does_not_contain_subjects(self) -> None:
        """Fallback for study plan should not contain fabricated sessions."""
        state = _study_plan_state(tool_result=None)
        result = review_response(state)
        answer = result["answer"]
        self.assertNotIn("Math", answer)
        self.assertNotIn("2 hours", answer)

    def test_schedule_fallback_does_not_contain_classes(self) -> None:
        """Fallback for schedule should not contain fabricated classes."""
        state = _schedule_state(tool_result=None)
        result = review_response(state)
        answer = result["answer"]
        self.assertNotIn("Math 101", answer)
        self.assertNotIn("Physics 201", answer)


if __name__ == "__main__":
    unittest.main()
