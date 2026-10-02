"""
Unit Tests for the LangGraph Router Module
Owned by: Member 2

Tests cover:
- All five valid intent → route mappings
- Missing intent handling
- Empty / whitespace intent handling
- Unsupported intent handling
- None intent handling

No real Groq API calls are required — tests use plain AgentState dicts.
"""

import unittest

from agents.router import (
    INTENT_TO_ROUTE,
    SUPPORTED_INTENTS,
    VALID_ROUTES,
    route_question,
)
from agents.state import AgentState


class TestRouteQuestion(unittest.TestCase):
    """Core unit tests for the route_question function."""

    # ------------------------------------------------------------------
    # Happy-path: every valid intent maps to the correct route
    # ------------------------------------------------------------------

    def test_academic_query_routes_to_rag(self) -> None:
        """ACADEMIC_QUERY intent should route to RAG."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "What is the attendance policy?",
            "intent": "ACADEMIC_QUERY",
        }
        self.assertEqual(route_question(state), "RAG")

    def test_study_plan_routes_to_study_plan(self) -> None:
        """STUDY_PLAN intent should route to STUDY_PLAN."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Make me a study plan.",
            "intent": "STUDY_PLAN",
        }
        self.assertEqual(route_question(state), "STUDY_PLAN")

    def test_cgpa_routes_to_cgpa(self) -> None:
        """CGPA intent should route to CGPA."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Calculate my CGPA.",
            "intent": "CGPA",
        }
        self.assertEqual(route_question(state), "CGPA")

    def test_schedule_routes_to_schedule(self) -> None:
        """SCHEDULE intent should route to SCHEDULE."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "What classes do I have tomorrow?",
            "intent": "SCHEDULE",
        }
        self.assertEqual(route_question(state), "SCHEDULE")

    def test_general_routes_to_general(self) -> None:
        """GENERAL intent should route to GENERAL."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": "GENERAL",
        }
        self.assertEqual(route_question(state), "GENERAL")

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_intent_raises_value_error(self) -> None:
        """State without 'intent' key should raise ValueError."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("missing", str(ctx.exception).lower())

    def test_none_intent_raises_value_error(self) -> None:
        """State with intent=None should raise ValueError."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": None,
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("missing", str(ctx.exception).lower())

    def test_empty_intent_raises_value_error(self) -> None:
        """State with an empty string intent should raise ValueError."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": "",
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_whitespace_intent_raises_value_error(self) -> None:
        """State with a whitespace-only intent should raise ValueError."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": "   ",
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_unsupported_intent_raises_value_error(self) -> None:
        """State with an unrecognised intent should raise ValueError."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": "UNKNOWN_INTENT",
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("Unsupported intent", str(ctx.exception))

    def test_case_sensitive_intent_rejected(self) -> None:
        """Intents must be uppercase — lowercase should be rejected."""
        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Hello!",
            "intent": "general",
        }
        with self.assertRaises(ValueError) as ctx:
            route_question(state)
        self.assertIn("Unsupported intent", str(ctx.exception))


class TestRouterConstants(unittest.TestCase):
    """Tests for exported constants and mapping consistency."""

    def test_all_intents_have_routes(self) -> None:
        """Every supported intent must map to a defined route."""
        for intent in SUPPORTED_INTENTS:
            route = INTENT_TO_ROUTE[intent]
            self.assertIn(route, VALID_ROUTES)

    def test_supported_intents_match_question_analyzer(self) -> None:
        """Router's supported intents must align with the Question Analyzer's set."""
        expected = {"ACADEMIC_QUERY", "STUDY_PLAN", "CGPA", "SCHEDULE", "GENERAL"}
        self.assertEqual(SUPPORTED_INTENTS, expected)

    def test_route_map_completeness(self) -> None:
        """INTENT_TO_ROUTE must contain exactly 5 entries."""
        self.assertEqual(len(INTENT_TO_ROUTE), 5)

    def test_valid_routes_set(self) -> None:
        """VALID_ROUTES must contain exactly 5 route identifiers."""
        expected = {"RAG", "STUDY_PLAN", "CGPA", "SCHEDULE", "GENERAL"}
        self.assertEqual(VALID_ROUTES, expected)


if __name__ == "__main__":
    unittest.main()
