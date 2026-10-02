"""
Focused Unit & Integration Tests for Question Analyzer Module
Owned by: Member 2
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from agents.question_analyzer import (
    VALID_INTENTS,
    QuestionAnalysis,
    analyze_question,
    classify_question_intent,
)
from agents.state import AgentState
from llm.groq import get_groq_api_key, get_groq_llm


class TestQuestionAnalyzerUnit(unittest.TestCase):
    """Unit tests for Question Analyzer using mocked LLM responses."""

    def _create_mock_llm(self, intent_to_return: str):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QuestionAnalysis(intent=intent_to_return)
        return mock_llm

    def test_all_five_valid_intents(self):
        """Test that all five Phase 1 intents can be correctly returned and stored."""
        test_cases = [
            ("What is the attendance requirement?", "ACADEMIC_QUERY"),
            ("Make me a study plan for my exams.", "STUDY_PLAN"),
            ("Calculate my CGPA.", "CGPA"),
            ("What classes do I have tomorrow?", "SCHEDULE"),
            ("Hello there!", "GENERAL"),
        ]

        for question, expected_intent in test_cases:
            with self.subTest(intent=expected_intent, question=question):
                mock_llm = self._create_mock_llm(expected_intent)
                state: AgentState = {
                    "student_id": "24CSE001",
                    "question": question,
                }
                updated_state = analyze_question(state, llm=mock_llm)

                self.assertEqual(updated_state["intent"], expected_intent)
                self.assertEqual(updated_state["student_id"], "24CSE001")
                self.assertEqual(updated_state["question"], question)

    def test_invalid_intent_rejected(self):
        """Test that unsupported intents returned by a model fail validation."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        # Return an arbitrary/unsupported intent string
        mock_structured.invoke.return_value = {"intent": "INVALID_UNSUPPORTED_INTENT"}

        state: AgentState = {
            "student_id": "24CSE001",
            "question": "Some query",
        }
        with self.assertRaises(ValueError) as ctx:
            analyze_question(state, llm=mock_llm)
        self.assertIn("Unsupported intent", str(ctx.exception))

    def test_invalid_structured_output_rejected(self):
        """Test that malformed or non-structured model responses raise RuntimeError."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = "just a plain text string"

        state: AgentState = {
            "student_id": "24CSE001",
            "question": "What is the syllabus?",
        }
        with self.assertRaises(RuntimeError) as ctx:
            analyze_question(state, llm=mock_llm)
        self.assertIn("invalid structured output", str(ctx.exception))

    def test_empty_or_missing_question_rejected(self):
        """Test that empty or missing questions in state raise ValueError."""
        mock_llm = self._create_mock_llm("GENERAL")

        with self.assertRaises(ValueError):
            analyze_question({"student_id": "24CSE001", "question": ""}, llm=mock_llm)

        with self.assertRaises(ValueError):
            analyze_question({"student_id": "24CSE001", "question": "   "}, llm=mock_llm)

        with self.assertRaises(ValueError):
            analyze_question({"student_id": "24CSE001"}, llm=mock_llm)

        with self.assertRaises(ValueError):
            analyze_question(None, llm=mock_llm)

    def test_missing_api_key_raises_value_error(self):
        """Test that missing GROQ_API_KEY raises a descriptive ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("llm.groq.load_dotenv"):
                with self.assertRaises(ValueError) as ctx:
                    get_groq_api_key()
                self.assertIn("GROQ_API_KEY environment variable is not set", str(ctx.exception))

    def test_valid_intents_constant(self):
        """Verify the five mandatory intents are defined."""
        expected = {"ACADEMIC_QUERY", "STUDY_PLAN", "CGPA", "SCHEDULE", "GENERAL"}
        self.assertEqual(VALID_INTENTS, expected)


class TestQuestionAnalyzerLive(unittest.TestCase):
    """Live integration tests against Groq if GROQ_API_KEY is available."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.api_key = get_groq_api_key()
        except ValueError:
            cls.api_key = None

    def setUp(self):
        if not self.api_key:
            self.skipTest("GROQ_API_KEY not set in environment or .env; skipping live API tests.")

    def test_live_groq_classification_accuracy(self):
        """Test live model against core classification principles."""
        scenarios = [
            ("What is the attendance requirement for exams?", "ACADEMIC_QUERY"),
            ("What is the CGPA requirement for an internship?", "ACADEMIC_QUERY"),
            ("Make me a study plan for my end semester exams.", "STUDY_PLAN"),
            ("What subjects should I study for tomorrow's exam?", "STUDY_PLAN"),
            ("Calculate my CGPA from these semester grades.", "CGPA"),
            ("What will my CGPA be if I score 9 in this subject?", "CGPA"),
            ("What classes do I have tomorrow morning?", "SCHEDULE"),
            ("Move my study session from Wednesday to Friday.", "SCHEDULE"),
            ("Hello, how are you doing today?", "GENERAL"),
        ]

        for question, expected in scenarios:
            with self.subTest(question=question, expected=expected):
                state: AgentState = {"student_id": "24CSE001", "question": question}
                result = analyze_question(state)
                self.assertEqual(result["intent"], expected)


if __name__ == "__main__":
    unittest.main()
