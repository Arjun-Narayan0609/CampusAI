"""
Unit Tests for the LangGraph Workflow Skeleton
Owned by: Member 2

Tests cover:
- Graph construction and compilation
- Routing through each of the five capability branches
- State preservation across graph execution
- Reviewer execution for all branches
- Invalid intent handling

No real Groq API calls are required — tests inject a deterministic
fake analyzer that sets the intent directly.
"""

import unittest
from typing import Dict, List

from agents.graph import (
    ROUTE_TO_NODE,
    build_graph,
    cgpa_node,
    general_node,
    rag_node,
    reviewer_node,
    schedule_node,
    study_plan_node,
)
from agents.state import AgentState


# ---------------------------------------------------------------------------
# Deterministic test analyzer (replaces real Groq-based analyzer)
# ---------------------------------------------------------------------------


def _make_fake_analyzer(intent: str):
    """Create a deterministic analyzer that sets a fixed intent.

    Args:
        intent: The intent string to inject into state.

    Returns:
        A function with the same signature as ``analyze_question``.
    """

    def fake_analyzer(state: AgentState) -> AgentState:
        state["intent"] = intent
        return state

    return fake_analyzer


# ---------------------------------------------------------------------------
# Tracking wrappers for observability
# ---------------------------------------------------------------------------

# Module-level tracker reset before each test via setUp
_execution_log: List[str] = []


def _make_tracking_node(node_name: str, original_fn):
    """Wrap a placeholder node to record its execution.

    Args:
        node_name: Logical name to log (e.g. "rag", "reviewer").
        original_fn: The original node function to delegate to.

    Returns:
        A wrapper function that records execution, then delegates.
    """

    def wrapper(state: AgentState) -> AgentState:
        _execution_log.append(node_name)
        return original_fn(state)

    return wrapper


def _build_tracked_graph(intent: str):
    """Build a graph with tracking wrappers and a fake analyzer.

    Uses dependency injection via ``build_graph()`` for the analyzer,
    and monkey-patches the graph's node functions with tracking
    wrappers to verify which branch executed.

    Returns:
        A compiled graph ready for invocation.
    """
    from langgraph.graph import END, START, StateGraph

    from agents.router import route_question

    graph = StateGraph(AgentState)

    # Analyzer: deterministic fake
    graph.add_node("analyzer", _make_fake_analyzer(intent))

    # Capability nodes: tracked versions
    graph.add_node("rag", _make_tracking_node("rag", rag_node))
    graph.add_node("study_plan", _make_tracking_node("study_plan", study_plan_node))
    graph.add_node("cgpa", _make_tracking_node("cgpa", cgpa_node))
    graph.add_node("schedule", _make_tracking_node("schedule", schedule_node))
    graph.add_node("general", _make_tracking_node("general", general_node))
    graph.add_node("reviewer", _make_tracking_node("reviewer", reviewer_node))

    # Edges
    graph.add_edge(START, "analyzer")
    graph.add_conditional_edges("analyzer", route_question, ROUTE_TO_NODE)
    for node_name in ROUTE_TO_NODE.values():
        graph.add_edge(node_name, "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> Dict:
    """Create a minimal valid input state for graph invocation."""
    state: Dict = {
        "student_id": "24CSE001",
        "question": "Test question",
    }
    state.update(overrides)
    return state


# ===========================================================================
# A. Graph Construction
# ===========================================================================


class TestGraphConstruction(unittest.TestCase):
    """Verify that build_graph() creates a valid compiled graph."""

    def test_build_graph_compiles_successfully(self) -> None:
        """build_graph with a fake analyzer should compile without errors."""
        fake_analyzer = _make_fake_analyzer("GENERAL")
        compiled = build_graph(analyzer_fn=fake_analyzer)
        self.assertIsNotNone(compiled)

    def test_build_graph_returns_invokable(self) -> None:
        """The compiled graph should have an invoke method."""
        fake_analyzer = _make_fake_analyzer("GENERAL")
        compiled = build_graph(analyzer_fn=fake_analyzer)
        self.assertTrue(hasattr(compiled, "invoke"))


# ===========================================================================
# B–F. Routing through each capability branch
# ===========================================================================


class TestAcademicQueryRouting(unittest.TestCase):
    """B. ACADEMIC_QUERY → rag → reviewer."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_academic_query_reaches_rag_and_reviewer(self) -> None:
        compiled = _build_tracked_graph("ACADEMIC_QUERY")
        result = compiled.invoke(_base_state())
        self.assertIn("rag", _execution_log)
        self.assertNotIn("study_plan", _execution_log)
        self.assertNotIn("cgpa", _execution_log)
        self.assertNotIn("schedule", _execution_log)
        self.assertNotIn("general", _execution_log)
        self.assertIn("reviewer", _execution_log)
        self.assertEqual(result["intent"], "ACADEMIC_QUERY")


class TestStudyPlanRouting(unittest.TestCase):
    """C. STUDY_PLAN → study_plan → reviewer."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_study_plan_reaches_study_plan_and_reviewer(self) -> None:
        compiled = _build_tracked_graph("STUDY_PLAN")
        result = compiled.invoke(_base_state())
        self.assertIn("study_plan", _execution_log)
        self.assertNotIn("rag", _execution_log)
        self.assertIn("reviewer", _execution_log)
        self.assertEqual(result["intent"], "STUDY_PLAN")


class TestCgpaRouting(unittest.TestCase):
    """D. CGPA → cgpa → reviewer."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_cgpa_reaches_cgpa_and_reviewer(self) -> None:
        compiled = _build_tracked_graph("CGPA")
        result = compiled.invoke(_base_state())
        self.assertIn("cgpa", _execution_log)
        self.assertNotIn("rag", _execution_log)
        self.assertIn("reviewer", _execution_log)
        self.assertEqual(result["intent"], "CGPA")


class TestScheduleRouting(unittest.TestCase):
    """E. SCHEDULE → schedule → reviewer."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_schedule_reaches_schedule_and_reviewer(self) -> None:
        compiled = _build_tracked_graph("SCHEDULE")
        result = compiled.invoke(_base_state())
        self.assertIn("schedule", _execution_log)
        self.assertNotIn("rag", _execution_log)
        self.assertIn("reviewer", _execution_log)
        self.assertEqual(result["intent"], "SCHEDULE")


class TestGeneralRouting(unittest.TestCase):
    """F. GENERAL → general → reviewer."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_general_reaches_general_and_reviewer(self) -> None:
        compiled = _build_tracked_graph("GENERAL")
        result = compiled.invoke(_base_state())
        self.assertIn("general", _execution_log)
        self.assertNotIn("rag", _execution_log)
        self.assertIn("reviewer", _execution_log)
        self.assertEqual(result["intent"], "GENERAL")


# ===========================================================================
# G. State Preservation
# ===========================================================================


class TestStatePreservation(unittest.TestCase):
    """Verify that important fields survive graph execution."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_student_id_preserved(self) -> None:
        compiled = _build_tracked_graph("GENERAL")
        result = compiled.invoke(_base_state(student_id="24CSE042"))
        self.assertEqual(result["student_id"], "24CSE042")

    def test_question_preserved(self) -> None:
        compiled = _build_tracked_graph("ACADEMIC_QUERY")
        result = compiled.invoke(_base_state(question="What is the attendance policy?"))
        self.assertEqual(result["question"], "What is the attendance policy?")

    def test_intent_set_after_execution(self) -> None:
        compiled = _build_tracked_graph("CGPA")
        result = compiled.invoke(_base_state())
        self.assertEqual(result["intent"], "CGPA")

    def test_all_core_fields_present(self) -> None:
        compiled = _build_tracked_graph("SCHEDULE")
        result = compiled.invoke(
            _base_state(student_id="24CSE099", question="Show my timetable")
        )
        self.assertEqual(result["student_id"], "24CSE099")
        self.assertEqual(result["question"], "Show my timetable")
        self.assertEqual(result["intent"], "SCHEDULE")


# ===========================================================================
# H. Reviewer Execution
# ===========================================================================


class TestReviewerExecution(unittest.TestCase):
    """Verify that every valid branch reaches the reviewer before END."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_reviewer_runs_after_every_branch(self) -> None:
        """All five capability branches must end with the reviewer."""
        for intent, expected_node in [
            ("ACADEMIC_QUERY", "rag"),
            ("STUDY_PLAN", "study_plan"),
            ("CGPA", "cgpa"),
            ("SCHEDULE", "schedule"),
            ("GENERAL", "general"),
        ]:
            with self.subTest(intent=intent):
                _execution_log.clear()
                compiled = _build_tracked_graph(intent)
                compiled.invoke(_base_state())
                self.assertIn(expected_node, _execution_log)
                self.assertIn("reviewer", _execution_log)
                # Reviewer must execute AFTER the capability node
                cap_idx = _execution_log.index(expected_node)
                rev_idx = _execution_log.index("reviewer")
                self.assertGreater(
                    rev_idx, cap_idx,
                    f"Reviewer should execute after {expected_node}"
                )


# ===========================================================================
# I. Invalid Intent
# ===========================================================================


class TestInvalidIntent(unittest.TestCase):
    """Verify that unsupported intents raise errors, not silent fallback."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_unknown_intent_does_not_route_to_general(self) -> None:
        """An unsupported intent must raise an error, not silently fall back."""
        compiled = _build_tracked_graph("UNKNOWN_INTENT")
        with self.assertRaises(Exception):
            compiled.invoke(_base_state())
        # GENERAL node must NOT have been reached
        self.assertNotIn("general", _execution_log)

    def test_empty_intent_raises_error(self) -> None:
        """Empty string intent must raise an error."""
        compiled = _build_tracked_graph("")
        with self.assertRaises(Exception):
            compiled.invoke(_base_state())

    def test_none_intent_raises_error(self) -> None:
        """None intent must raise an error."""

        def none_analyzer(state: AgentState) -> AgentState:
            state["intent"] = None
            return state

        from langgraph.graph import END, START, StateGraph

        from agents.router import route_question

        graph = StateGraph(AgentState)
        graph.add_node("analyzer", none_analyzer)
        graph.add_node("rag", rag_node)
        graph.add_node("study_plan", study_plan_node)
        graph.add_node("cgpa", cgpa_node)
        graph.add_node("schedule", schedule_node)
        graph.add_node("general", general_node)
        graph.add_node("reviewer", reviewer_node)
        graph.add_edge(START, "analyzer")
        graph.add_conditional_edges("analyzer", route_question, ROUTE_TO_NODE)
        for node_name in ROUTE_TO_NODE.values():
            graph.add_edge(node_name, "reviewer")
        graph.add_edge("reviewer", END)
        compiled = graph.compile()

        with self.assertRaises(Exception):
            compiled.invoke(_base_state())


# ===========================================================================
# Route-to-node mapping consistency
# ===========================================================================


class TestRouteToNodeMapping(unittest.TestCase):
    """Verify the ROUTE_TO_NODE constant is consistent."""

    def test_all_five_routes_mapped(self) -> None:
        expected_routes = {"RAG", "STUDY_PLAN", "CGPA", "SCHEDULE", "GENERAL"}
        self.assertEqual(set(ROUTE_TO_NODE.keys()), expected_routes)

    def test_node_names_are_lowercase(self) -> None:
        for route, node in ROUTE_TO_NODE.items():
            self.assertEqual(node, node.lower(), f"Node name for {route} should be lowercase")


# ===========================================================================
# J. RAG Integration Tests (mocked — no live API calls)
# ===========================================================================


class TestRagNodeFoundTrue(unittest.TestCase):
    """TEST 2: RAG returns found=True — context, sources, and LLM answer reach state."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_rag_found_true_populates_state(self) -> None:
        """When RAG finds evidence, context/sources/answer should be set."""
        from unittest.mock import MagicMock, patch

        fake_retrieval = {
            "found": True,
            "context": "Semester 5 CSE has subjects: DBMS, OS.",
            "sources": [{"document": "curriculum__CSE__5.pdf", "page": 1, "score": 0.92}],
            "query": "What are the subjects in semester 5?",
            "filters": {"department": "CSE", "semester": 5, "subject": None},
            "best_score": 0.92,
        }
        fake_profile = {"student_id": "STU001", "name": "Alex Rivera",
                        "department": "Computer Science", "semester": 4,
                        "year": 2, "cgpa": 8.45}
        mock_llm_response = MagicMock()
        mock_llm_response.content = "The subjects in semester 5 for CSE are DBMS and OS."

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval) as mock_rc, \
             patch("agents.graph.get_student_profile", return_value=fake_profile), \
             patch("agents.graph.get_groq_llm") as mock_llm_factory, \
             patch("agents.graph.get_rag_chat_prompt") as mock_prompt_factory:

            # Set up the chain mock: prompt | llm
            mock_prompt = MagicMock()
            mock_llm = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_llm_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_factory.return_value = mock_prompt
            mock_llm_factory.return_value = mock_llm

            result = rag_node({
                "student_id": "STU001",
                "question": "What are the subjects in semester 5?",
                "intent": "ACADEMIC_QUERY",
            })

        self.assertEqual(result["context"], "Semester 5 CSE has subjects: DBMS, OS.")
        self.assertEqual(result["sources"], [{"document": "curriculum__CSE__5.pdf", "page": 1, "score": 0.92}])
        self.assertEqual(result["answer"], "The subjects in semester 5 for CSE are DBMS and OS.")
        self.assertEqual(result["student_id"], "STU001")
        self.assertEqual(result["question"], "What are the subjects in semester 5?")
        self.assertEqual(result["intent"], "ACADEMIC_QUERY")


class TestRagNodeFoundFalse(unittest.TestCase):
    """TEST 3: RAG returns found=False — no LLM call, bounded answer."""

    def test_rag_found_false_no_llm_call(self) -> None:
        """When RAG finds no evidence, LLM must NOT be called."""
        from unittest.mock import patch

        from rag.prompts import NOT_FOUND_MESSAGE

        fake_retrieval = {
            "found": False,
            "context": "",
            "sources": [],
            "query": "What is the canteen menu?",
            "filters": {"department": None, "semester": None, "subject": None},
            "best_score": None,
        }

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval), \
             patch("agents.graph.get_student_profile", return_value=None), \
             patch("agents.graph.get_groq_llm") as mock_llm_factory:

            result = rag_node({
                "student_id": "UNKNOWN",
                "question": "What is the canteen menu?",
                "intent": "ACADEMIC_QUERY",
            })

        # LLM factory must NOT have been called
        mock_llm_factory.assert_not_called()

        self.assertEqual(result["context"], "")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], NOT_FOUND_MESSAGE)


class TestRagNodeStudentProfileFilters(unittest.TestCase):
    """TEST 4: Student profile department/semester are passed to retrieve_context."""

    def test_profile_filters_forwarded(self) -> None:
        from unittest.mock import MagicMock, patch

        fake_profile = {"student_id": "STU003", "name": "Charlie Das",
                        "department": "Electronics and Communication", "semester": 2,
                        "year": 1, "cgpa": 7.80}
        fake_retrieval = {
            "found": False, "context": "", "sources": [],
            "query": "test", "filters": {}, "best_score": None,
        }

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval) as mock_rc, \
             patch("agents.graph.get_student_profile", return_value=fake_profile):

            rag_node({
                "student_id": "STU003",
                "question": "test",
                "intent": "ACADEMIC_QUERY",
            })

        mock_rc.assert_called_once_with(
            query="test",
            department="Electronics and Communication",
            semester=2,
        )

    def test_no_profile_passes_none_filters(self) -> None:
        """When student profile is not found, department/semester should be None."""
        from unittest.mock import patch

        fake_retrieval = {
            "found": False, "context": "", "sources": [],
            "query": "test", "filters": {}, "best_score": None,
        }

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval) as mock_rc, \
             patch("agents.graph.get_student_profile", return_value=None):

            rag_node({
                "student_id": "NONEXISTENT",
                "question": "test",
                "intent": "ACADEMIC_QUERY",
            })

        mock_rc.assert_called_once_with(
            query="test",
            department=None,
            semester=None,
        )

    def test_unknown_student_raises_not_found_handled_gracefully(self) -> None:
        """When student profile raises StudentNotFoundError, department/semester are None and graph does not crash."""
        from unittest.mock import patch
        from student.registry import StudentNotFoundError

        fake_retrieval = {
            "found": False, "context": "", "sources": [],
            "query": "test", "filters": {}, "best_score": None,
        }

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval) as mock_rc, \
             patch("agents.graph.get_student_profile", side_effect=StudentNotFoundError("Unknown student ID: 'UNKNOWN999'")):

            result = rag_node({
                "student_id": "UNKNOWN999",
                "question": "test",
                "intent": "ACADEMIC_QUERY",
            })

        mock_rc.assert_called_once_with(
            query="test",
            department=None,
            semester=None,
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["sources"], [])


class TestOtherRoutesUnchanged(unittest.TestCase):
    """TEST 5: Non-RAG routes remain placeholder pass-throughs."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_study_plan_still_passthrough(self) -> None:
        state = _base_state()
        result = study_plan_node(state)
        self.assertIs(result, state)

    def test_cgpa_still_passthrough(self) -> None:
        state = _base_state()
        result = cgpa_node(state)
        self.assertIs(result, state)

    def test_schedule_still_passthrough(self) -> None:
        state = _base_state()
        result = schedule_node(state)
        self.assertIs(result, state)

    def test_general_still_passthrough(self) -> None:
        state = _base_state()
        result = general_node(state)
        self.assertIs(result, state)


class TestRagNodeInGraph(unittest.TestCase):
    """TEST 1: ACADEMIC_QUERY reaches the RAG node within the full graph."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_academic_query_reaches_rag_in_full_graph(self) -> None:
        """End-to-end: ACADEMIC_QUERY routes through rag_node in the compiled graph."""
        from unittest.mock import MagicMock, patch

        fake_retrieval = {
            "found": True,
            "context": "Some curriculum context.",
            "sources": [{"document": "doc.pdf", "page": 1, "score": 0.9}],
            "query": "test q", "filters": {}, "best_score": 0.9,
        }
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Grounded answer from LLM."

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval), \
             patch("agents.graph.get_student_profile", return_value=None), \
             patch("agents.graph.get_groq_llm") as mock_llm_factory, \
             patch("agents.graph.get_rag_chat_prompt") as mock_prompt_factory:

            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_llm_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_factory.return_value = mock_prompt
            mock_llm_factory.return_value = MagicMock()

            compiled = build_graph(analyzer_fn=_make_fake_analyzer("ACADEMIC_QUERY"))
            result = compiled.invoke(_base_state(question="test q"))

        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["context"], "Some curriculum context.")
        self.assertEqual(result["answer"], "Grounded answer from LLM.")
        self.assertEqual(result["student_id"], "24CSE001")

    def test_unknown_student_in_full_graph_completes_without_error(self) -> None:
        """End-to-end: An unknown student ID in ACADEMIC_QUERY does not crash the compiled graph."""
        from unittest.mock import MagicMock, patch
        from student.registry import StudentNotFoundError

        fake_retrieval = {
            "found": True,
            "context": "Context for general query.",
            "sources": [{"document": "general.pdf", "page": 1, "score": 0.85}],
            "query": "test q", "filters": {}, "best_score": 0.85,
        }
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Answer without student filtering."

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval) as mock_rc, \
             patch("agents.graph.get_student_profile", side_effect=StudentNotFoundError("Unknown student")):

            with patch("agents.graph.get_groq_llm") as mock_llm_factory, \
                 patch("agents.graph.get_rag_chat_prompt") as mock_prompt_factory:

                mock_prompt = MagicMock()
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = mock_llm_response
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt_factory.return_value = mock_prompt
                mock_llm_factory.return_value = MagicMock()

                compiled = build_graph(analyzer_fn=_make_fake_analyzer("ACADEMIC_QUERY"))
                result = compiled.invoke(_base_state(student_id="UNKNOWN_SID", question="test q"))

        mock_rc.assert_called_once_with(
            query="test q",
            department=None,
            semester=None,
        )
        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["answer"], "Answer without student filtering.")
        self.assertEqual(result["student_id"], "UNKNOWN_SID")


class TestReviewerIntegrationInGraph(unittest.TestCase):
    """Verify that the real reviewer is integrated and invoked by the graph."""

    def setUp(self) -> None:
        _execution_log.clear()

    def test_reviewer_node_delegates_to_review_response(self) -> None:
        """Verify reviewer_node calls review_response."""
        from unittest.mock import patch
        test_state = _base_state(intent="GENERAL", answer="Hello there")
        with patch("agents.graph.review_response", return_value={"mocked": True}) as mock_rr:
            result = reviewer_node(test_state)
            mock_rr.assert_called_once_with(test_state)
            self.assertEqual(result, {"mocked": True})

    def test_academic_query_with_valid_evidence_preserves_answer(self) -> None:
        """ACADEMIC_QUERY with valid context, sources, and answer is preserved by reviewer in graph."""
        from unittest.mock import MagicMock, patch
        from agents.reviewer import FALLBACK_ACADEMIC

        fake_retrieval = {
            "found": True,
            "context": "Passing grade is 40% in each subject.",
            "sources": [{"document": "exam_rules.pdf", "page": 4, "score": 0.95}],
            "query": "What is the passing grade?",
            "filters": {},
            "best_score": 0.95,
        }
        mock_llm_response = MagicMock()
        mock_llm_response.content = "The minimum passing grade is 40% according to institutional rules."

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval), \
             patch("agents.graph.get_student_profile", return_value=None), \
             patch("agents.graph.get_groq_llm") as mock_llm_factory, \
             patch("agents.graph.get_rag_chat_prompt") as mock_prompt_factory:

            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_llm_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_factory.return_value = mock_prompt
            mock_llm_factory.return_value = MagicMock()

            compiled = build_graph(analyzer_fn=_make_fake_analyzer("ACADEMIC_QUERY"))
            result = compiled.invoke(_base_state(question="What is the passing grade?"))

        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["context"], "Passing grade is 40% in each subject.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["answer"], "The minimum passing grade is 40% according to institutional rules.")
        self.assertNotEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_academic_query_missing_context_or_sources_triggers_fallback(self) -> None:
        """When RAG retrieval finds no context, reviewer replaces answer with FALLBACK_ACADEMIC."""
        from unittest.mock import patch
        from agents.reviewer import FALLBACK_ACADEMIC

        fake_retrieval = {
            "found": False,
            "context": "",
            "sources": [],
            "query": "Unknown topic",
            "filters": {},
            "best_score": None,
        }

        with patch("agents.graph.retrieve_context", return_value=fake_retrieval), \
             patch("agents.graph.get_student_profile", return_value=None):

            compiled = build_graph(analyzer_fn=_make_fake_analyzer("ACADEMIC_QUERY"))
            result = compiled.invoke(_base_state(question="Unknown topic"))

        self.assertEqual(result["intent"], "ACADEMIC_QUERY")
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_academic_query_with_answer_but_missing_sources_triggers_fallback(self) -> None:
        """If RAG node produced an answer and context but sources is empty, reviewer falls back."""
        from agents.reviewer import FALLBACK_ACADEMIC

        ungrounded_state = _base_state(
            intent="ACADEMIC_QUERY",
            context="Some context",
            sources=[],
            answer="Invented claim",
        )
        result = reviewer_node(ungrounded_state)
        self.assertEqual(result["answer"], FALLBACK_ACADEMIC)

    def test_non_academic_routes_handled_by_reviewer(self) -> None:
        """Non-academic routes receive appropriate fallbacks from reviewer."""
        from agents.reviewer import (
            FALLBACK_CGPA,
            FALLBACK_GENERAL,
            FALLBACK_SCHEDULE,
            FALLBACK_STUDY_PLAN,
        )

        for intent, expected_fallback in [
            ("CGPA", FALLBACK_CGPA),
            ("STUDY_PLAN", FALLBACK_STUDY_PLAN),
            ("SCHEDULE", FALLBACK_SCHEDULE),
            ("GENERAL", FALLBACK_GENERAL),
        ]:
            with self.subTest(intent=intent):
                compiled = build_graph(analyzer_fn=_make_fake_analyzer(intent))
                result = compiled.invoke(_base_state(question="test"))
                self.assertEqual(result["intent"], intent)
                self.assertEqual(result["answer"], expected_fallback)


if __name__ == "__main__":
    unittest.main()

