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


if __name__ == "__main__":
    unittest.main()
