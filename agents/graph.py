"""
LangGraph Workflow Orchestration Module
Owned by: Member 2

Responsibilities:
- LangGraph workflow construction and compilation
- Connects: Question Analyzer → Router → Capability Branch → Reviewer
- Placeholder capability and reviewer nodes for future implementation
- Future contract:
    run_agent(student_id, question)

Workflow:

    START
      ↓
    analyzer
      ↓
    conditional routing (route_question)
      ↓
    one of: rag | study_plan | cgpa | schedule | general
      ↓
    reviewer
      ↓
    END
"""

from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents.question_analyzer import analyze_question
from agents.router import route_question
from agents.state import AgentState

# ---------------------------------------------------------------------------
# Route identifier → graph node name mapping
# ---------------------------------------------------------------------------

ROUTE_TO_NODE: dict[str, str] = {
    "RAG": "rag",
    "STUDY_PLAN": "study_plan",
    "CGPA": "cgpa",
    "SCHEDULE": "schedule",
    "GENERAL": "general",
}

# ---------------------------------------------------------------------------
# Placeholder capability nodes
# ---------------------------------------------------------------------------


def rag_node(state: AgentState) -> AgentState:
    """Placeholder for RAG capability. Returns state unchanged."""
    return state


def study_plan_node(state: AgentState) -> AgentState:
    """Placeholder for Study Plan capability. Returns state unchanged."""
    return state


def cgpa_node(state: AgentState) -> AgentState:
    """Placeholder for CGPA capability. Returns state unchanged."""
    return state


def schedule_node(state: AgentState) -> AgentState:
    """Placeholder for Schedule capability. Returns state unchanged."""
    return state


def general_node(state: AgentState) -> AgentState:
    """Placeholder for General capability. Returns state unchanged."""
    return state


# ---------------------------------------------------------------------------
# Placeholder reviewer node
# ---------------------------------------------------------------------------


def reviewer_node(state: AgentState) -> AgentState:
    """Placeholder for Reviewer. Returns state unchanged."""
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    analyzer_fn: Callable[[AgentState], AgentState] = analyze_question,
) -> StateGraph:
    """Build and compile the CampusAI LangGraph workflow.

    Args:
        analyzer_fn: The question analyzer node function. Defaults to the
            production ``analyze_question``. Override with a deterministic
            function for testing without Groq API calls.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for invocation.
    """
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("analyzer", analyzer_fn)
    graph.add_node("rag", rag_node)
    graph.add_node("study_plan", study_plan_node)
    graph.add_node("cgpa", cgpa_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("general", general_node)
    graph.add_node("reviewer", reviewer_node)

    # --- Edges ---
    # START → analyzer
    graph.add_edge(START, "analyzer")

    # analyzer → conditional routing via route_question
    graph.add_conditional_edges(
        "analyzer",
        route_question,
        ROUTE_TO_NODE,
    )

    # Each capability branch → reviewer
    for node_name in ROUTE_TO_NODE.values():
        graph.add_edge(node_name, "reviewer")

    # reviewer → END
    graph.add_edge("reviewer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Default compiled graph (production)
# ---------------------------------------------------------------------------
# NOTE: This uses the real analyze_question which requires Groq.
# For tests, call build_graph() with a deterministic analyzer_fn.

agent_graph = build_graph()
