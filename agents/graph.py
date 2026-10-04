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
from agents.reviewer import review_response
from agents.router import route_question
from agents.state import AgentState
from llm.groq import get_groq_llm
from rag.prompts import NOT_FOUND_MESSAGE, get_rag_chat_prompt
from rag.retriever import retrieve_context
from student.registry import StudentNotFoundError, get_student_profile

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
    """RAG-backed academic query node.

    1. Looks up the student profile for department/semester filters.
    2. Calls ``retrieve_context`` with the question and filters.
    3. If no evidence is found, sets a bounded no-information answer
       without calling the LLM.
    4. If evidence is found, generates a grounded answer via the
       existing Groq/LangChain prompt pipeline.
    """
    question = state["question"]
    student_id = state.get("student_id", "")

    # --- Student profile for optional filters ---
    department = None
    semester = None
    try:
        profile = get_student_profile(student_id)
    except StudentNotFoundError:
        profile = None

    if profile:
        department = profile.get("department")
        semester = profile.get("semester")

    # --- RAG retrieval ---
    rag_result = retrieve_context(
        query=question,
        department=department,
        semester=semester,
    )

    if not rag_result["found"]:
        state["context"] = ""
        state["sources"] = []
        state["answer"] = NOT_FOUND_MESSAGE
        return state

    # --- Grounded LLM generation ---
    state["context"] = rag_result["context"]
    state["sources"] = rag_result["sources"]

    llm = get_groq_llm()
    prompt = get_rag_chat_prompt()
    chain = prompt | llm
    response = chain.invoke({
        "context": rag_result["context"],
        "question": question,
    })
    state["answer"] = response.content
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
# Reviewer node
# ---------------------------------------------------------------------------


def reviewer_node(state: AgentState) -> AgentState:
    """Validate and review the generated response via review_response."""
    return review_response(state)


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
