"""
Conditional Router Module
Owned by: Member 2

Responsibilities:
- Deterministic routing based on classified intent from the Question Analyzer.
- Maps each Phase-1 intent to a stable downstream route identifier.
- Designed for use as a LangGraph conditional edge function.

Usage as a LangGraph conditional edge (future graph.py):
    graph.add_conditional_edges("question_analyzer", route_question, { ... })
"""

from typing import Literal, get_args

from agents.state import AgentState

# ---------------------------------------------------------------------------
# Route identifiers — stable constants consumed by LangGraph conditional edges
# ---------------------------------------------------------------------------

RouteType = Literal[
    "RAG",
    "STUDY_PLAN",
    "CGPA",
    "SCHEDULE",
    "GENERAL",
]

VALID_ROUTES: set[str] = set(get_args(RouteType))

# ---------------------------------------------------------------------------
# Intent → Route mapping
# ---------------------------------------------------------------------------

INTENT_TO_ROUTE: dict[str, str] = {
    "ACADEMIC_QUERY": "RAG",
    "STUDY_PLAN": "STUDY_PLAN",
    "CGPA": "CGPA",
    "SCHEDULE": "SCHEDULE",
    "GENERAL": "GENERAL",
}

SUPPORTED_INTENTS: set[str] = set(INTENT_TO_ROUTE.keys())


# ---------------------------------------------------------------------------
# Router function
# ---------------------------------------------------------------------------


def route_question(state: AgentState) -> str:
    """Route a classified question to the appropriate downstream capability.

    Reads ``state["intent"]`` (populated by the Question Analyzer) and returns
    the corresponding route identifier.  This function is deterministic — it
    performs no LLM calls and does not inspect the original question text.

    Intended to be plugged directly into a LangGraph conditional edge::

        graph.add_conditional_edges(
            "question_analyzer",
            route_question,
            {
                "RAG":        "rag_node",
                "STUDY_PLAN": "study_plan_node",
                "CGPA":       "cgpa_node",
                "SCHEDULE":   "schedule_node",
                "GENERAL":    "general_node",
            },
        )

    Args:
        state: Shared ``AgentState`` dictionary.  Must contain an ``"intent"``
            key whose value is one of the supported Phase-1 intents.

    Returns:
        A route identifier string (one of ``RAG``, ``STUDY_PLAN``, ``CGPA``,
        ``SCHEDULE``, ``GENERAL``).

    Raises:
        ValueError: If ``"intent"`` is missing, empty, or not a recognised
            Phase-1 intent.
    """
    # 1. Validate that intent exists in state
    if "intent" not in state or state["intent"] is None:
        raise ValueError(
            "AgentState is missing 'intent'. "
            "Ensure the Question Analyzer has run before the router."
        )

    intent: str = state["intent"]

    # 2. Validate that intent is a non-empty string
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError(
            f"AgentState 'intent' must be a non-empty string, got: {intent!r}"
        )

    intent = intent.strip()

    # 3. Map intent to route
    route = INTENT_TO_ROUTE.get(intent)

    if route is None:
        raise ValueError(
            f"Unsupported intent '{intent}'. "
            f"Must be one of {sorted(SUPPORTED_INTENTS)}."
        )

    return route
