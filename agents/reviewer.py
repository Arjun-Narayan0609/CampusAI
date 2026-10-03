"""
Reviewer and Response Validation Module
Owned by: Member 2

Responsibilities:
- Validate, verify, and format generated responses against quality and groundedness criteria
- Deterministic evidence-presence checks (no LLM calls)
- Enforce no-invention rule: block unsupported institutional claims
- Intent-specific validation for ACADEMIC_QUERY, CGPA, STUDY_PLAN, SCHEDULE, GENERAL

Review rules by intent:
    ACADEMIC_QUERY → requires context + sources + answer
    CGPA           → requires tool_result + answer
    STUDY_PLAN     → requires tool_result + answer
    SCHEDULE       → requires tool_result + answer
    GENERAL        → requires answer only (no institutional claims)
"""

from typing import Any, Collection

from agents.state import AgentState

# ---------------------------------------------------------------------------
# Bounded fallback messages
# ---------------------------------------------------------------------------

FALLBACK_ACADEMIC = (
    "I couldn't verify that information from the provided institutional documents. "
    "Please consult official college resources for accurate details."
)

FALLBACK_CGPA = (
    "I couldn't provide a verified CGPA/SGPA result because the "
    "calculation tool did not return a result."
)

FALLBACK_STUDY_PLAN = (
    "I couldn't produce a verified study plan because the "
    "planning tool did not return a result."
)

FALLBACK_SCHEDULE = (
    "I couldn't retrieve or modify your schedule because the "
    "scheduling tool did not return a result."
)

FALLBACK_GENERAL = (
    "I'm sorry, I wasn't able to generate a response. "
    "Please try rephrasing your question."
)

FALLBACK_UNKNOWN_INTENT = (
    "I encountered an unexpected request type and cannot provide "
    "a verified response."
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_value(value: Any) -> bool:
    """Check whether a state field contains meaningful, non-empty data.

    Returns False for None, empty strings, empty collections, and
    whitespace-only strings. Returns True otherwise.
    """
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, Collection) and not isinstance(value, str) and len(value) == 0:
        return False
    return True


def _review_academic_query(state: AgentState) -> AgentState:
    """Validate an ACADEMIC_QUERY response for grounding.

    Requires all three of context, sources, and answer to be present.
    If any evidence component is missing, replaces the answer with a
    bounded fallback — never invents institutional information.
    """
    context = state.get("context")
    sources = state.get("sources")
    answer = state.get("answer")

    if not _has_value(answer):
        state["answer"] = FALLBACK_ACADEMIC
        return state

    if not _has_value(context) or not _has_value(sources):
        state["answer"] = FALLBACK_ACADEMIC
        return state

    # All evidence present — answer passes review unchanged
    return state


def _review_tool_based(state: AgentState, fallback: str) -> AgentState:
    """Validate a tool-dependent response (CGPA, STUDY_PLAN, SCHEDULE).

    Checks that tool_result exists and is non-empty. If missing,
    replaces the answer with the provided fallback message.
    """
    tool_result = state.get("tool_result")
    answer = state.get("answer")

    if not _has_value(answer):
        state["answer"] = fallback
        return state

    if not _has_value(tool_result):
        state["answer"] = fallback
        return state

    # Tool result present — answer passes review unchanged
    return state


def _review_general(state: AgentState) -> AgentState:
    """Validate a GENERAL response.

    GENERAL questions do not require institutional evidence or tool
    results. Only checks that an answer is present.
    """
    answer = state.get("answer")

    if not _has_value(answer):
        state["answer"] = FALLBACK_GENERAL
        return state

    return state


# ---------------------------------------------------------------------------
# Intent → review function dispatch
# ---------------------------------------------------------------------------

_REVIEW_DISPATCH = {
    "ACADEMIC_QUERY": _review_academic_query,
    "CGPA": lambda state: _review_tool_based(state, FALLBACK_CGPA),
    "STUDY_PLAN": lambda state: _review_tool_based(state, FALLBACK_STUDY_PLAN),
    "SCHEDULE": lambda state: _review_tool_based(state, FALLBACK_SCHEDULE),
    "GENERAL": _review_general,
}


# ---------------------------------------------------------------------------
# Primary reviewer function (LangGraph node)
# ---------------------------------------------------------------------------


def review_response(state: AgentState) -> AgentState:
    """Validate and review a generated response before presenting to the student.

    This is a deterministic validation layer — no LLM calls.

    Reads the intent from state and dispatches to the appropriate
    review function. Each review function enforces evidence-presence
    checks and replaces unverified answers with bounded fallback
    messages.

    Core invariant (no-invention rule):
        If required evidence or tool results are missing, the reviewer
        NEVER invents institutional facts, CGPA calculations, study
        sessions, or schedule modifications. It replaces the answer
        with a bounded "information unavailable" response.

    State preservation:
        All existing fields (student_id, question, intent, context,
        tool_result, sources) are preserved. Only the 'answer' field
        may be modified when validation fails.

    Args:
        state: Shared AgentState containing the response to validate.

    Returns:
        AgentState with the answer either preserved (valid) or replaced
        with a bounded fallback (invalid/unverified).
    """
    intent = state.get("intent")

    # Unknown or missing intent — cannot validate, provide safe fallback
    if not _has_value(intent):
        state["answer"] = FALLBACK_UNKNOWN_INTENT
        return state

    review_fn = _REVIEW_DISPATCH.get(intent)  # type: ignore[arg-type]

    if review_fn is None:
        # Unrecognised intent — do not silently pass
        state["answer"] = FALLBACK_UNKNOWN_INTENT
        return state

    return review_fn(state)
