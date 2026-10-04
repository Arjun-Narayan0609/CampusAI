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

import re
from datetime import date
from typing import Callable

from groq import GroqError
from langchain_core.exceptions import LangChainException
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from agents.question_analyzer import analyze_question
from agents.reviewer import FALLBACK_GENERAL, review_response
from agents.router import route_question
from agents.state import AgentState
from llm.groq import get_groq_llm
from rag.prompts import NOT_FOUND_MESSAGE, get_rag_chat_prompt
from rag.retriever import retrieve_context
from student.registry import RegistryError, StudentNotFoundError, get_student_profile
from tools.cgpa import GradeError, calculate_cgpa, calculate_sgpa
from tools.planner import (
    create_study_plan,
    reschedule_unavailable_day,
    validate_plan,
)
from tools.scheduler import (
    ScheduleConflictError,
    ScheduleError,
    SessionNotFoundError,
    WEEKDAYS,
    add_session,
    complete_session,
    find_free_slot,
    fmt_minutes,
    get_schedule,
    miss_session,
    modify_schedule,
    modify_session,
    resolve_weekday,
)

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
    """Deterministic Study Plan capability node.

    Invokes Member 3's deterministic study planner (create_study_plan),
    validates the generated plan (validate_plan), and derives a grounded,
    user-facing answer strictly from the planner output.
    """
    student_id = state.get("student_id")
    question = state.get("question", "")
    q_lower = question.lower()
    metadata = state.get("metadata") or {}

    if not student_id or not str(student_id).strip():
        state["tool_result"] = {"ok": False, "error": "Missing student ID"}
        state["answer"] = "Student ID is required to generate a study plan."
        return state

    # Extract optional planner parameters if provided in state/metadata
    today = state.get("today") or metadata.get("today")
    if isinstance(today, str):
        try:
            today = date.fromisoformat(today)
        except ValueError:
            today = None

    persist = state.get("persist")
    if persist is None:
        persist = metadata.get("persist", True)

    weights = state.get("weights") or metadata.get("weights")
    blocked_dates = state.get("blocked_dates") or metadata.get("blocked_dates")
    action = state.get("action") or metadata.get("action")

    # Check for reschedule unavailable day action / keywords
    if action == "reschedule_unavailable_day" or any(kw in q_lower for kw in ("cannot study", "unavailable", "can't study")):
        target_date = state.get("unavailable_date") or metadata.get("unavailable_date")
        if not target_date:
            iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
            if iso_match:
                try:
                    target_date = date.fromisoformat(iso_match.group(1))
                except ValueError:
                    target_date = None
            else:
                for wd in WEEKDAYS:
                    if re.search(rf"\b{wd}\b", q_lower):
                        try:
                            target_date = resolve_weekday(wd)
                        except ScheduleError:
                            target_date = None
                        break

        if target_date is None:
            state["tool_result"] = {"ok": False, "error": "Missing date to mark unavailable"}
            state["answer"] = "Please specify which date or weekday you cannot study (e.g., 'I cannot study Wednesday' or '2026-10-07')."
            return state

        try:
            res = reschedule_unavailable_day(student_id, target_date, today=today)
            res.setdefault("ok", True)
            state["tool_result"] = res
            strategy = res.get("strategy", "unknown")
            dropped = res.get("sessions_dropped", 0)
            date_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
            if strategy == "moved":
                state["answer"] = f"Rescheduled study sessions away from {date_str} into free slots on other days (0 sessions dropped)."
            else:
                state["answer"] = f"Replanned your study timetable with {date_str} marked unavailable ({dropped} session(s) dropped)."
            return state
        except StudentNotFoundError:
            state["tool_result"] = {"ok": False, "error": f"Student '{student_id}' not found in registry"}
            state["answer"] = f"No profile found for student ID '{student_id}'. Cannot reschedule study plan."
            return state
        except RegistryError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not access student profile data: {e}"
            return state
        except (ScheduleError, SessionNotFoundError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Scheduling error while rescheduling: {e}"
            return state
        except ValueError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Invalid date or parameter for rescheduling: {e}"
            return state

    # Primary workflow: generate deterministic study plan
    try:
        plan = create_study_plan(
            student_id=student_id,
            today=today,
            persist=persist,
            weights=weights,
            blocked_dates=blocked_dates,
        )
    except StudentNotFoundError:
        state["tool_result"] = {"ok": False, "error": f"Student '{student_id}' not found in registry"}
        state["answer"] = f"No profile found for student ID '{student_id}'. Cannot generate a study plan."
        return state
    except RegistryError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not access student profile data: {e}"
        return state
    except (ScheduleError, SessionNotFoundError) as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Scheduling error while generating study plan: {e}"
        return state
    except ValueError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Invalid date or parameter for study plan: {e}"
        return state

    # Validate the generated plan using Member 3's validate_plan
    issues = validate_plan(plan)
    plan["issues"] = issues
    plan.setdefault("ok", True)
    state["tool_result"] = plan

    sessions = plan.get("plan", [])
    per_subject = plan.get("per_subject", [])
    warnings = plan.get("warnings", [])
    feasible = plan.get("feasible", True)

    if issues:
        state["answer"] = f"Study plan generated with validation issues: {'; '.join(issues)}."
    elif not sessions:
        if warnings:
            state["answer"] = f"Could not generate study plan: {warnings[0]}"
        else:
            state["answer"] = "No study sessions could be scheduled based on your current exam dates and profile."
    else:
        first_date = sessions[0]["date"]
        last_date = sessions[-1]["date"]
        subj_count = len(per_subject)
        sess_count = len(sessions)
        subj_label = f" ({', '.join(p['subject'] for p in per_subject[:3])}{'...' if len(per_subject) > 3 else ''})" if per_subject else ""
        if not feasible and warnings:
            warn_msg = warnings[-1]
            state["answer"] = f"Generated a study plan with {sess_count} session(s) across {subj_count} subject(s){subj_label} from {first_date} to {last_date}. Note: {warn_msg}"
        else:
            state["answer"] = f"Generated a study plan with {sess_count} session(s) across {subj_count} subject(s){subj_label} from {first_date} to {last_date}."

    return state


def cgpa_node(state: AgentState) -> AgentState:
    """Deterministic CGPA/SGPA capability node.

    1. Obtains the student ID from state['student_id'].
    2. If explicit calculation inputs (semesters or courses) are provided,
       invokes calculate_cgpa or calculate_sgpa.
    3. Otherwise, looks up the student's profile via get_student_profile.
    4. Stores the resulting tool output in state['tool_result'].
    5. Produces a user-facing answer in state['answer'].
    6. Preserves student_id, question, and intent.
    """
    student_id = state.get("student_id", "")

    # Multi-semester CGPA calculation input
    if "semesters" in state and state["semesters"]:
        try:
            res = calculate_cgpa(state["semesters"])
            state["tool_result"] = res
            state["answer"] = f"Calculated CGPA is {res['cgpa']}. {res.get('formula', '')}"
            return state
        except GradeError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not calculate CGPA: {e}"
            return state

    # Course-level SGPA calculation input
    if "courses" in state and state["courses"]:
        try:
            res = calculate_sgpa(state["courses"])
            state["tool_result"] = res
            state["answer"] = f"Calculated SGPA is {res['sgpa']}. {res.get('formula', '')}"
            return state
        except GradeError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not calculate SGPA: {e}"
            return state

    # Standard CGPA lookup via student registry
    if not student_id:
        state["tool_result"] = {"ok": False, "error": "Missing student ID"}
        state["answer"] = "Could not retrieve CGPA because no student ID was provided."
        return state

    try:
        profile = get_student_profile(student_id)
    except StudentNotFoundError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not retrieve CGPA: student ID '{student_id}' was not found in the registry."
        return state
    except RegistryError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not retrieve CGPA due to registry error: {e}"
        return state

    cgpa_val = profile.get("cgpa")
    if cgpa_val is None:
        state["tool_result"] = {"ok": False, "error": f"No CGPA on record for student {student_id}"}
        state["answer"] = f"No CGPA record was found for student ID '{student_id}'."
        return state

    name = profile.get("name", student_id)
    sem = profile.get("semester")

    state["tool_result"] = {
        "ok": True,
        **profile,
    }
    sem_text = f" in Semester {sem}" if sem else ""
    state["answer"] = f"Student {name} ({profile.get('student_id', student_id)}) has a current CGPA of {cgpa_val}{sem_text}."
    return state


def schedule_node(state: AgentState) -> AgentState:
    """Deterministic Schedule capability node.

    1. Obtains student_id and question from state.
    2. Validates student presence via student registry contract.
    3. Dispatches to Member 3 deterministic scheduler operations based on
       explicit state fields or natural-language query cues.
    4. Populates state['tool_result'] and state['answer'].
    5. Preserves student_id, question, and intent.
    """
    student_id = state.get("student_id", "")
    question = state.get("question", "")

    # 1. Validate student_id
    if not student_id:
        state["tool_result"] = {"ok": False, "error": "Missing student ID"}
        state["answer"] = "Could not access schedule because no student ID was provided."
        return state

    try:
        profile = get_student_profile(student_id)
    except StudentNotFoundError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not retrieve schedule: student ID '{student_id}' was not found in the registry."
        return state
    except RegistryError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not retrieve schedule due to registry error: {e}"
        return state

    # 2. Check for programmatic/structured action payloads in state
    action = state.get("action")

    # Explicit modify_schedule action
    if action == "modify_schedule" or "unavailable_date" in state:
        unavailable_date = state.get("unavailable_date")
        move_to_date = state.get("move_to_date")
        try:
            res = modify_schedule(student_id, unavailable_date=unavailable_date, move_to_date=move_to_date)
            state["tool_result"] = res
            moved = res.get("moved", [])
            failed = res.get("failed", [])
            ans = f"Rescheduled {len(moved)} session(s) from {res.get('unavailable_date')}."
            if failed:
                ans += f" ({len(failed)} session(s) could not be relocated: {failed[0].get('reason', '')})."
            state["answer"] = ans
            return state
        except (ScheduleError, SessionNotFoundError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not modify schedule: {e}"
            return state

    # Explicit add_session action
    if action == "add_session" or all(k in state for k in ("subject", "session_date", "start_time", "duration_min")):
        try:
            res = add_session(
                student_id=student_id,
                subject=state["subject"],
                session_date=state["session_date"],
                start_time=state["start_time"],
                duration_min=state["duration_min"],
                note=state.get("note", ""),
            )
            state["tool_result"] = res
            state["answer"] = f"Scheduled {res['subject']} on {res['date']} from {res['start_time']} to {res['end_time']}."
            return state
        except (ScheduleError, ScheduleConflictError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not add session: {e}"
            return state

    # Explicit complete_session action
    if action == "complete_session" or "complete_session_id" in state:
        sess_id = state.get("complete_session_id") or state.get("session_id")
        try:
            res = complete_session(sess_id, student_id=student_id)
            state["tool_result"] = res
            state["answer"] = f"Session {sess_id} marked as completed."
            return state
        except (ScheduleError, SessionNotFoundError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not complete session: {e}"
            return state

    # Explicit miss_session action
    if action == "miss_session" or "miss_session_id" in state:
        sess_id = state.get("miss_session_id") or state.get("session_id")
        try:
            res = miss_session(sess_id, student_id=student_id)
            state["tool_result"] = res
            state["answer"] = f"Session {sess_id} marked as missed."
            return state
        except (ScheduleError, SessionNotFoundError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not mark session as missed: {e}"
            return state

    # Explicit find_free_slot action
    if action == "find_free_slot" or "free_slot_date" in state:
        slot_day = state.get("free_slot_date") or state.get("date")
        dur = int(state.get("duration_min", 60))
        pref = state.get("preferred_start", "18:00")
        try:
            start_m = find_free_slot(student_id, slot_day, dur, preferred_start=pref)
            if start_m is not None:
                fmt_t = fmt_minutes(start_m)
                res = {"tool": "find_free_slot", "student_id": student_id, "date": str(slot_day), "start_time": fmt_t, "duration_min": dur}
                state["tool_result"] = res
                state["answer"] = f"A free {dur}-minute slot is available on {slot_day} at {fmt_t}."
            else:
                res = {"tool": "find_free_slot", "student_id": student_id, "date": str(slot_day), "start_time": None, "duration_min": dur}
                state["tool_result"] = res
                state["answer"] = f"No free {dur}-minute slot found on {slot_day}."
            return state
        except ScheduleError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not search for free slot: {e}"
            return state

    # 3. Natural language query handling
    q_lower = question.lower() if question else ""

    # Check for reschedule / unavailability intent in question
    is_reschedule = any(kw in q_lower for kw in ("cannot study", "can't study", "reschedule", "unavailable", "move session"))
    if is_reschedule:
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
        target_date = None
        if iso_match:
            try:
                target_date = date.fromisoformat(iso_match.group(1))
            except ValueError:
                target_date = None
        else:
            for wd in WEEKDAYS:
                if re.search(rf"\b{wd}\b", q_lower):
                    try:
                        target_date = resolve_weekday(wd)
                    except ScheduleError:
                        target_date = None
                    break

        if target_date is None:
            state["tool_result"] = {"ok": False, "error": "Missing date to reschedule"}
            state["answer"] = "Please specify which day or date you would like to reschedule (e.g., 'I cannot study Wednesday' or 'reschedule 2026-10-07')."
            return state

        try:
            res = modify_schedule(student_id, unavailable_date=target_date)
            state["tool_result"] = res
            moved = res.get("moved", [])
            failed = res.get("failed", [])
            if moved:
                state["answer"] = f"Rescheduled {len(moved)} session(s) from {target_date.isoformat()}."
            elif failed:
                state["answer"] = f"Could not reschedule sessions from {target_date.isoformat()}: {failed[0].get('reason', 'no slot available')}."
            else:
                state["answer"] = f"No planned sessions found on {target_date.isoformat()} to reschedule."
            return state
        except (ScheduleError, SessionNotFoundError) as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not modify schedule: {e}"
            return state

    # Check for free slot search in question
    if any(kw in q_lower for kw in ("free slot", "available slot", "find slot")):
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
        target_date = None
        if iso_match:
            try:
                target_date = date.fromisoformat(iso_match.group(1))
            except ValueError:
                target_date = None
        else:
            for wd in WEEKDAYS:
                if re.search(rf"\b{wd}\b", q_lower):
                    try:
                        target_date = resolve_weekday(wd)
                    except ScheduleError:
                        target_date = None
                    break

        if target_date is None:
            state["tool_result"] = {"ok": False, "error": "Missing date for free slot search"}
            state["answer"] = "Please specify a date or weekday to check for available study slots."
            return state

        try:
            start_m = find_free_slot(student_id, target_date, duration_min=60)
            if start_m is not None:
                fmt_t = fmt_minutes(start_m)
                state["tool_result"] = {"tool": "find_free_slot", "student_id": student_id, "date": target_date.isoformat(), "start_time": fmt_t, "duration_min": 60}
                state["answer"] = f"A free 60-minute slot is available on {target_date.isoformat()} at {fmt_t}."
            else:
                state["tool_result"] = {"tool": "find_free_slot", "student_id": student_id, "date": target_date.isoformat(), "start_time": None, "duration_min": 60}
                state["answer"] = f"No free 60-minute study slot found on {target_date.isoformat()}."
            return state
        except ScheduleError as e:
            state["tool_result"] = {"ok": False, "error": str(e)}
            state["answer"] = f"Could not search for free slot: {e}"
            return state

    # Default: retrieve schedule
    start_d = state.get("start_date")
    end_d = state.get("end_date")
    status = state.get("status")

    # If specific date or weekday is mentioned in view question:
    date_label = ""
    if not start_d and not end_d:
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
        if iso_match:
            try:
                d_val = date.fromisoformat(iso_match.group(1))
                start_d = d_val
                end_d = d_val
                date_label = f" for {d_val.isoformat()}"
            except ValueError:
                pass
        else:
            for wd in WEEKDAYS:
                if re.search(rf"\b{wd}\b", q_lower):
                    try:
                        d_val = resolve_weekday(wd)
                        start_d = d_val
                        end_d = d_val
                        date_label = f" for {wd.capitalize()} ({d_val.isoformat()})"
                    except ScheduleError:
                        pass
                    break

    try:
        sessions = get_schedule(student_id, start_date=start_d, end_date=end_d, status=status)
    except ScheduleError as e:
        state["tool_result"] = {"ok": False, "error": str(e)}
        state["answer"] = f"Could not retrieve schedule: {e}"
        return state

    state["tool_result"] = {
        "ok": True,
        "tool": "get_schedule",
        "student_id": student_id,
        "sessions": sessions,
    }
    if sessions:
        items = [f"{s['subject']} on {s['date']} ({s['start_time']}-{s['end_time']})" for s in sessions[:3]]
        more = f" (and {len(sessions) - 3} more)" if len(sessions) > 3 else ""
        state["answer"] = f"You have {len(sessions)} session(s) scheduled{date_label}: {', '.join(items)}{more}."
    else:
        state["answer"] = f"You have no scheduled sessions{date_label}."
    return state


# ---------------------------------------------------------------------------
# General capability node and prompt
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """You are CampusAI, an academic assistant for students.
Answer general, conversational, or non-academic inquiries directly, politely, and concisely.

Rules:
1. Identify as CampusAI when appropriate.
2. Answer the user's question directly and concisely.
3. You do not have access to private student records or institutional documents in this general mode. Never invent or fabricate grades, CGPA, schedules, study plans, or institutional policies.
4. If the user asks for personal academic data (such as CGPA, timetable/schedule, or study plan) or institutional regulations, guide them to ask specifically for those services so CampusAI can route to the appropriate academic tool or official documents.
5. Do not make claims about access to information that was not provided."""


def get_general_chat_prompt() -> ChatPromptTemplate:
    """Chat prompt template for GENERAL inquiries."""
    return ChatPromptTemplate.from_messages([
        ("system", GENERAL_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def general_node(state: AgentState) -> AgentState:
    """General conversation and assistance capability node.

    Handles basic conversational questions, greetings, and general inquiries
    that do not require institutional documents (RAG) or student academic tools.
    Uses the project's existing Groq LLM infrastructure.
    """
    question = state.get("question", "")

    if not question or not str(question).strip():
        state["answer"] = "Hello! I am CampusAI, your academic assistant. How can I help you today?"
        return state

    try:
        llm = get_groq_llm()
        prompt = get_general_chat_prompt()
        chain = prompt | llm
        response = chain.invoke({"question": question})
        state["answer"] = response.content if hasattr(response, "content") else str(response)
    except (GroqError, LangChainException, ValueError):
        state["answer"] = FALLBACK_GENERAL

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
