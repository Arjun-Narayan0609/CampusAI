"""
CampusAI - Streamlit Application Entrypoint
Owned by: Member 4

Responsibilities:
- Streamlit web application user interface
- User interactions and input management
- Integration with agent orchestration layer (Member 2's run_agent interface)
"""

import streamlit as st

from agents.graph import run_agent
from student.registry import (
    RegistryError,
    StudentNotFoundError,
    get_student_profile,
    list_student_ids,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CampusAI",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎓 CampusAI")
st.caption("Your Agentic Academic Assistant — powered by LangGraph")

st.divider()

# ---------------------------------------------------------------------------
# Student selection
# ---------------------------------------------------------------------------

# Load student IDs from registry once per session
if "student_list" not in st.session_state:
    try:
        ids = list_student_ids()
        # Build display labels: "ID — Name (Dept, Sem N)"
        labels = []
        id_map = {}
        for sid in ids:
            try:
                p = get_student_profile(sid)
                label = f"{sid} — {p['name']} ({p['department']}, Sem {p['semester']})"
            except (StudentNotFoundError, RegistryError):
                label = sid
            labels.append(label)
            id_map[label] = sid
        st.session_state["student_list"] = labels
        st.session_state["student_id_map"] = id_map
    except RegistryError as e:
        st.error(f"Could not load student registry: {e}")
        st.stop()

student_labels = st.session_state["student_list"]
id_map = st.session_state["student_id_map"]

selected_label = st.selectbox(
    "Select Student",
    options=student_labels,
    index=0,
    help="Choose a student profile from the campus registry.",
)

student_id = id_map[selected_label]

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------

question = st.text_area(
    "Ask CampusAI a question",
    placeholder=(
        "e.g.  What is my CGPA?  ·  Create a study plan  ·  "
        "Show my timetable  ·  What are the attendance rules?"
    ),
    height=100,
)

ask_clicked = st.button("Ask CampusAI", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if ask_clicked:
    # --- Input validation (UI-side) ---
    if not question or not question.strip():
        st.warning("Please enter a question before clicking Ask.")
        st.stop()

    # --- Call the public backend interface ---
    with st.spinner("CampusAI is thinking…"):
        try:
            result = run_agent(student_id, question.strip())
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
            st.stop()
        except (RegistryError, StudentNotFoundError) as exc:
            st.error(f"Student registry error: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"An error occurred while processing your request: {exc}")
            st.stop()

    # --- Extract result fields ---
    answer = result.get("answer", "")
    intent = result.get("intent", "")
    sources = result.get("sources") or []
    tool_result = result.get("tool_result")

    # --- Display answer ---
    st.subheader("Answer")
    st.markdown(answer)

    # --- Display detected intent ---
    intent_labels = {
        "ACADEMIC_QUERY": "📚 Academic Query (RAG)",
        "CGPA": "🎯 CGPA / SGPA",
        "SCHEDULE": "📅 Schedule",
        "STUDY_PLAN": "📝 Study Plan",
        "GENERAL": "💬 General",
    }
    intent_display = intent_labels.get(intent, intent)
    st.caption(f"Intent: {intent_display}")

    st.divider()

    # -----------------------------------------------------------------
    # Intent-specific structured output
    # -----------------------------------------------------------------

    if intent == "CGPA" and isinstance(tool_result, dict):
        if tool_result.get("ok") is not False:
            st.subheader("CGPA Details")
            cols = st.columns(3)
            cols[0].metric("CGPA", tool_result.get("cgpa", "—"))
            cols[1].metric("Semester", tool_result.get("semester", "—"))
            cols[2].metric("Department", tool_result.get("department", "—"))

            # SGPA detail if present
            if "sgpa" in tool_result:
                st.metric("SGPA", tool_result["sgpa"])
            if "formula" in tool_result:
                st.caption(tool_result["formula"])

    elif intent == "STUDY_PLAN" and isinstance(tool_result, dict):
        if tool_result.get("ok") is not False:
            plan_sessions = tool_result.get("plan", [])
            if plan_sessions:
                st.subheader("Study Plan")
                # Display as a table
                table_rows = []
                for s in plan_sessions:
                    table_rows.append({
                        "Date": s.get("date", ""),
                        "Subject": s.get("subject", ""),
                        "Start": s.get("start_time", ""),
                        "End": s.get("end_time", ""),
                        "Duration (min)": s.get("duration_min", ""),
                    })
                st.table(table_rows)

            # Per-subject summary
            per_subject = tool_result.get("per_subject", [])
            if per_subject:
                st.subheader("Per-Subject Summary")
                for ps in per_subject:
                    subj = ps.get("subject", "Unknown")
                    sess = ps.get("sessions", 0)
                    mins = ps.get("total_minutes", 0)
                    st.write(f"**{subj}**: {sess} session(s), {mins} min total")

            # Validation issues
            issues = tool_result.get("issues", [])
            if issues:
                st.warning("Validation issues: " + "; ".join(issues))

            # Warnings
            warnings = tool_result.get("warnings", [])
            if warnings:
                for w in warnings:
                    st.info(w)

    elif intent == "SCHEDULE" and isinstance(tool_result, dict):
        if tool_result.get("ok") is not False:
            sessions = tool_result.get("sessions", [])
            if sessions:
                st.subheader("Schedule")
                table_rows = []
                for s in sessions:
                    table_rows.append({
                        "Date": s.get("date", ""),
                        "Subject": s.get("subject", ""),
                        "Start": s.get("start_time", ""),
                        "End": s.get("end_time", ""),
                        "Status": s.get("status", ""),
                    })
                st.table(table_rows)

            # Moved / failed info for modify operations
            moved = tool_result.get("moved", [])
            if moved:
                st.success(f"{len(moved)} session(s) rescheduled successfully.")
            failed = tool_result.get("failed", [])
            if failed:
                st.warning(f"{len(failed)} session(s) could not be rescheduled.")

    elif intent == "ACADEMIC_QUERY":
        # Answer already displayed above; sources below
        pass

    elif intent == "GENERAL":
        # Answer already displayed above
        pass

    # -----------------------------------------------------------------
    # RAG Sources
    # -----------------------------------------------------------------

    if sources:
        st.subheader("Sources")
        for src in sources:
            if isinstance(src, dict):
                name = src.get("source") or src.get("name") or str(src)
                st.write(f"- {name}")
            elif isinstance(src, str):
                st.write(f"- {src}")
            else:
                st.write(f"- {src}")

    # -----------------------------------------------------------------
    # Error detail for failed tool results (debug aid)
    # -----------------------------------------------------------------

    if isinstance(tool_result, dict) and tool_result.get("ok") is False:
        with st.expander("Error Details"):
            st.json(tool_result)
