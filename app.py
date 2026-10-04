"""CampusAI - Streamlit Application Entrypoint
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
# Design System & Styling (Dark-First, Clean Academic AI)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Theme CSS Variables */
    :root {
        --bg-main: #0c0e14;
        --card-bg: #141824;
        --card-subtle: #191e2e;
        --border-subtle: #242b3e;
        --border-hover: #37415c;
        --accent: #4f6bf5;
        --accent-hover: #3d56cf;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Streamlit Container constraints */
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }

    /* Brand / Header */
    .campus-header {
       margin-bottom: 1.5rem;
    padding-top: 0.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border-subtle);
    }
    .campus-brand {
         display: flex;
    align-items: center;
    gap: 0.6rem;
    }
    .campus-logo {
        font-size: 2rem;
        line-height: 1;
    }
    .campus-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .campus-subtitle {
         font-size: 0.92rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
    margin-left: 2.6rem;
    font-weight: 400;
    }

    /* Section heading labels */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.45rem;
    }

    /* Student metadata chips */
    .student-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.65rem;
    }
    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: var(--card-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 0.22rem 0.6rem;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }
    .meta-chip-val {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Response Header & Badges */
    .response-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .response-sender {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
    }
    .badge-base {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.01em;
    }
    .badge-academic {
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #93c5fd;
    }
    .badge-cgpa {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }
    .badge-schedule {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fcd34d;
    }
    .badge-study-plan {
        background-color: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #d8b4fe;
    }
    .badge-general {
        background-color: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #cbd5e1;
    }

    /* Content division */
    .card-divider {
        height: 1px;
        background-color: var(--border-subtle);
        margin: 1.25rem 0 1rem 0;
    }
    .sub-title {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.65rem;
    }

    /* Source Cards */
    .source-card-box {
        background-color: var(--card-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
    }
    .source-doc-name {
        font-family: monospace;
        font-size: 0.83rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .source-doc-meta {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Study Summary Items */
    .subject-summary-item {
        background-color: var(--card-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Primary CTA Button Polish */
    button[kind="primary"] {
        background-color: var(--accent) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        transition: background-color 0.15s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }

    /* Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border-subtle) !important;
        border-radius: 10px !important;
        background-color: var(--card-bg) !important;
    }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: var(--card-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    /* Input text area and selectbox styling */
    div[data-testid="stTextArea"] textarea {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Brand / Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="campus-header">
        <div class="campus-brand">
            <span class="campus-logo">🎓</span>
            <span class="campus-title">CampusAI</span>
        </div>
        <div class="campus-subtitle">Your Agentic Academic Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Student Selection (Live Registry)
# ---------------------------------------------------------------------------

if "student_list" not in st.session_state:
    try:
        ids = list_student_ids()
        labels = []
        id_map = {}
        profile_map = {}
        for sid in ids:
            try:
                p = get_student_profile(sid)
                dept = p.get("department", "")
                sem = p.get("semester", "")
                name = p.get("name", "")
                label = f"{sid} · {name} · {dept} · Semester {sem}"
                profile_map[sid] = p
            except (StudentNotFoundError, RegistryError):
                label = sid
                profile_map[sid] = None
            labels.append(label)
            id_map[label] = sid
        st.session_state["student_list"] = labels
        st.session_state["student_id_map"] = id_map
        st.session_state["student_profile_map"] = profile_map
    except RegistryError as e:
        st.error(f"Could not load student registry: {e}")
        st.stop()

student_labels = st.session_state["student_list"]
id_map = st.session_state["student_id_map"]
profile_map = st.session_state.get("student_profile_map", {})

with st.container(border=True):
    st.markdown('<div class="section-label">STUDENT</div>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "Select Student Profile",
        options=student_labels,
        index=0,
        help="Choose a student profile from the campus registry.",
        label_visibility="collapsed",
    )
    student_id = id_map[selected_label]
    active_profile = profile_map.get(student_id)
    if active_profile:
        st.markdown(
            f"""
            <div class="student-meta-row">
                <span class="meta-chip">ID <span class="meta-chip-val">{student_id}</span></span>
                <span class="meta-chip">Name <span class="meta-chip-val">{active_profile.get('name', '—')}</span></span>
                <span class="meta-chip">Dept <span class="meta-chip-val">{active_profile.get('department', '—')}</span></span>
                <span class="meta-chip">Semester <span class="meta-chip-val">{active_profile.get('semester', '—')}</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Reset cached result if student profile changed
if st.session_state.get("last_student_id") and st.session_state.get("last_student_id") != student_id:
    st.session_state.pop("last_result", None)

# ---------------------------------------------------------------------------
# Question Composer
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown('<div class="section-label">ASK CAMPUSAI</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Ask CampusAI a question",
        placeholder="Ask about your subjects, CGPA, schedule, or study plan...",
        height=100,
        label_visibility="collapsed",
    )
    ask_clicked = st.button("Ask CampusAI →", type="primary", use_container_width=True)

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

    st.session_state["last_result"] = result
    st.session_state["last_student_id"] = student_id

# ---------------------------------------------------------------------------
# Response Rendering
# ---------------------------------------------------------------------------

result = st.session_state.get("last_result")

if result:
    answer = result.get("answer", "")
    intent = result.get("intent", "")
    sources = result.get("sources") or []
    tool_result = result.get("tool_result")

    # Intent badge configuration
    intent_config = {
        "ACADEMIC_QUERY": ("📚 Academic Query · RAG", "badge-academic"),
        "CGPA": ("🎯 CGPA / SGPA", "badge-cgpa"),
        "SCHEDULE": ("🗓️ Schedule", "badge-schedule"),
        "STUDY_PLAN": ("📖 Study Plan", "badge-study-plan"),
        "GENERAL": ("💬 General", "badge-general"),
    }
    intent_display, badge_class = intent_config.get(intent, (intent or "General", "badge-general"))

    with st.container(border=True):
        # AI Response Header
        st.markdown(
            f"""
            <div class="response-top-bar">
                <div class="response-sender">
                    <span>🤖</span>
                    <span>CampusAI</span>
                </div>
                <span class="badge-base {badge_class}">{intent_display}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Answer Section
        st.markdown('<div class="sub-title">Answer</div>', unsafe_allow_html=True)
        st.markdown(answer)

        # -----------------------------------------------------------------
        # Intent-specific structured output
        # -----------------------------------------------------------------

        if intent == "CGPA" and isinstance(tool_result, dict):
            if tool_result.get("ok") is not False:
                st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sub-title">CGPA Details</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                cols[0].metric("CGPA", tool_result.get("cgpa", "—"))
                cols[1].metric("Semester", tool_result.get("semester", "—"))
                cols[2].metric("Department", tool_result.get("department", "—"))

                if "sgpa" in tool_result:
                    st.metric("SGPA", tool_result["sgpa"])
                if "formula" in tool_result:
                    st.caption(tool_result["formula"])

        elif intent == "STUDY_PLAN" and isinstance(tool_result, dict):
            if tool_result.get("ok") is not False:
                plan_sessions = tool_result.get("plan", [])
                if plan_sessions:
                    st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="sub-title">Study Plan Timetable</div>', unsafe_allow_html=True)
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

                per_subject = tool_result.get("per_subject", [])
                if per_subject:
                    st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="sub-title">Per-Subject Summary</div>', unsafe_allow_html=True)
                    for ps in per_subject:
                        subj = ps.get("subject", "Unknown")
                        sess = ps.get("sessions", 0)
                        mins = ps.get("total_minutes", 0)
                        st.markdown(
                            f"""
                            <div class="subject-summary-item">
                                <strong>{subj}</strong>
                                <span>{sess} session(s) · {mins} min total</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                issues = tool_result.get("issues", [])
                if issues:
                    st.warning("Validation issues: " + "; ".join(issues))

                warnings = tool_result.get("warnings", [])
                if warnings:
                    for w in warnings:
                        st.info(w)

        elif intent == "SCHEDULE" and isinstance(tool_result, dict):
            if tool_result.get("ok") is not False:
                sessions = tool_result.get("sessions", [])
                if sessions:
                    st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="sub-title">Schedule</div>', unsafe_allow_html=True)
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

                moved = tool_result.get("moved", [])
                if moved:
                    st.success(f"{len(moved)} session(s) rescheduled successfully.")
                failed = tool_result.get("failed", [])
                if failed:
                    st.warning(f"{len(failed)} session(s) could not be rescheduled.")

        # -----------------------------------------------------------------
        # Sources / Supporting Evidence
        # -----------------------------------------------------------------

        if sources:
            st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-title">Sources</div>', unsafe_allow_html=True)
            for src in sources:
                if isinstance(src, dict):
                    doc = src.get("document") or src.get("source") or src.get("name") or str(src)
                    page = src.get("page")
                    score = src.get("score")
                    meta_parts = []
                    if page is not None:
                        meta_parts.append(f"Page {page}")
                    if score is not None:
                        try:
                            meta_parts.append(f"Relevance {float(score):.2f}")
                        except (ValueError, TypeError):
                            meta_parts.append(f"Relevance {score}")
                    meta_str = " · ".join(meta_parts)
                    st.markdown(
                        f"""
                        <div class="source-card-box">
                            <div class="source-doc-name">{doc}</div>
                            {f'<div class="source-doc-meta">{meta_str}</div>' if meta_str else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="source-card-box">
                            <div class="source-doc-name">{str(src)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # -----------------------------------------------------------------
        # Error Details for Failed Tool Results
        # -----------------------------------------------------------------

        if isinstance(tool_result, dict) and tool_result.get("ok") is False:
            with st.expander("Technical Details"):
                st.json(tool_result)
