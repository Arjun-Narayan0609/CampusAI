"""
Question Analyzer Module
Owned by: Member 2

Responsibilities:
- Phase 1 intent classification of student inquiries
- Strict structured output validation into one of five operational intents:
  ACADEMIC_QUERY, STUDY_PLAN, CGPA, SCHEDULE, GENERAL
- LangGraph node implementation: analyze_question(state) -> state
"""

from typing import Any, Literal, Optional, get_args
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from agents.state import AgentState
from llm.groq import get_groq_llm

IntentType = Literal[
    "ACADEMIC_QUERY",
    "STUDY_PLAN",
    "CGPA",
    "SCHEDULE",
    "GENERAL",
]

VALID_INTENTS = set(get_args(IntentType))


class QuestionAnalysis(BaseModel):
    """Structured intent classification for a student inquiry."""

    intent: IntentType = Field(
        ...,
        description=(
            "The single operational intent category representing the student's primary request. "
            "Must be one of: ACADEMIC_QUERY, STUDY_PLAN, CGPA, SCHEDULE, GENERAL."
        ),
    )


SYSTEM_PROMPT = """You are the Question Analyzer for CampusAI, an agentic academic assistant.
Your task is to classify the student's question into exactly ONE of the following five operational intents based on their primary request:

1. ACADEMIC_QUERY:
   - Questions seeking information about academic concepts, course/subject material, institutional policies, or student academic records intended for the campus knowledge base (RAG).
   - Academic concepts & subject-matter questions: questions asking for explanations, definitions, or conceptual understanding of topics and theory taught in academic college courses and subjects (e.g., DBMS, Software Engineering, Operating Systems, Computer Networks, Data Structures, Algorithms, Theory of Computation, Mathematics).
   - Coursework, syllabus & study material references: questions explicitly or implicitly referring to academic subjects, student notes, syllabus, curriculum, textbooks, lectures, course units, or coursework.
   - Institutional & administrative academic information: college regulations, attendance rules/requirements, official curriculum/syllabus, exam schedules/procedures, department courses, semester subjects, internship eligibility, placement criteria, scholarship rules, campus policies, and student academic FAQs.
   - Student-specific academic queries: student academic inquiries (e.g., 'What are my fifth-semester subjects?', 'What are the attendance requirements?') unless they explicitly request one of the specialized operations below.
   - CRITICAL: If the student asks about an institutional policy, rule, or requirement—even if it mentions CGPA, exams, or courses (e.g., 'What is the CGPA requirement for an internship?', 'When are the semester exams?')—the intent is ACADEMIC_QUERY, NOT CGPA or STUDY_PLAN.
   - Examples of ACADEMIC_QUERY:
     * 'What is normalization in DBMS?'
     * 'Explain normalization in DBMS from my notes.'
     * 'What is the three schema architecture in DBMS?'
     * 'Explain functional and non-functional requirements in Software Engineering.'
     * 'What is a primary key?'
     * 'Explain the difference between process and thread.'
     * 'What are the attendance requirements?'
     * 'What are my fifth-semester subjects?'
     * 'Explain operating system scheduling.'

2. STUDY_PLAN:
   - Requests specifically asking to generate, structure, or modify a student's personal study plan, revision timetable, study schedule, or exam preparation strategy.
   - Examples: 'Make me a study plan for my exams.', 'Create a timetable for my preparation.', 'Help me plan my study sessions for the next two weeks.', 'What subjects should I study for tomorrow\\'s exam?', 'Create a study plan for me.'.

3. CGPA:
   - Requests specifically asking for personal CGPA/SGPA calculations, grade point conversions, or checking personal academic GPA performance.
   - Examples: 'What is my CGPA?', 'Calculate my SGPA.', 'What will my CGPA be if I get these grades?', 'Calculate my CGPA from these grades.'.

4. SCHEDULE:
   - Requests specifically involving viewing, checking, or modifying personal timetable, class schedule, availability, or calendar study sessions.
   - Examples: 'What classes do I have tomorrow?', 'Show my timetable.', 'Show my schedule.', 'I cannot attend my Wednesday class.', 'Move my study session to Friday.', 'I cannot study Wednesday.'.

5. GENERAL:
   - Casual conversation, greetings, chit-chat, assistant capabilities, or general-purpose questions outside the academic curriculum and institutional scope of CampusAI.
   - Includes general knowledge questions, general technology or tool inquiries not tied to college academic subjects/curriculum (e.g., 'What is Docker?'), how-to guides for everyday tasks, trivia, and jokes.
   - Examples of GENERAL:
     * 'Hello.'
     * 'Hi there.'
     * 'How are you?'
     * 'What can you do?'
     * 'Who are you?'
     * 'Thanks.'
     * 'Thank you!'
     * 'What is Docker?'
     * 'Tell me a joke.'
     * 'Who invented the telephone?'
     * 'How do I cook pasta?'
     * 'Explain how to schedule a Python script.'

Classification Principles:
- Focus on the PRIMARY SEMANTIC INTENT and context of the whole question, not superficial keyword matches.
- Academic subject matter (theory, concepts, models taught in college courses like DBMS, OS, SE, etc.) and institutional rules MUST be classified as ACADEMIC_QUERY so they route to the academic knowledge layer / RAG.
- General-purpose questions outside academic coursework (such as general devops/tool queries like 'What is Docker?', jokes, general trivia, everyday questions) MUST be classified as GENERAL.
- Specialized intents (STUDY_PLAN, CGPA, SCHEDULE) take precedence when the request is explicitly asking for those specific operations (calculating personal GPA, creating personal study plans, or viewing personal class schedules).
- Choose exactly ONE intent from the five allowed values: ACADEMIC_QUERY, STUDY_PLAN, CGPA, SCHEDULE, GENERAL.
Respond ONLY with a JSON object containing the field 'intent'."""


def get_analyzer_prompt() -> ChatPromptTemplate:
    """Create the ChatPromptTemplate for question analysis."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )


def get_structured_llm(llm: Optional[Any] = None) -> Any:
    """
    Configure structured output on the provided or default Groq LLM.

    Args:
        llm: Optional chat model instance. If None, retrieves default ChatGroq.

    Returns:
        Structured LLM runnable enforcing QuestionAnalysis schema.
    """
    if llm is None:
        llm = get_groq_llm()

    if hasattr(llm, "with_structured_output"):
        try:
            return llm.with_structured_output(QuestionAnalysis, method="json_mode")
        except (TypeError, ValueError):
            return llm.with_structured_output(QuestionAnalysis)

    return llm


def classify_question_intent(question: str, llm: Optional[Any] = None) -> str:
    """
    Classify a student's question into exactly one Phase 1 intent.

    Args:
        question: Natural-language inquiry from the student.
        llm: Optional LLM instance (defaults to configured ChatGroq).

    Returns:
        str: Exactly one of ACADEMIC_QUERY, STUDY_PLAN, CGPA, SCHEDULE, GENERAL.

    Raises:
        ValueError: If question is empty or intent validation fails.
        RuntimeError: If LLM invocation or structured output parsing fails.
    """
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    cleaned_question = question.strip()

    prompt = get_analyzer_prompt()
    structured_llm = get_structured_llm(llm=llm)

    try:
        messages = prompt.format_messages(question=cleaned_question)
        if hasattr(structured_llm, "invoke"):
            result = structured_llm.invoke(messages)
        else:
            result = structured_llm(messages)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Question analysis failed during Groq API request: {exc}") from exc

    if isinstance(result, QuestionAnalysis):
        raw_intent = result.intent
    elif isinstance(result, dict) and "intent" in result:
        raw_intent = result["intent"]
    else:
        raise RuntimeError(
            f"Question Analyzer received invalid structured output from model: {result!r}"
        )

    if not raw_intent or not isinstance(raw_intent, str):
        raise RuntimeError(
            f"Question Analyzer received empty or non-string intent: {raw_intent!r}"
        )

    intent = raw_intent.strip()
    if intent not in VALID_INTENTS:
        raise ValueError(
            f"Unsupported intent '{intent}' returned. Must be one of {sorted(VALID_INTENTS)}."
        )

    return intent


def analyze_question(state: AgentState, llm: Optional[Any] = None) -> AgentState:
    """
    LangGraph node for question analysis and intent classification.

    Reads state['question'], classifies the primary request into exactly one
    operational intent, stores it in state['intent'], and returns the updated state.

    Args:
        state: Shared AgentState mapping containing 'question'.
        llm: Optional LLM instance for testing or dependency injection.

    Returns:
        AgentState: Updated state with 'intent' populated.

    Raises:
        ValueError: If state['question'] is missing/invalid or intent is unsupported.
        RuntimeError: If model classification or structured output fails.
    """
    if not isinstance(state, dict):
        raise ValueError("AgentState must be a dictionary-like object.")

    question = state.get("question")
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("AgentState must contain a non-empty string in 'question' for analysis.")

    intent = classify_question_intent(question, llm=llm)
    state["intent"] = intent
    return state
