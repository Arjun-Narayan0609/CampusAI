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
   - Institutional or academic information questions intended for the campus knowledge base (RAG).
   - Covers college regulations, attendance rules, official curriculum/syllabus, exam dates/procedures, internship eligibility, placement criteria, scholarship rules, campus policies, and student FAQs.
   - CRITICAL: If the student asks about an institutional policy, rule, or requirement—even if it mentions CGPA, exams, or courses (e.g., "What is the CGPA requirement for an internship?", "When are the semester exams?")—the intent is ACADEMIC_QUERY, NOT CGPA or STUDY_PLAN.

2. STUDY_PLAN:
   - Requests to generate, structure, or modify a student's personal study plan, revision timetable, study schedule, or exam preparation strategy.
   - Examples: "Make me a study plan for my exams.", "Create a timetable for my preparation.", "Help me plan my study sessions for the next two weeks.", "What subjects should I study for tomorrow's exam?".

3. CGPA:
   - Requests specifically asking for personal CGPA/SGPA calculations, grade point conversions, or checking academic GPA performance.
   - Examples: "What is my CGPA?", "Calculate my SGPA.", "What will my CGPA be if I get these grades?", "Calculate my CGPA from these grades.".

4. SCHEDULE:
   - Requests involving viewing, checking, or modifying personal timetable, class schedule, availability, or calendar study sessions.
   - Examples: "What classes do I have tomorrow?", "Show my timetable.", "I cannot attend my Wednesday class.", "Move my study session to Friday.", "I cannot study Wednesday.".

5. GENERAL:
   - General conversational greetings, chit-chat, questions about assistant capabilities, or polite remarks not requiring institutional or academic tool capabilities.
   - Examples: "Hello.", "Hi there.", "How are you?", "What can you do?", "Who are you?", "Thanks.", "Thank you!".

Classification Principles:
- Focus on the PRIMARY INTENT, not superficial keyword matches.
- Choose exactly ONE intent from the five allowed values.
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
