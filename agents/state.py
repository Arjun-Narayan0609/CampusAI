"""
Agent State Module
Owned by: Member 2

This module defines the shared AgentState passed between CampusAI LangGraph nodes.
"""

from typing import Any, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state representation for the CampusAI LangGraph workflow.

    Fields are populated progressively across graph execution:
    - student_id: Identifier of the active student.
    - question: User input question or instruction.
    - intent: Classified intent from question analysis.
    - context: Retrieved institutional context chunks.
    - tool_result: Result payload from academic/scheduler tools.
    - answer: Generated or validated response text.
    - sources: Source attribution and reference citations.
    """

    student_id: str
    question: str
    intent: Optional[str]
    context: Optional[str]
    tool_result: Optional[Any]
    answer: Optional[str]
    sources: Optional[List[Any]]
