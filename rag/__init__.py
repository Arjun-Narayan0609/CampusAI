"""CampusAI RAG Package.

Public interface for Member 2 agent integration:
- retrieve_context: Primary retrieval function returning structured context & citations
- normalize_department: Deterministic normalization of department names (e.g., 'Computer Science' -> 'CSE')
- DEPARTMENT_MAPPING: Mapping of department aliases to canonical abbreviations
"""

from rag.retriever import DEPARTMENT_MAPPING, normalize_department, retrieve_context

__all__ = [
    "retrieve_context",
    "normalize_department",
    "DEPARTMENT_MAPPING",
]
