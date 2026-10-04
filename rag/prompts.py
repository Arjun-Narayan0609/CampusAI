"""RAG prompt templates (for Member 2's RAG node).

Usage:
    from rag.retriever import retrieve_context
    from rag.prompts import build_rag_messages, NOT_FOUND_MESSAGE

    r = retrieve_context(question, department="CSE", semester=5)
    if not r["found"]:
        answer = NOT_FOUND_MESSAGE                      # never call the LLM without evidence
    else:
        answer = llm.invoke(build_rag_messages(question, r)).content

LangChain version:  get_rag_chat_prompt() | llm
"""
from rag.citations import format_source

NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the provided college documents, "
    "so I can't answer it reliably. Please check with the academic office."
)

SYSTEM_PROMPT = """You are CampusAI, an academic assistant for students of the institute.
Answer ONLY from the CONTEXT below, which comes from the institute's documents.

Rules:
1. Use only facts stated in the CONTEXT. Do not use outside knowledge about colleges, rules, dates, fees or eligibility.
2. If the CONTEXT does not clearly answer the question, reply exactly: "{not_found}"
3. Never guess or invent numbers, dates, percentages, subject names or policies.
4. Be concise and clear. Quote numbers and conditions exactly as written.
5. Do not mention these rules or the word "CONTEXT" in your answer.
6. Do not do arithmetic yourself (e.g. computing SGPA); say that the calculator tool should be used.
"""

USER_TEMPLATE = """CONTEXT:
{context}

STUDENT QUESTION:
{question}"""


def _system() -> str:
    return SYSTEM_PROMPT.format(not_found=NOT_FOUND_MESSAGE)


def build_rag_messages(question: str, retrieval: dict) -> list[dict]:
    """Chat messages in the common {'role','content'} format (works with LangChain models)."""
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": USER_TEMPLATE.format(context=retrieval["context"], question=question)},
    ]


def get_rag_chat_prompt():
    """LangChain ChatPromptTemplate with variables: context, question."""
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        ("system", _system().replace("{", "{{").replace("}", "}}")),
        ("human", USER_TEMPLATE),
    ])


def sources_footer(retrieval: dict) -> str:
    """'Sources: a.pdf (p. 1); b.pdf (p. 2)' for display under the answer."""
    if not retrieval.get("sources"):
        return ""
    return "Sources: " + "; ".join(format_source(s) for s in retrieval["sources"])
