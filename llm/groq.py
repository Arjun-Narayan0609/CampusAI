"""
Groq LLM Configuration Module
Owned by: Member 2

Responsibilities:
- Configure and initialize Groq client via langchain-groq
- Manage model hyperparameters and environment variable loading
"""

import os
from typing import Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_groq_api_key() -> str:
    """
    Retrieve GROQ_API_KEY from environment, loading from .env if present.

    Raises:
        ValueError: If GROQ_API_KEY is not set or empty.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please configure GROQ_API_KEY in your .env file or environment."
        )
    return api_key.strip()


def get_groq_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGroq:
    """
    Create and configure a ChatGroq instance.

    Args:
        model_name: Name of the Groq model (defaults to GROQ_MODEL env var or DEFAULT_GROQ_MODEL).
        temperature: Sampling temperature (defaults to 0.0 for deterministic classification/reasoning).
        api_key: Optional API key override (defaults to loading GROQ_API_KEY from environment).
        **kwargs: Additional parameters passed to ChatGroq.

    Returns:
        ChatGroq: Configured LangChain ChatGroq model instance.
    """
    resolved_key = api_key or get_groq_api_key()
    resolved_model = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(
        groq_api_key=resolved_key,
        model=resolved_model,
        temperature=temperature,
        **kwargs,
    )
