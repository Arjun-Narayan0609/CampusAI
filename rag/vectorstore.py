"""Persistent ChromaDB collection (survives Streamlit reruns)."""
import chromadb
from config import CHROMA_DIR, COLLECTION

_client = None


def _get_client():
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    return _get_client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def reset_collection():
    client = _get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return get_collection()
