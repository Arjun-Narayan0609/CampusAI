"""Local sentence-transformer embeddings (loaded once)."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    # normalised vectors -> cosine similarity == dot product
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
