"""Central RAG settings. Change here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "documents"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION = "campus_docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, 384-dim
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 120
TOP_K = 4
# Cosine similarity of the BEST chunk must reach this, else "no evidence".
# Tune with evaluation/rag_evaluation.py (typical MiniLM range: 0.25-0.40).
MIN_SCORE = 0.30
ALL = "ALL"            # metadata value for documents that apply to everyone
