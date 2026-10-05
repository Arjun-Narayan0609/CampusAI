"""Central RAG settings. Change here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "documents"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION = "campus_docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, 384-dim
CHUNK_SIZE = 400        # characters
CHUNK_OVERLAP = 80
TOP_K = 4
# Cosine similarity of the BEST chunk must reach this, else "no evidence".
# Tuned on our eval set: weakest real query ~0.44, closest no-evidence query ~0.32. Re-tune if the corpus changes.
MIN_SCORE = 0.37
ALL = "ALL"            # metadata value for documents that apply to everyone
