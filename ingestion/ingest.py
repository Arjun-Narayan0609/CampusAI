"""Build the vector store.   python -m ingestion.ingest [--rebuild]"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DOCS_DIR
from ingestion.loader import load_pdf
from ingestion.metadata import metadata_from_filename
from ingestion.chunker import chunk_pages


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    chunks = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        pages = load_pdf(pdf)
        c = chunk_pages(pages, metadata_from_filename(pdf))
        print(f"  {pdf.name}: {len(pages)} pages -> {len(c)} chunks")
        chunks += c
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="wipe and recreate the collection")
    args = ap.parse_args()

    from rag.vectorstore import get_collection, reset_collection
    from rag.embeddings import embed

    chunks = build_chunks()
    if not chunks:
        sys.exit(f"No PDFs found in {DOCS_DIR}")
    col = reset_collection() if args.rebuild else get_collection()
    B = 64
    for i in range(0, len(chunks), B):
        b = chunks[i:i + B]
        col.upsert(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                   metadatas=[c["metadata"] for c in b], embeddings=embed([c["text"] for c in b]))
    print(f"Done. {col.count()} chunks stored.")


if __name__ == "__main__":
    main()
