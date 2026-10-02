"""Page-aware chunking: chunks never cross page boundaries, so page citations stay exact.

Structure-aware: FAQ-style "Q: ... A: ..." entries are split first, so each question becomes its own
focused chunk instead of being merged with unrelated questions (merging dilutes the embedding)."""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],   # paragraph > line > sentence > word
)
_BLOCK_START = re.compile(r"\n(?=Q\d*\s*[:.)]\s)")   # a new "Q:" / "Q1." entry begins
_MIN_CHARS = 25                                     # drop stray titles / fragments


def chunk_pages(pages: list[dict], base_meta: dict) -> list[dict]:
    chunks = []
    for p in pages:
        j = 0
        for block in _BLOCK_START.split(p["text"]):
            block = block.strip()
            if len(block) < _MIN_CHARS:
                continue
            for text in _splitter.split_text(block):
                meta = {**base_meta, "page": p["page"], "chunk": j}
                chunks.append({"id": f'{base_meta["document"]}::p{p["page"]}::c{j}',
                               "text": text, "metadata": meta})
                j += 1
    return chunks
