"""Page-aware chunking: chunks never cross page boundaries, so page citations stay exact."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],   # paragraph > line > sentence > word
)


def chunk_pages(pages: list[dict], base_meta: dict) -> list[dict]:
    chunks = []
    for p in pages:
        for j, text in enumerate(_splitter.split_text(p["text"])):
            meta = {**base_meta, "page": p["page"], "chunk": j}
            chunks.append({
                "id": f'{base_meta["document"]}::p{p["page"]}::c{j}',
                "text": text,
                "metadata": meta,
            })
    return chunks
