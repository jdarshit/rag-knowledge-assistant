import hashlib
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the Chroma store once and reuse it (loading the embedding model is slow)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def make_chunk_id(chunk: Document) -> str:
    """Same file + page + chunk + text always gives the same id, so re-adding never duplicates."""
    meta = chunk.metadata
    raw = f"{meta['source']}|{meta['page']}|{meta['chunk_index']}|{chunk.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def add_chunks(chunks: list[Document]) -> None:
    """Embed the chunks and save them in Chroma."""
    if not chunks:
        return
    ids = [make_chunk_id(chunk) for chunk in chunks]
    get_vector_store().add_documents(chunks, ids=ids)



def list_documents() -> dict[str, int]:
    """Return {file_name: number_of_chunks} for every document in the store."""
    data = get_vector_store().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def delete_document(source: str) -> int:
    """Delete all chunks of one file. Return how many chunks were deleted."""
    store = get_vector_store()
    ids = store.get(where={"source": source})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)    