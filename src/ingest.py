from pathlib import Path

from src.chunker import chunk_pages
from src.loader import load_pdf
from src.vector_store import add_chunks


def ingest_pdf(pdf_path: str | Path) -> int:
    """Load one PDF, split it into chunks and store them. Return the number of chunks."""
    pages = load_pdf(str(pdf_path))
    chunks = chunk_pages(pages)
    add_chunks(chunks)
    return len(chunks)


def ingest_folder(folder: str | Path) -> dict[str, int]:
    """Ingest every PDF in a folder. Return {file_name: chunk_count}."""
    results = {}
    for pdf in sorted(Path(folder).glob("*.pdf")):
        results[pdf.name] = ingest_pdf(pdf)
    return results