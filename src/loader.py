from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    """Return the text and metadata (file name, page number) of every page in a PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    pages = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue  # skip empty or scanned pages
        pages.append({
            "text": text,
            "source": path.name,
            "page": page_index + 1,
        })
    return pages