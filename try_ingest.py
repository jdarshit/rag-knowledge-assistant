import time

from src.chunker import chunk_pages
from src.loader import load_pdf
from src.vector_store import add_chunks, get_vector_store

pages = load_pdf("data/sample_pdfs/sample.pdf")
chunks = chunk_pages(pages)
print(f"Chunks to add: {len(chunks)}")

start = time.time()
add_chunks(chunks)
print(f"Added in {time.time() - start:.1f} seconds")

stored = get_vector_store().get()
print(f"Chunks stored in Chroma: {len(stored['ids'])}")