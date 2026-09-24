from src.loader import load_pdf
from src.chunker import chunk_pages

pages = load_pdf("data/sample_pdfs/sample.pdf")
chunks = chunk_pages(pages)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")

lengths = [len(chunk.page_content) for chunk in chunks]
print(f"Shortest chunk: {min(lengths)} characters")
print(f"Longest chunk: {max(lengths)} characters")
print(f"Average chunk: {sum(lengths) // len(lengths)} characters")

print("\n--- Chunk 0 ---")
print(chunks[0].metadata)
print(chunks[0].page_content)

print("\n--- Chunk 1 ---")
print(chunks[1].metadata)
print(chunks[1].page_content)