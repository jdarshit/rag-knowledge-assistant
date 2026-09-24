from src.loader import load_pdf

pages = load_pdf("data/sample_pdfs/sample.pdf")
print(f"Pages with text: {len(pages)}")

if pages:
    first = pages[0]
    print(first["source"], "| page", first["page"])
    print(first["text"][:300])
else:
    print("No text found (the PDF may be scanned).")