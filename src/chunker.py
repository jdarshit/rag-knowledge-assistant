from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split each page into smaller chunks and keep the metadata with every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for page in pages:
        for index, text in enumerate(splitter.split_text(page["text"])):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": index,
                    },
                )
            )
    return chunks