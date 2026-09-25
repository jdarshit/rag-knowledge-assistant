from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
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