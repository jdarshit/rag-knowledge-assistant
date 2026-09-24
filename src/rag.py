from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.vector_store import get_vector_store

LLM_MODEL = "llama3.2:3b"
TOP_K = 4
MAX_DISTANCE = 0.70  # chunks farther than this are treated as not relevant
NOT_FOUND = "I could not find this in the documents."

PROMPT = ChatPromptTemplate.from_template(
    """You are a document assistant. Answer ONLY using the context below.
If the answer is not in the context, reply exactly: "I could not find this in the documents."
Answer in the same language as the question. Be concise.

Context:
{context}

Question: {question}

Answer:"""
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0)


def retrieve(question: str, k: int = TOP_K):
    """Return [(Document, cosine_distance), ...]. Lower distance means more relevant."""
    return get_vector_store().similarity_search_with_score(question, k=k)


def format_context(results) -> str:
    """Join the retrieved chunks into one text block, each labelled with its source and page."""
    parts = []
    for doc, _ in results:
        meta = doc.metadata
        parts.append(f"[{meta['source']}, page {meta['page']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> dict:
    """Retrieve chunks, drop the irrelevant ones, and ask the LLM only if something is left."""
    results = retrieve(question, k)
    if not results:
        return {"answer": "No documents found. Please ingest a PDF first.", "sources": []}

    closest = float(results[0][1])
    relevant = [(doc, distance) for doc, distance in results if distance <= max_distance]
    if not relevant:
        return {"answer": NOT_FOUND, "sources": [], "closest_distance": closest}

    chain = PROMPT | get_llm()
    response = chain.invoke({"context": format_context(relevant), "question": question})

    sources = [
        {
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "distance": float(distance),
        }
        for doc, distance in relevant
    ]
    return {"answer": response.content, "sources": sources, "closest_distance": closest}