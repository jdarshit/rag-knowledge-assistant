from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.query_rewriter import rewrite_query
from src.reranker import rerank
from src.vector_store import get_vector_store

LLM_MODEL = "llama3.2:3b"
TOP_K = 8  # retrieve more, then rerank narrows it down
RERANK_TOP_N = 4  # how many chunks survive reranking
MAX_DISTANCE = 0.70  # chunks farther than this are dropped before reranking
MIN_RERANK_SCORE = 0.0  # chunks scoring below this after reranking are dropped
NOT_FOUND = "I could not find this in the documents."

PROMPT = ChatPromptTemplate.from_template(
    """You are a document assistant. Answer ONLY using the context below.

Rules:
- If the context only mentions the topic in passing, but does not actually explain
  or define what the question asks, reply exactly: "I could not find this in the documents."
- Do not guess, infer, or use outside knowledge.
- If the answer is not clearly stated in the context, reply exactly:
  "I could not find this in the documents."
- Answer in the same language as the question. Be concise.

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
    """Join chunks into one text block, each labelled with its source and page. Works with
    either (doc, distance) pairs or (doc, distance, rerank_score) triples."""
    parts = []
    for item in results:
        doc = item[0]
        meta = doc.metadata
        parts.append(f"[{meta['source']}, page {meta['page']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(
    question: str,
    k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
    top_n: int = RERANK_TOP_N,
) -> dict:
    """Rewrite the question, retrieve candidates, rerank them, and ask the LLM."""
    search_query = rewrite_query(question)
    candidates = retrieve(search_query, k)

    if not candidates:
        return {
            "answer": "No documents found. Please ingest a PDF first.",
            "sources": [],
            "search_query": search_query,
        }

    closest = float(candidates[0][1])
    filtered = [(doc, distance) for doc, distance in candidates if distance <= max_distance]
    if not filtered:
        return {
            "answer": NOT_FOUND,
            "sources": [],
            "closest_distance": closest,
            "search_query": search_query,
        }

    reranked = rerank(search_query, filtered, top_n=top_n)
    relevant = [(doc, distance, score) for doc, distance, score in reranked if score >= MIN_RERANK_SCORE]

    if not relevant:
        return {
            "answer": NOT_FOUND,
            "sources": [],
            "closest_distance": closest,
            "search_query": search_query,
        }

    chain = PROMPT | get_llm()
    response = chain.invoke({"context": format_context(relevant), "question": question})

    sources = [
        {
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "distance": float(distance),
            "rerank_score": float(score),
        }
        for doc, distance, score in relevant
    ]
    return {
        "answer": response.content,
        "sources": sources,
        "closest_distance": closest,
        "search_query": search_query,
    }