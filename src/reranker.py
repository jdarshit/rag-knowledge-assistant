from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, candidates: list[tuple], top_n: int = 4) -> list[tuple]:
    """Score each (doc, distance) pair against the query and return the top_n as
    (doc, distance, rerank_score) triples, best first."""
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = get_reranker().predict(pairs)

    scored = [
        (doc, distance, float(score))
        for (doc, distance), score in zip(candidates, scores)
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_n]