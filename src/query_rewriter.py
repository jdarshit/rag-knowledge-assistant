from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

LLM_MODEL = "llama3.2:3b"

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's input as a single clear, well-formed question.
Keep the same meaning and language. Do not answer the question.
If it is already a clear question, return it unchanged.
Return ONLY the rewritten question, nothing else.

User input: {query}

Rewritten question:"""
)


@lru_cache(maxsize=1)
def get_rewriter_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0)


def rewrite_query(query: str) -> str:
    """Turn a short or messy query into a clear question. Falls back to the original on any error."""
    query = query.strip()
    if not query:
        return query
    try:
        chain = REWRITE_PROMPT | get_rewriter_llm()
        response = chain.invoke({"query": query})
        rewritten = response.content.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        return query