from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's input as a single clear, well-formed question.
Keep the same meaning and language. Do not answer the question.
If it is already a clear question, return it unchanged.
Return ONLY the rewritten question, nothing else.

User input: {query}

Rewritten question:"""
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the follow-up
question as a single standalone question that makes sense without the history.
Replace pronouns (it, this, that, they) with what they refer to.
If the follow-up question is already standalone, return it unchanged.
Keep the same language. Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:"""
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


def format_history(history: list[dict], max_turns: int = 3) -> str:
    """Format the last few (user, assistant) turns as plain text for the condense prompt."""
    recent = history[-(max_turns * 2):]
    lines = []
    for message in recent:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def condense_question(query: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone one using recent chat history.
    Falls back to plain rewriting if there is no history, and to the original query on error."""
    query = query.strip()
    if not query:
        return query
    if not history:
        return rewrite_query(query)
    try:
        chain = CONDENSE_PROMPT | get_rewriter_llm()
        response = chain.invoke({"history": format_history(history), "query": query})
        rewritten = response.content.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        return query