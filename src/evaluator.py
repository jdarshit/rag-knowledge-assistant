from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.rag import NOT_FOUND

LLM_MODEL = "llama3.2:3b"

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are checking whether an answer is faithful to the given context.
An answer is faithful if every claim in it is supported by the context.
Reply with exactly one word: "yes" or "no".

Context:
{context}

Answer:
{answer}

Is the answer faithful to the context? (yes/no):"""
)


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0)


def check_faithfulness(answer: str, context: str) -> bool:
    """Ask the LLM whether the answer's claims are supported by the context."""
    if not context.strip():
        return False
    chain = FAITHFULNESS_PROMPT | get_judge_llm()
    response = chain.invoke({"context": context, "answer": answer})
    return "yes" in response.content.strip().lower()


def check_retrieval(sources: list[dict], expected_source: str | None) -> bool:
    """True if the expected source appears among the retrieved sources.
    If expected_source is None, this check is not applicable (return None instead)."""
    if expected_source is None:
        return None
    return any(s["source"] == expected_source for s in sources)