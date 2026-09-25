from src.query_rewriter import rewrite_query

queries = [
    "what ai",
    "what is ml",
    "attention heads how many",
    "What is multi-head attention?",
]

for q in queries:
    print(f"{q!r} -> {rewrite_query(q)!r}")