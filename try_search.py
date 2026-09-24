from src.vector_store import get_vector_store

store = get_vector_store()
print(f"Chunks in store: {len(store.get()['ids'])}")

question = "What is the Transformer architecture?"
results = store.similarity_search_with_score(question, k=4)

for rank, (doc, distance) in enumerate(results, start=1):
    print(f"\n#{rank} | {doc.metadata['source']} | page {doc.metadata['page']} | distance {distance:.3f}")
    print(doc.page_content[:200])