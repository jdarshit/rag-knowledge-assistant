import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# One sentence -> one vector (a list of numbers)
vector = embeddings.embed_query("What is machine learning?")
print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")

sentences = [
    "What is machine learning?",
    "Explain how ML models learn from data.",
    "What is the capital of France?",
]
vectors = embeddings.embed_documents(sentences)


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"Sentence 0 vs 1: {cosine(vectors[0], vectors[1]):.3f}")
print(f"Sentence 0 vs 2: {cosine(vectors[0], vectors[2]):.3f}")