import time

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)

start = time.time()
response = llm.invoke("In one sentence, what is a vector database?")
print(response.content)
print(f"\nTook {time.time() - start:.1f} seconds")
