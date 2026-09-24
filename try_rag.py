import time

from src.rag import ask

questions = [
    "What is multi-head attention?",
    "Who won the FIFA World Cup in 2018?",
]

for question in questions:
    start = time.time()
    result = ask(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (page {s['page']}) distance={s['distance']:.3f}")
    print(f"Took {time.time() - start:.1f} seconds")