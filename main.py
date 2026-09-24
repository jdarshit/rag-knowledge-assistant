import argparse

from src.ingest import ingest_folder
from src.rag import ask
from src.vector_store import delete_document, list_documents

DATA_DIR = "data/sample_pdfs"


def run_ingest() -> None:
    results = ingest_folder(DATA_DIR)
    if not results:
        print(f"No PDFs found in {DATA_DIR}")
    for name, count in results.items():
        print(f"{name}: {count} chunks")


def run_list() -> None:
    documents = list_documents()
    if not documents:
        print("No documents stored yet. Run: python main.py ingest")
    for name, count in documents.items():
        print(f"{name}: {count} chunks")


def run_delete(name: str) -> None:
    deleted = delete_document(name)
    if deleted == 0:
        print(f"No document named '{name}' found. Use 'list' to see stored files.")
    else:
        print(f"Deleted {deleted} chunks of {name}")


def run_chat() -> None:
    print("Ask a question about your documents (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = ask(question)
        print(f"\nAnswer: {result['answer']}")

        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f"  - {s['source']} (page {s['page']}) distance={s['distance']:.3f}")
        else:
            closest = result.get("closest_distance")
            if closest is not None:
                print(f"(No relevant chunks. Closest distance: {closest:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant")
    parser.add_argument("command", choices=["ingest", "ask", "list", "delete"])
    parser.add_argument("name", nargs="?", help="file name, used by the delete command")
    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest()
    elif args.command == "list":
        run_list()
    elif args.command == "delete":
        if not args.name:
            parser.error("delete needs a file name, e.g. python main.py delete bert.pdf")
        run_delete(args.name)
    else:
        run_chat()


if __name__ == "__main__":
    main()