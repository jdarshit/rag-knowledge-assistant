# RAG Knowledge Assistant

A local, privacy-first Retrieval-Augmented Generation (RAG) system that answers questions from your own PDF documents, with page-level source citations. Built to run entirely on local hardware — no paid APIs, no data leaving the machine.

![Status](https://img.shields.io/badge/status-active-brightgreen)

## Features

- **Multi-PDF ingestion** — upload and manage multiple documents, with automatic deduplication
- **Conversational RAG** — follow-up questions ("What about the large model?") are understood using chat history
- **Query rewriting** — short or ambiguous questions are rewritten into clear, standalone questions before retrieval
- **Hybrid retrieval pipeline** — dense vector search (Chroma) followed by cross-encoder reranking for higher precision
- **Grounded answers** — a strict prompt and a similarity threshold make the system say "I could not find this in the documents" instead of hallucinating
- **Evaluation suite** — a small labelled test set with measured retrieval accuracy and false-positive rate (see [Evaluation](#evaluation))
- **Query logging & feedback** — every question is logged with latency and sources; users can rate answers 👍/👎
- **Two interfaces** — a Streamlit chat UI and a FastAPI REST API, both backed by the same core pipeline

## Architecture
PDF → loader → chunker → embeddings (MiniLM) → Chroma (vector DB)

question + chat history
→ query condensing (LLM)
→ dense retrieval (top-8)
→ cross-encoder reranking (top-4)
→ similarity threshold filter
→ grounded prompt → Ollama (Llama 3.2 3B)
→ answer + page citations


## Tech stack

Python · LangChain · Ollama (Llama 3.2 3B) · ChromaDB · HuggingFace sentence-transformers · Streamlit · FastAPI

## Setup

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com) installed with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`).

```bash
git clone https://github.com/jdarshit/rag-knowledge-assistant.git
cd rag-knowledge-assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
cp .env.example .env             # adjust settings if needed
```

Add PDFs to `data/sample_pdfs/`, then:

```bash
python main.py ingest
```

## Usage

**Streamlit UI:**
```bash
python -m streamlit run app.py
```

**Command line:**
```bash
python main.py ask
python main.py list
python main.py delete <filename>
```

**REST API:**
```bash
uvicorn api:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

## Evaluation

An 8-question labelled test set (`evaluation/test_set.json`) measures two things:

| Metric | Result |
|---|---|
| Retrieval accuracy (expected source retrieved) | 100% |
| Correct rejection of out-of-scope questions | 100% |
| Average response time | ~56s (CPU-only, 3B model) |

Run it yourself: `python run_evaluation.py`. Full per-question results are saved to `evaluation/results.json`.

*Note: this is a small evaluation set (2 documents, 8 questions) intended to demonstrate the evaluation methodology, not a claim of production-scale accuracy.*

## Design decisions

- **Local-first (Ollama + Chroma + local embeddings):** keeps data private and avoids API costs/rate limits, at the cost of slower inference on CPU-only hardware.
- **Two-stage retrieval (dense search + reranking):** embeddings alone retrieve approximate candidates fast; a cross-encoder reranker then scores each candidate more precisely against the query, improving relevance.
- **Similarity threshold + strict prompt:** distance-based filtering catches clearly irrelevant questions; a tightened prompt additionally rejects chunks that merely mention a topic without answering it, reducing hallucination on partially-relevant context.
- **Query condensing for conversation:** follow-up questions are rewritten into standalone questions using recent chat history before retrieval, so pronouns and short references still retrieve correctly.

## Known limitations

- CPU-only inference on a small (3B) model is slow (30–120s per answer) and occasionally still produces overconfident answers on ambiguous queries.
- Reranker and evaluation are run on a small document set; results may not generalize to larger, more diverse corpora.
- No authentication; not intended for multi-user deployment as-is.

## Project structure
├── app.py # Streamlit UI
├── api.py # FastAPI REST API
├── main.py # CLI (ingest / ask / list / delete)
├── run_evaluation.py # Evaluation runner
├── src/
│ ├── loader.py # PDF text extraction
│ ├── chunker.py # Text splitting
│ ├── vector_store.py # Chroma wrapper
│ ├── ingest.py # Ingestion pipeline
│ ├── query_rewriter.py # Query rewriting + conversation condensing
│ ├── reranker.py # Cross-encoder reranking
│ ├── rag.py # Core RAG pipeline
│ ├── evaluator.py # Faithfulness/retrieval checks
│ ├── logger.py # Query logging + feedback
│ └── config.py # Environment-based configuration
├── evaluation/ # Test set + results
└── data/sample_pdfs/ # Source PDFs