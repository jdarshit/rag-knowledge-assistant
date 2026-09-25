import html
import time
from pathlib import Path

import streamlit as st

from src.ingest import ingest_pdf
from src.logger import log_feedback, log_query
from src.rag import LLM_MODEL, MAX_DISTANCE, TOP_K, ask
from src.vector_store import delete_document, list_documents

DATA_DIR = Path("data/sample_pdfs")
USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

#MainMenu, footer {visibility: hidden;}

html, body, [class*="css"] {font-family: 'Share Tech Mono', monospace;}

.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(0,245,255,0.08), transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(255,0,200,0.08), transparent 40%),
                #0A0E17;
}

.block-container {max-width: 900px; padding-top: 2.2rem;}

.app-title {
    font-size: 2.1rem; font-weight: 700; letter-spacing: 0.04em; margin: 0;
    color: #00F5FF;
    text-shadow: 0 0 8px rgba(0,245,255,0.7), 0 0 18px rgba(0,245,255,0.35);
}
.app-subtitle {color: #7DD8E8; margin: 0.35rem 0 1.5rem 0; letter-spacing: 0.02em;}

[data-testid="stMetric"] {
    background: #0D1524;
    border: 1px solid #00F5FF55;
    border-radius: 10px; padding: 0.8rem 1rem;
    box-shadow: 0 0 12px rgba(0,245,255,0.15), inset 0 0 20px rgba(0,245,255,0.03);
}
[data-testid="stMetricLabel"] {color: #7DD8E8 !important;}
[data-testid="stMetricValue"] {color: #00F5FF !important; text-shadow: 0 0 6px rgba(0,245,255,0.5);}

.source-card {
    border: 1px solid #FF00C855;
    background: #12081A;
    border-radius: 8px;
    padding: 0.55rem 0.8rem; margin-bottom: 0.45rem; font-size: 0.88rem;
    box-shadow: 0 0 10px rgba(255,0,200,0.12);
    color: #F0D9FF;
}
.chip {
    background: #1A0B2E; color: #FF6BE0; border: 1px solid #FF00C866;
    border-radius: 999px;
    padding: 2px 10px; font-size: 0.72rem; margin-left: 0.5rem;
}

section[data-testid="stSidebar"] {
    background: #0B0F1C;
    border-right: 1px solid #00F5FF33;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #0D1524;
    color: #00F5FF;
    border: 1px solid #00F5FF66;
    box-shadow: 0 0 8px rgba(0,245,255,0.15);
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #00F5FF;
    box-shadow: 0 0 14px rgba(0,245,255,0.4);
    color: #FFFFFF;
}

.stChatInput textarea, [data-testid="stChatInput"] {
    border: 1px solid #00F5FF55 !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.1);
}

div[data-testid="stChatMessage"] {
    background: #0D1220;
    border: 1px solid #1F2A44;
    border-radius: 10px;
}

hr, [data-testid="stDivider"] {border-color: #00F5FF33 !important;}
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def render_details(message: dict, index: int) -> None:
    """Show sources (or not-found note), response time, and feedback buttons."""
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                score = s.get("rerank_score")
                score_chip = f'<span class="chip">score {score:.2f}</span>' if score is not None else ""
                st.markdown(
                    f'<div class="source-card"><b>{html.escape(s["source"])}</b>'
                    f' · page {s["page"]}'
                    f'<span class="chip">distance {s["distance"]:.3f}</span>{score_chip}</div>',
                    unsafe_allow_html=True,
                )
    elif message.get("closest") is not None:
        st.caption(
            f"No chunk was close enough (closest distance {message['closest']:.3f}). "
            "Try raising the max distance in the sidebar."
        )
    if message.get("seconds") is not None:
        st.caption(f"Answered in {message['seconds']:.0f}s")

    log_id = message.get("log_id")
    if log_id:
        if message.get("feedback"):
            st.caption(f"Feedback recorded: {message['feedback']}")
        else:
            col_up, col_down, _ = st.columns([1, 1, 8])
            if col_up.button("👍", key=f"up_{index}"):
                log_feedback(log_id, "up")
                message["feedback"] = "up"
                st.rerun()
            if col_down.button("👎", key=f"down_{index}"):
                log_feedback(log_id, "down")
                message["feedback"] = "down"
                st.rerun()


documents = list_documents()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📚 KNOWLEDGE_BASE")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded and st.button("＋ ADD TO KNOWLEDGE BASE", type="primary"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        added = []
        for file in uploaded:
            path = DATA_DIR / file.name
            path.write_bytes(file.getbuffer())
            with st.spinner(f"Processing {file.name}..."):
                count = ingest_pdf(path)
            added.append(f"{file.name} ({count} chunks)")
        st.session_state.flash = "Added: " + ", ".join(added)
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.markdown("### DOCUMENTS")
    if not documents:
        st.caption("No documents yet.")
    for name, count in documents.items():
        col_name, col_delete = st.columns([5, 2])
        col_name.markdown(f"**{name}**  \n{count} chunks")
        if col_delete.button("✕", key=f"del_{name}", help=f"Remove {name}"):
            delete_document(name)
            st.session_state.flash = f"Removed {name}"
            st.rerun()

    st.divider()
    st.markdown("### RETRIEVAL_SETTINGS")
    top_k = st.slider("Top-K chunks", 1, 8, TOP_K)
    max_distance = st.slider("Max distance", 0.30, 1.00, MAX_DISTANCE, 0.01)
    st.caption("Lower max distance = stricter relevance filter.")

    st.divider()
    if st.button("⟲ CLEAR CHAT"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main area ----------------
st.markdown('<p class="app-title">⌁ RAG KNOWLEDGE ASSISTANT</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask questions about your documents. '
    "Answers run locally and come with page-level sources.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("DOCUMENTS", len(documents))
col2.metric("CHUNKS", sum(documents.values()))
col3.metric("MODEL", LLM_MODEL)

st.write("")

if not documents:
    st.info("Upload a PDF from the sidebar to get started.")

for i, message in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_details(message, i)

question = st.chat_input("Ask a question about your documents", disabled=not documents)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        start = time.time()
        with st.spinner("Searching documents and generating an answer..."):
            try:
                history = st.session_state.messages[:-1]
                result = ask(question, history=history, k=top_k, max_distance=max_distance)
                seconds = time.time() - start
                log_id = log_query(question, result, seconds)
                answer = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "closest": result.get("closest_distance"),
                    "seconds": seconds,
                    "log_id": log_id,
                    "feedback": None,
                }
            except Exception as error:
                answer = {
                    "role": "assistant",
                    "content": f"Something went wrong: {error}. Is the Ollama app running?",
                }
        st.write(answer["content"])
        render_details(answer, len(st.session_state.messages))
    st.session_state.messages.append(answer)