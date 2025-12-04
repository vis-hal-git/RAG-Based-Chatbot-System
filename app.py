# app.py (FINAL)
import os
import tempfile
import shutil
import time
from dotenv import load_dotenv
import streamlit as st

# local modules (these must exist in the project root)
from ingestion import ingest
from chunker import chunk_document
from vectorstore_utils import build_faiss_from_chunks, save_faiss, load_faiss
from llm_query import query_openai_chat

# optional helpers (if present in your project)
try:
    from hybrid_retriever import hybrid_retrieve
except Exception:
    hybrid_retrieve = None

try:
    from reranker import cross_modal_rerank
except Exception:
    cross_modal_rerank = None

try:
    from summarizer import summarize_short, summarize_brief
except Exception:
    summarize_short = None
    summarize_brief = None

load_dotenv()

# Basic check for OpenAI key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("OPENAI_API_KEY is not set. Set it in your environment or in a .env file and restart.")
    st.stop()

st.set_page_config(page_title="RAG Based Chatbot System", layout="wide")
st.title("Vision-RAG Chatbot — Document Q&A (silent ingestion)")

# ---------------- NEW: Tabs for Chat and Evaluation Dashboard ----------------
tab1, tab2 = st.tabs(["Chat", "Evaluation Dashboard"])

with tab2:
    st.header("Retrieval Evaluation Dashboard")
    if "eval_logs" not in st.session_state or not st.session_state.eval_logs:
        st.info("No evaluation logs yet — start asking queries to populate metrics.")
    else:
        import pandas as pd
        df = pd.DataFrame(st.session_state.eval_logs)
        st.dataframe(df)
        st.markdown("### Latency Summary")
        st.write(df["latency"].describe())
        st.markdown("### Number of Queries")
        st.write(len(df))
# -------------------------------------------------------------------------------

# initialize session state
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []       # list of text chunks indexed in FAISS
if "image_docs" not in st.session_state:
    st.session_state.image_docs = []        # list of image metas: {'image_path', 'page', ...}
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of {"role":"user"/"assistant","content":...}
if "k" not in st.session_state:
    st.session_state.k = 4
if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    # Hardcode k to 4 and remove from UI
    st.session_state.k = 4
    st.markdown("---")
    if st.button("Reset index & chat"):
        st.session_state.faiss_store = None
        st.session_state.text_chunks = []
        st.session_state.image_docs = []
        st.session_state.indexed = False
        st.session_state.chat_history = []
        st.session_state.eval_logs = []
        st.success("Reset complete")

# Helper: process upload and index text (images stored separately)
def process_and_index(uploaded_file):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    try:
        # save upload
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        # ingest returns a list of docs: {'content','meta'}
        docs = ingest(tmp_path)

        text_chunks = []
        image_docs = []

        for d in docs:
            meta = d.get("meta", {})
            # If this doc is an extracted image (meta.type == "image"), save metadata separately
            if meta.get("type") == "image" and meta.get("image_path"):
                image_docs.append(meta.copy())
                continue

            # else treat as text (including OCRed text from images)
            content = d.get("content", "") or ""
            if content.strip():
                # chunk text into LLM-friendly pieces
                chunks = chunk_document(content, meta)
                # ensure snippet + consistent meta kept
                for c in chunks:
                    if "snippet" not in c["meta"]:
                        c["meta"]["snippet"] = c["content"][:300]
                # standardize format to match hybrid_retriever expectations
                for c in chunks:
                    text_chunks.append({"content": c["content"], "meta": c["meta"], "meta_raw": c["meta"]})
        # build FAISS only from text_chunks
        if text_chunks:
            faiss_store = build_faiss_from_chunks([{"content": t["content"], "meta": t["meta"]} for t in text_chunks])
            st.session_state.faiss_store = faiss_store
            st.session_state.text_chunks = text_chunks
        else:
            st.session_state.faiss_store = None
            st.session_state.text_chunks = []

        # store image docs (may be empty)
        st.session_state.image_docs = image_docs
        st.session_state.indexed = True

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

# File uploader (silent ingestion)
uploaded = st.file_uploader("Upload a PDF / image / text file", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded is not None and not st.session_state.indexed:
    with st.spinner("Processing document..."):
        process_and_index(uploaded)
    st.success("Document indexed — you can now chat with the document.", icon="💬")

# If not indexed, instruct user
if not st.session_state.indexed:
    with tab1:
        st.info("Upload a document to start the chat. The app will process it automatically (silent indexing).")
    st.stop()

# Chat UI inside tab1
with tab1:
    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.write("Status")
        st.success("Indexed ✓")
        st.write(f"Text chunks: {len(st.session_state.text_chunks)}")
        st.write(f"Extracted images: {len(st.session_state.image_docs)}")
        st.write(f"Retrieved passages (k): {st.session_state.k}")
        st.markdown("---")
        st.write("Tip: k is fixed to 4 to control cost.")

    with chat_col:
        st.markdown("### Chat")
        # display history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        # input
        user_input = st.chat_input("Ask a question about the uploaded document...")
        if user_input:
            # show user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            start_time = time.time()

            # ---------- Retrieval ----------
            passages = []
            images_for_llm = []
            pages_hit = set()

            # Use hybrid_retrieve if available; otherwise use FAISS similarity_search_with_score
            if hybrid_retrieve is not None:
                try:
                    candidates = hybrid_retrieve(user_input, st.session_state.text_chunks, k= max(12, st.session_state.k * 3))
                    # candidates are list of {'content','meta', 'score'}
                    for c in candidates:
                        passages.append({"page_content": c["content"], "metadata": c["meta"]})
                        meta = c.get("meta") or {}
                        if meta.get("page") is not None:
                            pages_hit.add(meta.get("page"))
                        else:
                            pages_hit.add(None)
                except Exception:
                    # fallback to FAISS
                    hybrid_fallback = True
                    hybrid_retriever = None

            if (not passages) and st.session_state.faiss_store:
                try:
                    results = st.session_state.faiss_store.similarity_search_with_score(user_input, k=st.session_state.k)
                except Exception:
                    results = st.session_state.faiss_store.similarity_search_with_score(user_input, k=st.session_state.k)
                for doc, score in results:
                    passages.append({"page_content": doc.page_content, "metadata": doc.metadata})
                    meta = doc.metadata or {}
                    if meta.get("page") is not None:
                        pages_hit.add(meta.get("page"))
                    else:
                        pages_hit.add(None)

            # Collect images that are on retrieved pages (if any)
            for img_meta in st.session_state.image_docs:
                try:
                    img_page = img_meta.get("page")
                    if (img_page in pages_hit) or (None in pages_hit):
                        images_for_llm.append({"image_path": img_meta.get("image_path"), "meta": img_meta})
                except Exception:
                    continue

            # ---------- Cross-modal reranking ----------
            final_passages = passages
            final_images = images_for_llm
            if cross_modal_rerank is not None:
                try:
                    # cross_modal_rerank expects text_items and image_items
                    reranked = cross_modal_rerank(user_input, passages, images_for_llm, text_weight=0.75, image_weight=0.25, top_k=st.session_state.k)
                    # partition reranked results into text passages and image docs
                    final_passages = [r for r in reranked if r.get("page_content")]
                    final_images = [r for r in reranked if r.get("image_path") or (r.get("meta") and r["meta"].get("image_path"))]
                except Exception:
                    final_passages = passages
                    final_images = images_for_llm

            # ---------- Summarization shortcut ----------
            lowered = user_input.strip().lower()
            if lowered.startswith("summar") or lowered in ["summary", "summary in short", "brief"]:
                if summarize_short is not None:
                    with st.spinner("Generating summary..."):
                        try:
                            summary = summarize_short(final_passages[:st.session_state.k])
                        except Exception:
                            # fallback: ask LLM directly with a short prompt
                            combined = "\n\n".join([p["page_content"] for p in final_passages[:st.session_state.k]])
                            fallback_prompt = f"Summarize the following content in 2-3 short lines:\n\n{combined}"
                            summary = query_openai_chat(final_passages[:st.session_state.k], final_images[:st.session_state.k], fallback_prompt)
                else:
                    combined = "\n\n".join([p["page_content"] for p in final_passages[:st.session_state.k]])
                    fallback_prompt = f"Summarize the following content in 2-3 short lines:\n\n{combined}"
                    summary = query_openai_chat(final_passages[:st.session_state.k], final_images[:st.session_state.k], fallback_prompt)

                st.session_state.chat_history.append({"role":"assistant","content": summary})
                st.chat_message("assistant").write(summary)

                latency = time.time() - start_time
                st.session_state.eval_logs.append({
                    "query": user_input,
                    "latency": latency,
                    "k": st.session_state.k,
                    "n_text_chunks": len(st.session_state.text_chunks),
                    "n_images": len(st.session_state.image_docs),
                    "action": "summarize"
                })
            else:
                # ---------- LLM Query (vision-aware) ----------
                with st.spinner("Thinking (vision + text)..."):
                    answer = query_openai_chat(final_passages[:st.session_state.k], final_images[:st.session_state.k], user_input)

                latency = time.time() - start_time

            # show assistant answer
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

            # log retrieval metrics for dashboard
            st.session_state.eval_logs.append({
                "query": user_input,
                "latency": latency,
                "k": st.session_state.k,
                "n_text_chunks": len(st.session_state.text_chunks),
                "n_images": len(st.session_state.image_docs),
                "action": "qa"
            })

