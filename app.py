# app.py (FINAL)
import os
import tempfile
import shutil
import time
import uuid
from dotenv import load_dotenv
import streamlit as st

# local modules (these must exist in the project root)
from ingestion import ingest
from chunker import chunk_document
from vectorstore_utils import build_faiss_from_chunks, add_chunks_to_faiss, save_faiss, load_faiss
from llm_query import query_openai_chat
from db_utils import get_all_threads, get_thread_history, save_chat_thread

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
st.title("RAG Based Chatbot System")



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
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "k" not in st.session_state:
    st.session_state.k = 4
if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.session_state.k = 4
    st.markdown("---")
    if st.button("Reset index & chat"):
        st.session_state.faiss_store = None
        st.session_state.text_chunks = []
        st.session_state.image_docs = []
        st.session_state.indexed = False
        st.session_state.chat_history = []
        st.session_state.eval_logs = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.processed_files = set()
        st.success("Reset complete")

    # History UI
    st.markdown("---")
    st.header("Chat History")
    threads = get_all_threads()
    if threads:
        for th in threads:
            lbl = f"💬 {th.get('preview', 'New Chat')}"
            if st.button(lbl, key=f"btn_{th['thread_id']}"):
                st.session_state.thread_id = th["thread_id"]
                st.session_state.chat_history = get_thread_history(th["thread_id"])
                st.rerun()
    else:
        st.write("No past history.")

# Helper: process upload and index text (images stored separately)
def process_and_index(uploaded_files):
    tmp_dir = tempfile.mkdtemp()
    existing_chunks = list(st.session_state.text_chunks or [])
    existing_images = list(st.session_state.image_docs or [])
    text_chunks = []
    image_docs = []
    try:
        for idx, uploaded_file in enumerate(uploaded_files):
            safe_name = os.path.basename(uploaded_file.name) if uploaded_file.name else f"upload_{idx}"
            tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{safe_name}")
            # save upload
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # ingest returns a list of docs: {'content','meta'}
            docs = ingest(tmp_path)

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
        combined_chunks = existing_chunks + text_chunks
        combined_images = existing_images + image_docs

        faiss_store = st.session_state.faiss_store
        if faiss_store is not None and text_chunks:
            add_chunks_to_faiss(faiss_store, [{"content": t["content"], "meta": t["meta"]} for t in text_chunks])
        elif faiss_store is None and combined_chunks:
            faiss_store = build_faiss_from_chunks([{"content": t["content"], "meta": t["meta"]} for t in combined_chunks])

        st.session_state.faiss_store = faiss_store
        st.session_state.text_chunks = combined_chunks
        st.session_state.image_docs = combined_images
        st.session_state.indexed = bool(st.session_state.faiss_store is not None or st.session_state.image_docs)

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

# File uploader (silent ingestion)
uploaded = st.file_uploader(
    "Upload PDFs, images, or text files",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    accept_multiple_files=True
)

new_uploads = []
if uploaded:
    for f in uploaded:
        size = getattr(f, "size", None)
        if size is None:
            size = len(f.getbuffer())
        key = f"{f.name}:{size}"
        if key not in st.session_state.processed_files:
            new_uploads.append(f)
            st.session_state.processed_files.add(key)

if new_uploads:
    was_indexed = st.session_state.indexed
    with st.spinner("Processing documents..."):
        process_and_index(new_uploads)
    if was_indexed:
        st.success("Documents added to the index.", icon="📎")
    else:
        st.success("Documents indexed — you can now chat with the documents.", icon="💬")

# If not indexed, instruct user
if not st.session_state.indexed:
    st.info("Upload documents to start the chat. The app will process them automatically (silent indexing).")
    st.stop()

# Chat UI
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
        if hybrid_retrieve is not None and st.session_state.text_chunks:
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
                img_path = img_meta.get("image_path")
                if not img_path:
                    continue
                img_page = img_meta.get("page")
                if (not pages_hit) or (img_page in pages_hit):
                    images_for_llm.append({"image_path": img_path, "meta": img_meta})
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

        # Initialize answer with a default value
        answer = "I'm sorry, I couldn't process your request."
        
        # ---------- Summarization shortcut ----------
        lowered = user_input.strip().lower()
        if lowered.startswith("summar") or lowered in ["summary", "summary in short", "brief"]:
            with st.spinner("Generating summary..."):
                try:
                    if summarize_short is not None:
                        answer = summarize_short(final_passages[:st.session_state.k])
                    else:
                        combined = "\n\n".join([p["page_content"] for p in final_passages[:st.session_state.k]])
                        fallback_prompt = f"Summarize the following content in 2-3 short lines:\n\n{combined}"
                        answer = query_openai_chat(final_passages[:st.session_state.k], final_images[:st.session_state.k], fallback_prompt, chat_history=st.session_state.chat_history[:-1], stream=True)
                    
                    # Log the summary action
                    latency = time.time() - start_time
                    st.session_state.eval_logs.append({
                        "query": user_input,
                        "latency": latency,
                        "k": st.session_state.k,
                        "n_text_chunks": len(st.session_state.text_chunks),
                        "n_images": len(st.session_state.image_docs),
                        "action": "summarize"
                    })
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    latency = 0
        else:
            # ---------- Regular LLM Query (vision-aware) ----------
            with st.spinner("Thinking (vision + text)..."):
                try:
                    answer = query_openai_chat(final_passages[:st.session_state.k], final_images[:st.session_state.k], user_input, chat_history=st.session_state.chat_history[:-1], stream=True)
                    latency = time.time() - start_time
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
                    latency = 0
                    answer = "I'm sorry, I encountered an error processing your request."

        # Show assistant answer
        with st.chat_message("assistant"):
            if isinstance(answer, str):
                st.write(answer)
                final_answer = answer
            else:
                final_answer = st.write_stream(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
        
        # Save to MongoDB
        save_chat_thread(st.session_state.thread_id, st.session_state.chat_history)

        # log retrieval metrics for dashboard
        st.session_state.eval_logs.append({
            "query": user_input,
            "latency": latency,
            "k": st.session_state.k,
            "n_text_chunks": len(st.session_state.text_chunks),
            "n_images": len(st.session_state.image_docs),
            "action": "qa"
        })
