# server.py
import os
import tempfile
import shutil
import uuid
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

# local modules
from ingestion import ingest
from chunker import chunk_document
from vectorstore_utils import build_faiss_from_chunks, add_chunks_to_faiss, save_faiss, load_faiss
from llm_query import query_openai_chat
from db_utils import get_all_threads, get_thread_history, save_chat_thread

try:
    from hybrid_retriever import hybrid_retrieve
except Exception:
    hybrid_retrieve = None

try:
    from reranker import cross_modal_rerank
except Exception:
    cross_modal_rerank = None

load_dotenv()

app = FastAPI(title="RAG-Based Chatbot System API")

@app.on_event("startup")
async def _validate_required_env() -> None:
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set (required). If using Docker Compose, ensure you run it from the project folder and that a .env file exists next to docker-compose.yml."
        )

CONTEXT_ROOT = Path(os.getcwd()) / "thread_contexts"
CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/favicon.ico")
def favicon():
        svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
    <defs>
        <linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'>
            <stop offset='0%' stop-color='#6c63ff'/>
            <stop offset='100%' stop-color='#00d4aa'/>
        </linearGradient>
    </defs>
    <rect width='64' height='64' rx='16' fill='#0a0b0f'/>
    <path d='M18 23h28v6H18zM18 35h20v6H18z' fill='url(#g)'/>
    <circle cx='46' cy='38' r='7' fill='none' stroke='url(#g)' stroke-width='4'/>
</svg>"""
        return Response(content=svg, media_type="image/svg+xml")

# Global State (since it's a single-user system typically, mimicking st.session_state)
class AppState:
    def __init__(self):
        self.faiss_store = None
        self.text_chunks = []
        self.image_docs = []
        self.indexed = False
        self.current_thread_id = None
        self.chunk_count = 0
        self.image_count = 0
        self.uploaded_files = []

state = AppState()

def _thread_context_dir(thread_id: str) -> Path:
    safe_thread_id = "".join(ch for ch in thread_id if ch.isalnum() or ch in ("-", "_"))
    return CONTEXT_ROOT / safe_thread_id

def _context_manifest_path(thread_id: str) -> Path:
    return _thread_context_dir(thread_id) / "context.json"

def _load_context_manifest(thread_id: str) -> Optional[dict]:
    manifest_path = _context_manifest_path(thread_id)
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_thread_context(thread_id: str, faiss_store, image_docs: list, chunk_count: int, uploaded_files: list):
    ctx_dir = _thread_context_dir(thread_id)
    ctx_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir = ctx_dir / "faiss_index"

    if faiss_store is not None:
        save_faiss(faiss_store, path_prefix=str(faiss_dir))
    elif faiss_dir.exists():
        shutil.rmtree(faiss_dir)

    manifest = {
        "thread_id": thread_id,
        "chunk_count": int(chunk_count),
        "image_count": int(len(image_docs or [])),
        "image_docs": image_docs or [],
        "has_faiss": faiss_store is not None,
        "uploaded_files": uploaded_files or []
    }
    with open(_context_manifest_path(thread_id), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

def _restore_thread_context(thread_id: str) -> bool:
    manifest = _load_context_manifest(thread_id)
    if not manifest:
        state.faiss_store = None
        state.text_chunks = []
        state.image_docs = []
        state.chunk_count = 0
        state.image_count = 0
        state.uploaded_files = []
        state.indexed = False
        state.current_thread_id = None
        return False

    faiss_store = None
    if manifest.get("has_faiss"):
        faiss_dir = _thread_context_dir(thread_id) / "faiss_index"
        if faiss_dir.exists():
            try:
                faiss_store = load_faiss(path_prefix=str(faiss_dir))
            except Exception as e:
                print(f"Failed to load FAISS for thread {thread_id}: {e}")

    state.faiss_store = faiss_store
    state.text_chunks = []  # Do not persist original chunk payloads in DB.
    state.image_docs = manifest.get("image_docs", [])
    state.chunk_count = int(manifest.get("chunk_count", 0))
    state.image_count = int(manifest.get("image_count", len(state.image_docs)))
    state.uploaded_files = list(manifest.get("uploaded_files", []))
    state.indexed = bool(state.faiss_store is not None or state.image_docs)
    state.current_thread_id = thread_id if state.indexed else None
    return state.indexed

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class ThreadLoadRequest(BaseModel):
    thread_id: str

@app.get("/")
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/upload")
async def upload_file(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    thread_id: Optional[str] = Form(None)
):
    upload_files = []
    if files:
        upload_files.extend(files)
    if file:
        upload_files.append(file)
    if not upload_files:
        raise HTTPException(status_code=400, detail="No files provided.")

    active_thread_id = thread_id or str(uuid.uuid4())
    if thread_id and state.current_thread_id != thread_id:
        _restore_thread_context(thread_id)

    tmp_dir = tempfile.mkdtemp()
    existing_chunks = list(state.text_chunks or [])
    existing_images = list(state.image_docs or [])
    existing_files = list(state.uploaded_files or [])
    text_chunks = []
    image_docs = []
    new_files = []
    try:
        for idx, upload in enumerate(upload_files):
            safe_name = os.path.basename(upload.filename) if upload.filename else f"upload_{idx}"
            new_files.append(safe_name)
            tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{safe_name}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # ingest returns a list of docs {'content', 'meta'}
            docs = ingest(tmp_path)

            for d in docs:
                meta = d.get("meta", {})
                if meta.get("type") == "image" and meta.get("image_path"):
                    image_docs.append(meta.copy())
                    continue

                content = d.get("content", "") or ""
                if content.strip():
                    chunks = chunk_document(content, meta)
                    for c in chunks:
                        if "snippet" not in c["meta"]:
                            c["meta"]["snippet"] = c["content"][:300]
                    for c in chunks:
                        text_chunks.append({"content": c["content"], "meta": c["meta"], "meta_raw": c["meta"]})

        combined_chunks = existing_chunks + text_chunks
        combined_images = existing_images + image_docs
        combined_files = []
        seen_names = set()
        for name in existing_files + new_files:
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            combined_files.append(name)

        faiss_store = state.faiss_store
        if faiss_store is not None and text_chunks:
            add_chunks_to_faiss(faiss_store, [{"content": t["content"], "meta": t["meta"]} for t in text_chunks])
        elif faiss_store is None and combined_chunks:
            faiss_store = build_faiss_from_chunks([{"content": t["content"], "meta": t["meta"]} for t in combined_chunks])

        state.faiss_store = faiss_store
        state.text_chunks = combined_chunks
        state.image_docs = combined_images
        state.uploaded_files = combined_files
        state.indexed = bool(state.faiss_store is not None or state.image_docs)
        state.current_thread_id = active_thread_id
        state.chunk_count = len(combined_chunks)
        state.image_count = len(combined_images)
        _save_thread_context(active_thread_id, state.faiss_store, state.image_docs, state.chunk_count, state.uploaded_files)

        return JSONResponse(content={
            "chunks": state.chunk_count,
            "images": state.image_count,
            "thread_id": active_thread_id,
            "files": state.uploaded_files
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

@app.post("/api/chat")
async def chat(request: ChatRequest):
    thread_id = request.thread_id or state.current_thread_id or str(uuid.uuid4())
    if state.current_thread_id != thread_id:
        _restore_thread_context(thread_id)

    if not state.indexed:
        raise HTTPException(status_code=400, detail="No indexed document context found for this chat. Please upload a document first.")
        
    user_input = request.query
    k = 8
    
    # Get history
    chat_history = get_thread_history(thread_id)
    chat_history.append({"role": "user", "content": user_input})
    
    passages = []
    images_for_llm = []
    pages_hit = set()
    
    if hybrid_retrieve is not None and state.text_chunks:
        try:
            candidates = hybrid_retrieve(user_input, state.text_chunks, k=max(12, k * 3))
            for c in candidates:
                passages.append({"page_content": c["content"], "metadata": c["meta"]})
                meta = c.get("meta") or {}
                if meta.get("page") is not None:
                    pages_hit.add(meta.get("page"))
                else:
                    pages_hit.add(None)
        except Exception:
            pass
            
    if (not passages) and state.faiss_store:
        try:
            results = state.faiss_store.similarity_search_with_score(user_input, k=k)
            for doc, score in results:
                passages.append({"page_content": doc.page_content, "metadata": doc.metadata})
                meta = doc.metadata or {}
                if meta.get("page") is not None:
                    pages_hit.add(meta.get("page"))
                else:
                    pages_hit.add(None)
        except Exception as e:
           print(f"FAISS fallback search error: {e}")
           pass
           
    for img_meta in state.image_docs:
        try:
            img_path = img_meta.get("image_path")
            if not img_path:
                continue
            img_page = img_meta.get("page")
            if (not pages_hit) or (img_page in pages_hit):
                images_for_llm.append({"image_path": img_path, "meta": img_meta})
        except Exception:
            continue
            
    final_passages = passages
    final_images = images_for_llm
    if cross_modal_rerank is not None:
        try:
            reranked = cross_modal_rerank(user_input, passages, images_for_llm, text_weight=0.75, image_weight=0.25, top_k=k)
            final_passages = [r for r in reranked if r.get("page_content")]
            final_images = [r for r in reranked if r.get("image_path") or (r.get("meta") and r["meta"].get("image_path"))]
        except Exception:
            pass
            
    answer = "I'm sorry, I couldn't process your request."
    
    try:
        # We enforce Stream=False for standard REST API format
        answer_raw = query_openai_chat(final_passages[:k], final_images[:k], user_input, chat_history=chat_history[:-1], stream=False)
        if hasattr(answer_raw, '__iter__') and not isinstance(answer_raw, str):
            answer = "".join([chunk for chunk in answer_raw])
        else:
            answer = answer_raw
    except Exception as e:
        answer = f"Error processing your request: {str(e)}"
        
    chat_history.append({"role": "assistant", "content": answer})
    save_chat_thread(thread_id, chat_history)
    state.current_thread_id = thread_id
    
    return JSONResponse(content={"answer": answer, "thread_id": thread_id})

@app.post("/api/reset")
async def reset_state():
    global state
    state = AppState()
    return JSONResponse(content={"ok": True})

@app.get("/api/history")
async def get_history():
    threads = get_all_threads()
    formatted_threads = []
    for th in threads:
        formatted_threads.append({
            "id": th["thread_id"],
            "title": th.get("preview", "New Chat"),
            "active": False
        })
    return JSONResponse(content={"history": formatted_threads})

@app.get("/api/history/{thread_id}")
async def get_thread(thread_id: str):
    history = get_thread_history(thread_id)
    has_context = _restore_thread_context(thread_id)
    return JSONResponse(content={
        "messages": history,
        "has_context": has_context,
        "chunks": state.chunk_count,
        "images": state.image_count,
        "files": state.uploaded_files
    })

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
