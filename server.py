# server.py
import os
import tempfile
import shutil
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# local modules
from ingestion import ingest
from chunker import chunk_document
from vectorstore_utils import build_faiss_from_chunks
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

# Global State (since it's a single-user system typically, mimicking st.session_state)
class AppState:
    def __init__(self):
        self.faiss_store = None
        self.text_chunks = []
        self.image_docs = []
        self.indexed = False

state = AppState()

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
async def upload_file(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
             shutil.copyfileobj(file.file, f)
        
        # ingest returns a list of docs {'content', 'meta'}
        docs = ingest(tmp_path)
        text_chunks = []
        image_docs = []
        
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
                    
        if text_chunks:
            faiss_store = build_faiss_from_chunks([{"content": t["content"], "meta": t["meta"]} for t in text_chunks])
            state.faiss_store = faiss_store
            state.text_chunks = text_chunks
        else:
            state.faiss_store = None
            state.text_chunks = []
            
        state.image_docs = image_docs
        state.indexed = True
        
        return JSONResponse(content={"chunks": len(state.text_chunks), "images": len(state.image_docs)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not state.indexed:
        raise HTTPException(status_code=400, detail="No document is indexed. Please upload a document first.")
        
    user_input = request.query
    k = 8
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Get history
    chat_history = get_thread_history(thread_id)
    chat_history.append({"role": "user", "content": user_input})
    
    passages = []
    images_for_llm = []
    pages_hit = set()
    
    if hybrid_retrieve is not None:
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
            img_page = img_meta.get("page")
            if (img_page in pages_hit) or (None in pages_hit):
                images_for_llm.append({"image_path": img_meta.get("image_path"), "meta": img_meta})
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
    
    # We provide a clean source list for references
    source_names = []
    for p in final_passages[:k]:
        page = p['metadata'].get('page')
        if page:
             source_names.append(f"Page {page}")
        else:
             source_names.append("Text Segment")
    
    # De-duplicate
    source_names = list(set(source_names))
    
    return JSONResponse(content={"answer": answer, "sources": source_names, "thread_id": thread_id})

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
    return JSONResponse(content={"messages": history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
