# vectorstore_utils.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document   # ✅ Correct import
from dotenv import load_dotenv
load_dotenv()


EMBED_MODEL = "text-embedding-3-small"

def make_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)

def build_faiss_from_chunks(chunks):
    embedder = make_embeddings()
    docs = [Document(page_content=c["content"], metadata=c["meta"]) for c in chunks]
    faiss_store = FAISS.from_documents(docs, embedder)
    return faiss_store

def save_faiss(faiss_store, path_prefix="faiss_index"):
    faiss_store.save_local(path_prefix)

def load_faiss(path_prefix="faiss_index"):
    embedder = make_embeddings()
    return FAISS.load_local(path_prefix, embedder)
