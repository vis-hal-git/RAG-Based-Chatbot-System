# 🧠 Multimodal RAG-Based Chatbot System

A state-of-the-art **Multimodal Retrieval-Augmented Generation (RAG)** application built with Python, Streamlit, and OpenAI. This system goes beyond traditional text-only assistants by allowing users to upload unstructured text files, standard images, and complex PDFs, seamlessly blending OCR, local Vision embeddings, and dense-vector search functionality into a cohesive conversational UI.

---

## 🌟 Key Capabilities

1. **Multimodal Ingestion Pipeline:** 
   - Dynamically parses uploaded `.pdf`, `.png`, `.jpg`, and `.txt` files.
   - Leverages `pdfplumber` to scrape precise text chunks and bounding boxes, while simultaneously utilizing `pytesseract` to extract OCR text directly from embedded images and visual graphics mapping them precisely to metadata.
2. **Text & Vision Embeddings:** 
   - Pure text retrieval uses `OpenAIEmbeddings` alongside a local **FAISS CPU index**.
   - Native visual elements rely on local Hugging Face `sentence-transformers` models (using `clip-ViT-B-32`) to properly convert imagery into searchable vector dimensions without expensive third-party external API loops.
3. **Hybrid Search System (RRF):**
   - Merges Sparse semantic tracking (using `rank_bm25` lexical algorithms) alongside foundational Dense embedding searches (using FAISS).
   - Melds these candidates utilizing **Reciprocal Rank Fusion (RRF)** to organically return the best combined text matches.
4. **Cross-Modal Reranking Phase:**
   - Both textual paragraphs and raw extracted image paths are scored specifically against the user's live query using `cosine_similarity`. 
   - Ensures visually relevant diagrams are handed directly to the OpenAI generation head.
5. **Conversational Memory & State Persistence:**
   - Real-time LLM interactions are maintained logically in memory natively mapping `chat_history`.
   - Connected seamlessly to a **MongoDB Atlas Cloud Database**, the tool autosaves every conversation you have. Previous dynamic chats are mapped securely inside a "Chat History" sidebar to reload sessions asynchronously.
6. **Live Streaming UI:**
   - Built via Streamlit. Fully capable of OpenAI payload streaming so users don't wait awkwardly while an entire page generates.

---

## 🛠️ Tech Stack & Requirements

### Infrastructure & Backends
- **Python 3.12+**
- **Streamlit** (Frontend framework & reactivity)
- **MongoDB** (Database history persistence)

### AI Tooling & Frameworks
- **OpenAI API** (`gpt-4-turbo` for text/vision generation capability & `text-embedding-3-small` for dense vectors)
- **LangChain** (Structuring FAISS and data abstractions)
- **Hugging Face (`sentence-transformers`) & `torchvision`** (Underlying core algorithms utilized for evaluating visual cross-modals)

### Parsing & Data Scraping
- `pytesseract`, `Pillow`, `pdf2image`, `pdfplumber`

---

## 🚀 Setting Up the Project

### 1. Requirements

Ensure you have **Tesseract OCR** formally installed on your host machine to allow image-to-text processing. 

Install all the application requirements via Virtual Environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configurations

You must specify a few critical keys in an active `.env` file located at the root of the project:

```env
# Required for text-generation & core vector embeddings
OPENAI_API_KEY="sk-proj-YourKeyHere"
OPENAI_VISION_MODEL="gpt-4o-mini"

# Required for thread persistence in the UI 
MONGO_URI="mongodb+srv://<user>:<password>@cluster0...mongodb.net/YourDB"
```

### 3. Execution

Launch your native application server directly through Streamlit:

```bash
streamlit run app.py
```
*Note: Make sure that you are utilizing your active virtual environment when executing, to prevent local package collisions.*

---

## 🗂️ Codebase Architecture

- **`app.py`:** The primary orchestrator handling Streamlit states, managing frontend Chat flows, pushing MongoDB sync requests, and interacting directly with multimodal retrieval logic constraints.
- **`ingestion.py`:** Standard handler breaking down PDFs and Images. Stores physical file derivatives cleanly into an internal `extracted_images` buffer directory.
- **`chunker.py`:** Utility built to slice enormous text scripts down to manageable `1200` token chunks keeping critical surrounding metadata intact.
- **`hybrid_retriever.py`:** The core search algorithm dynamically intersecting BM25 hits alongside dense scoring rules.
- **`reranker.py` & `llm_query.py`:** Injects `sentence-transformers` CLIP integrations scoring content and querying OpenAI's API sequentially via structured system prompts.
- **`db_utils.py`:** Abstracted controller communicating to MongoDB to save/recall active thread logs.
