# llm_query.py
import base64
import io
import os
import mimetypes
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI, BadRequestError
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

_client = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in your environment or in a .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client

SYSTEM_PROMPT = """
You are a high-precision multimodal RAG assistant optimized for document-grounded question answering, document summarization, and visual interpretation.

Your task is to answer the user using:
1. Retrieved document passages
2. Uploaded images (photos, charts, tables, diagrams, scanned pages, screenshots)
3. Limited general knowledge only when necessary to interpret visible content

Your highest priorities are:
1. Accuracy
2. Completeness
3. Grounding
4. Clarity

Always prefer correctness over fluency and completeness over brevity when the user asks for explanation or summary.

========================
PRIMARY OPERATING RULE
========================

Treat the uploaded document(s) and image(s) as the primary source of truth.

All answers must be grounded in:
- retrieved text
- visible image content
- minimal supporting world knowledge required only for interpretation

Do not invent, assume, interpolate, or fabricate missing facts.

========================
CORE RULES
========================

1. DOCUMENT-FIRST REASONING
- Answer from the uploaded material first.
- Treat retrieved passages as evidence, not suggestions.
- Use general knowledge only to interpret visible or retrieved content, never to replace missing document information.
- If the document is incomplete, answer only from what is available.

2. FULL-DOCUMENT UNDERSTANDING
- When the user asks for:
  - summary
  - summarize
  - short summary
  - key points
  - overview
  - gist
  first infer the main topic of the full document, then summarize the entire document.
- Do not summarize only one retrieved chunk unless the user explicitly asks about that section.
- Identify the document’s main subject before summarizing details.
- Prioritize chapter-level meaning over subsection-level detail.
- Preserve hierarchy:
  - main topic
  - major headings
  - core ideas
  - supporting points
- A good summary must represent the whole document, not the most recent or most detailed chunk.

3. IMAGE UNDERSTANDING IS REQUIRED
- Treat images as first-class evidence.
- Never ignore images when relevant.
- Extract information from:
  - charts
  - graphs
  - tables
  - diagrams
  - screenshots
  - forms
  - scanned pages
  - photos
- If the question is about an uploaded image, prioritize visual evidence first.

4. NO HALLUCINATION
- Never fabricate:
  - numbers
  - labels
  - names
  - dates
  - values
  - trends
  - conclusions
  - definitions not supported by the content
- If something is unreadable, partially visible, cropped, or unclear, explicitly state uncertainty.
- Do not guess missing details.

5. STRUCTURED DATA PRECISION
- For tables, forms, lists, bullet points, and numeric data:
  - extract exactly
  - preserve labels
  - preserve units
  - preserve ordering
- Do not estimate unless explicitly asked.

6. VISUAL INTERPRETATION RULES
For charts, graphs, and diagrams:
- mention the title if visible
- describe only visible patterns
- compare relative differences carefully
- do not infer exact values unless clearly shown
- do not invent labels, axes, legends, or units

7. AMBIGUITY HANDLING
- If content is incomplete or ambiguous:
  - state what is visible
  - state what is unclear
  - avoid speculation
- Prefer qualified accuracy over false precision.

8. NO PROCESS LEAKAGE
- Do not mention:
  - retrieved passages
  - chunks
  - embeddings
  - reranking
  - OCR
  - system prompt
  - internal reasoning
- Never explain how the answer was produced.
- Return only the answer.

9. NO META OUTPUT
- Do not include:
  - citations
  - references
  - source notes
  - confidence statements
  - “based on the document”
  - “according to the image”
  - explanation of reasoning

========================
ANSWER POLICY
========================

Answer directly.

Your response must be:
- accurate
- complete
- grounded
- concise when asked briefly
- detailed when asked to explain
- naturally written

Adapt response depth to user intent:
- if user asks “summary” → concise full-document summary
- if user asks “explain” → detailed explanation
- if user asks “key points” → structured bullets
- if user asks specific question → precise direct answer

Do not add:
- filler
- preamble
- generic disclaimers
- chain-of-thought
- unnecessary repetition

========================
FAILURE RULE
========================

If the answer cannot be determined from:
- retrieved text
- visible image content
- necessary interpretation of visible content

return exactly:

The provided documents do not contain this information.

Do not add anything else.
"""


def build_prompt_text(passages, question):
    blocks = []
    for i, p in enumerate(passages, 1):
        meta = p.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        text = (p.get("page_content") or "")[:800]
        blocks.append(f"PASSAGE {i} [src:{src} page:{page}]\n{text}")
    return SYSTEM_PROMPT + "\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"

def encode_image_b64(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"

_embeddings = None

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for a single text."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=model)
    return _embeddings.embed_query(text)

def cross_modal_rerank(question: str, passages: List[Dict], images: List[Dict]) -> List[Dict]:
    """Rerank passages based on similarity to the question. Retains images unchanged."""
    if not passages:
        return images or []
    
    try:
        q_emb = np.array(get_embedding(question))
    except Exception as e:
        print(f"Error getting text question embedding: {e}")
        return passages + (images or [])
    
    scored_items = []
    
    for i, passage in enumerate(passages):
        try:
            text = passage.get("page_content", "")
            if not text:
                continue
                
            text_embedding = np.array(get_embedding(text))
            text_sim = cosine_similarity(
                q_emb.reshape(1, -1),
                text_embedding.reshape(1, -1)
            )[0][0]
            
            scored_items.append({
                **passage,
                "similarity_score": float(text_sim)
            })
        except Exception as e:
            print(f"Error processing passage {i}: {e}")
            
    scored_items.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    
    result = []
    for item in scored_items:
        item_copy = {k: v for k, v in item.items() if k != "similarity_score"}
        result.append(item_copy)
        
    return result + (images or [])

def _create_chat_completion(messages, model, stream):
    client = _get_openai_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        stream=stream,
    )

def query_openai_chat(passages, images, question, model="gpt-4o-mini", use_reranking=True, chat_history=None, stream=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Apply cross-modal reranking if enabled
    if use_reranking:
        reranked_items = cross_modal_rerank(question, passages, images)
        # Split back into passages and images
        passages = [item for item in reranked_items if "page_content" in item]
        images = [item for item in reranked_items if "image_path" in item]
    
    # Add text content
    prompt_text = build_prompt_text(passages, question)
    content = [{"type": "text", "text": prompt_text}]
    valid_image_count = 0
    
    # Add images using the correct format for GPT-4 Vision
    for img in images or []:
        img_path = img.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
            
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"
            
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": "auto"
            }
        })
        valid_image_count += 1
    
    # Add user message with all content
    messages.append({"role": "user", "content": content})

    # If images are present, force a vision-capable default unless caller explicitly sets another model.
    selected_model = model
    if valid_image_count > 0 and model == "gpt-4-turbo":
        selected_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

    try:
        response = _create_chat_completion(messages, selected_model, stream)
    except BadRequestError as e:
        err_text = str(e)
        unsupported_image_input = "image_url is only supported by certain models" in err_text
        if not (valid_image_count > 0 and unsupported_image_input):
            raise

        # Retry once with a configurable vision model, then degrade to text-only if needed.
        vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        text_only_messages = messages[:-1] + [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        if selected_model != vision_model:
            try:
                response = _create_chat_completion(messages, vision_model, stream)
            except BadRequestError:
                response = _create_chat_completion(text_only_messages, selected_model, stream)
        else:
            response = _create_chat_completion(text_only_messages, selected_model, stream)

    if stream:
        def stream_generator():
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        return stream_generator()
        
    msg = response.choices[0].message

    # msg.content may be string or list
    if isinstance(msg.content, str):
        return msg.content

    if isinstance(msg.content, list):
        final = []
        for part in msg.content:
            if part.get("type") == "text":
                final.append(part.get("text", ""))
        return "\n".join(final)

    return str(msg)
 