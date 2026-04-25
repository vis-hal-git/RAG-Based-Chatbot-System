# llm_query.py
import base64
import io
import os
import mimetypes
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI, BadRequestError
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer

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
You are an advanced Vision-RAG assistant designed to answer questions strictly using the
content extracted from the user’s uploaded documents (PDF, images, text files).

Your responsibilities:
1. Use ONLY the retrieved passages and OCR/image content. No outside knowledge.
2. For any text-based or visual information (tables, charts, figures), interpret it accurately.
3. Always attempt to answer the user’s question if ANY relevant information exists in the 
   provided passages—INCLUDING summaries, explanations, trends, key findings, or visual
   interpretations.
4. Do NOT respond with: 
      “The provided documents do not contain this information”
   unless **absolutely no retrieved passage or image contains any relevant details.**
5. If a question is broad, such as “summarize in short,” produce a concise summary based on 
   the passages and images.
    6. When interpreting images (charts, graphs, maps, diagrams):
    - Describe trends you can SEE visually.
    - Mention the graph title if available.
    - Do NOT hallucinate numbers not visible in the image.
    7. When working with tables, numeric data, bullet lists, and structured text:
    - Extract values exactly as shown.
    - Be precise and avoid fabrication.
    8. Respond with a clear, concise answer only. Do not include citations or any evidence section.

    If nothing relevant is found: return ONLY this sentence, no extra text:
    "The provided documents do not contain this information."

Your goal: Provide the most accurate, clear, and helpful answer based solely on the 
retrieved document content, including text, OCR extractions, tables, and visual elements.
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

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for a single text."""
    client = _get_openai_client()
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

_clip_model = None

def get_image_embedding(image_path: str, model: str = "clip-ViT-B-32") -> List[float]:
    """Get embedding for an image using CLIP sentence-transformer."""
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(model)
        
    img = Image.open(image_path)
    # encode handles image input for CLIP models in sentence-transformers
    embedding = _clip_model.encode(img)
    return embedding.tolist()

def get_clip_text_embedding(text: str, model: str = "clip-ViT-B-32") -> List[float]:
    """Get CLIP text embedding compatible with CLIP image embeddings."""
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(model)
    embedding = _clip_model.encode(text)
    return embedding.tolist()

def cross_modal_rerank(question: str, passages: List[Dict], images: List[Dict]) -> List[Dict]:
    """Rerank passages and images based on cross-modal similarity to the question."""
    if not (passages or images):
        return []
    
    # Question embedding for text-text similarity
    question_text_embedding = None
    try:
        question_text_embedding = np.array(get_embedding(question))
    except Exception as e:
        print(f"Error getting text question embedding: {e}")

    # Question embedding for CLIP text-image similarity
    question_image_embedding = None
    try:
        question_image_embedding = np.array(get_clip_text_embedding(question))
    except Exception as e:
        print(f"Error getting CLIP question embedding: {e}")

    if question_text_embedding is None and question_image_embedding is None:
        return passages + images
    
    # Score passages
    scored_items = []
    
    # Process text passages
    for i, passage in enumerate(passages):
        try:
            text = passage.get("page_content", "")
            if not text:
                continue
                
            if question_text_embedding is not None:
                text_embedding = np.array(get_embedding(text))
                text_sim = cosine_similarity(
                    question_text_embedding.reshape(1, -1),
                    text_embedding.reshape(1, -1)
                )[0][0]
            else:
                text_sim = 0.0
            
            # Store with metadata
            scored_items.append({
                **passage,
                "type": "text",
                "similarity_score": float(text_sim),
                "original_index": i
            })
        except Exception as e:
            print(f"Error processing passage {i}: {e}")
    
    # Process images
    for j, img in enumerate(images or []):
        try:
            img_path = img.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
                
            if question_image_embedding is not None:
                img_embedding = np.array(get_image_embedding(img_path))
                img_sim = cosine_similarity(
                    question_image_embedding.reshape(1, -1),
                    img_embedding.reshape(1, -1)
                )[0][0]
            else:
                img_sim = 0.0
            
            # Store with metadata
            scored_items.append({
                **img,
                "type": "image",
                "similarity_score": float(img_sim),
                "original_index": j
            })
        except Exception as e:
            print(f"Error processing image {j}: {e}")
    
    if not scored_items:
        return passages + images

    # Sort by similarity score in descending order
    scored_items.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    
    # Return top items, keeping original format
    result = []
    for item in scored_items:
        # Remove temporary fields and keep original structure
        item_copy = {k: v for k, v in item.items() 
                    if k not in ["similarity_score", "type", "original_index"]}
        result.append(item_copy)
        
    return result

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
