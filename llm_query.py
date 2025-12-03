# llm_query.py
import base64
import io
import os
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Use the new OpenAI client. Ensure OPENAI_API_KEY is set in environment or .env.
client = OpenAI()

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
6. When information is present, cite specific passages:
      (Passage 1), (Passage 3), etc.
7. When interpreting images (charts, graphs, maps, diagrams):
   - Describe trends you can SEE visually.
   - Mention the graph title if available.
   - Do NOT hallucinate numbers not visible in the image.
8. When working with tables, numeric data, bullet lists, and structured text:
   - Extract values exactly as shown.
   - Be precise and avoid fabrication.
9. ALWAYS structure your answer like this:

=== ANSWER ===
A clear, concise response to the question.

=== EVIDENCE (Cited Passages) ===
• Fact 1 — (Passage X)
• Fact 2 — (Passage Y)
• Image observation — (Image Page Z)

If nothing relevant is found:
Return ONLY this sentence, no extra text:
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
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def get_image_embedding(image_path: str, model: str = "clip-vit-base-patch32") -> List[float]:
    """Get embedding for an image using CLIP-like model."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    response = client.embeddings.create(
        model=model,
        input=[{"type": "image_url",
               "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
    )
    return response.data[0].embedding

def cross_modal_rerank(question: str, passages: List[Dict], images: List[Dict]) -> List[Dict]:
    """Rerank passages and images based on cross-modal similarity to the question."""
    if not (passages or images):
        return []
    
    # Get question embedding
    try:
        question_embedding = np.array(get_embedding(question))
    except Exception as e:
        print(f"Error getting question embedding: {e}")
        return passages + images
    
    # Score passages
    scored_items = []
    
    # Process text passages
    for i, passage in enumerate(passages):
        try:
            text = passage.get("page_content", "")
            if not text:
                continue
                
            # Get text embedding and compute similarity
            text_embedding = np.array(get_embedding(text))
            text_sim = cosine_similarity(
                question_embedding.reshape(1, -1),
                text_embedding.reshape(1, -1)
            )[0][0]
            
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
                
            # Get image embedding and compute similarity
            img_embedding = np.array(get_image_embedding(img_path))
            img_sim = cosine_similarity(
                question_embedding.reshape(1, -1),
                img_embedding.reshape(1, -1)
            )[0][0]
            
            # Store with metadata
            scored_items.append({
                **img,
                "type": "image",
                "similarity_score": float(img_sim),
                "original_index": j
            })
        except Exception as e:
            print(f"Error processing image {j}: {e}")
    
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

def query_openai_chat(passages, images, question, model="gpt-4-turbo", use_reranking=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Apply cross-modal reranking if enabled
    if use_reranking:
        reranked_items = cross_modal_rerank(question, passages, images)
        # Split back into passages and images
        passages = [item for item in reranked_items if "page_content" in item]
        images = [item for item in reranked_items if "image_path" in item]
    
    # Add text content
    prompt_text = build_prompt_text(passages, question)
    content = [{"type": "text", "text": prompt_text}]
    
    # Add images using the correct format for GPT-4 Vision
    for img in images or []:
        img_path = img.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
            
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }
        })
    
    # Add user message with all content
    messages.append({"role": "user", "content": content})
    
    # Call OpenAI with the correct message format
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

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
