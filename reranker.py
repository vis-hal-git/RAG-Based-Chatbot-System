# reranker.py
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from llm_query import get_embedding, get_image_embedding

def cross_modal_rerank(question: str,
                       text_items: List[Dict[str,Any]],
                       image_items: List[Dict[str,Any]],
                       text_weight: float = 0.7,
                       image_weight: float = 0.3,
                       top_k: int = 10) -> List[Dict[str,Any]]:
    """
    text_items: [{'page_content':..., 'metadata':{...}}...]
    image_items: [{'image_path':..., 'meta':{...}}...]
    Returns merged list of top_k items (text- and image-docs) sorted by fused similarity.
    """
    # compute question embedding
    try:
        q_emb = np.array(get_embedding(question))
    except Exception:
        # fallback: return text items untouched
        return (text_items[:top_k] + image_items[:top_k])[:top_k]

    scored = []

    # score text items
    for t in text_items:
        txt = t.get("page_content","")[:2000]
        if not txt.strip():
            continue
        try:
            emb = np.array(get_embedding(txt))
            sim = float(cosine_similarity(q_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
        except Exception:
            sim = 0.0
        scored.append({"type":"text","score": text_weight*sim, "item": t})

    # score image items
    for im in image_items:
        ip = im.get("image_path") or im.get("meta", {}).get("image_path")
        if not ip:
            continue
        try:
            emb = np.array(get_image_embedding(ip))
            sim = float(cosine_similarity(q_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
        except Exception:
            sim = 0.0
        scored.append({"type":"image","score": image_weight*sim, "item": im})

    # sort by score desc and return underlying items
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    merged = []
    for s in scored[:top_k]:
        merged.append(s["item"])
    return merged
