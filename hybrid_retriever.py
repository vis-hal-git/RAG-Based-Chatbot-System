# hybrid_retriever.py
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any
from llm_query import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

def build_bm25_corpus(chunks: List[Dict[str,Any]]):
    tokenized = [ (c["content"].split() if isinstance(c["content"], str) else [""]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def dense_scores(query: str, chunks: List[Dict[str,Any]]):
    q_emb = np.array(get_embedding(query))
    scores = []
    for c in chunks:
        try:
            emb = np.array(get_embedding(c["content"][:2000]))
            sim = float(cosine_similarity(q_emb.reshape(1,-1), emb.reshape(1,-1))[0][0])
        except Exception:
            sim = 0.0
        scores.append(sim)
    return np.array(scores)

def reciprocal_rank_fusion(bm25_scores, dense_scores, k=10):
    """
    bm25_scores: 1D array of bm25 raw scores
    dense_scores: 1D array of dense similarities (0..1)
    RRF: convert each ranking to reciprocal rank and sum.
    """
    n = len(bm25_scores)
    # build ranks (higher score -> rank 1)
    bm25_rank = (-bm25_scores).argsort().argsort()  # rank indices (0 best)
    dense_rank = (-dense_scores).argsort().argsort()
    # convert to reciprocal rank with constant
    K = 60.0
    rrf = 1.0/(bm25_rank + K) + 1.0/(dense_rank + K)
    # return top k indices
    top_idx = np.argsort(-rrf)[:k]
    return top_idx, rrf

def hybrid_retrieve(query: str, chunks: List[Dict[str,Any]], k:int=10):
    # BM25 part
    bm25, tokenized = build_bm25_corpus(chunks)
    tokenized_query = query.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    # Dense part
    dense = dense_scores(query, chunks)
    # RRF fusion
    top_idx, scores = reciprocal_rank_fusion(bm25_scores, dense, k=k)
    results = []
    for idx in top_idx:
        c = chunks[idx]
        results.append({"content": c["content"], "meta": c["meta"], "score": float(scores[idx])})
    return results
