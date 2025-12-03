# evaluation.py
import time
import numpy as np

def compute_mrr(ranked_lists, ground_truth_idx):
    """
    ranked_lists: list of lists of indices (retrieved order per query)
    ground_truth_idx: list of the ground-truth index for each query
    """
    rr = []
    for ranked, gt in zip(ranked_lists, ground_truth_idx):
        try:
            pos = ranked.index(gt) + 1
            rr.append(1.0/pos)
        except ValueError:
            rr.append(0.0)
    return float(np.mean(rr))

def recall_at_k(retrieved_idxs, gt_idx, k):
    return 1.0 if gt_idx in retrieved_idxs[:k] else 0.0
