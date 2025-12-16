
from __future__ import annotations
from typing import List
import math

def precision_at_k(rels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(rels[:k]) / k

def recall_at_k(rels: List[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(rels[:k]) / total_relevant

def mrr_at_k(rels: List[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r == 1:
            return 1.0 / i
    return 0.0

def ndcg_at_k(rels: List[int], total_relevant: int, k: int) -> float:
    """
    Correct nDCG@k for binary relevance:
    - DCG computed from retrieved relevance vector
    - IDCG computed from the *maximum possible* relevance in top-k,
      i.e., min(total_relevant, k) ones first.
    """
    def dcg(vals: List[int]) -> float:
        s = 0.0
        for i, r in enumerate(vals, start=1):
            if r:
                s += 1.0 / math.log2(i + 1)
        return s

    rels_k = rels[:k]
    dcg_val = dcg(rels_k)

    ideal_ones = min(total_relevant, k)
    if ideal_ones == 0:
        return 0.0

    idcg_val = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_ones + 1))
    return dcg_val / idcg_val

