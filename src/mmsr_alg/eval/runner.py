
from __future__ import annotations
from typing import Dict, List
from ..catalog import Catalog
from .relevance import is_relevant
from .metrics_accuracy import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k

def evaluate_one_query(catalog: Catalog, query_id: str, ranked_ids: List[str], k: int) -> Dict[str, float]:
    qidx = catalog.id_to_idx[query_id]
    gq = catalog.genres[qidx] if catalog.genres is not None else set()

    rels = []
    for tid in ranked_ids[:k]:
        tidx = catalog.id_to_idx[tid]
        gt = catalog.genres[tidx] if catalog.genres is not None else set()
        rels.append(is_relevant(gq, gt))

    # total relevant in whole catalog (excluding query)
    total_rel = 0
    if catalog.genres is not None:
        for i, g in enumerate(catalog.genres):
            if i == qidx: 
                continue
            if len(gq.intersection(g)) > 0:
                total_rel += 1

    return {
        f"precision@{k}": precision_at_k(rels, k),
        f"recall@{k}": recall_at_k(rels, total_rel, k),
        f"mrr@{k}": mrr_at_k(rels, k),
        f"ndcg@{k}": ndcg_at_k(rels, total_rel, k),
    }
