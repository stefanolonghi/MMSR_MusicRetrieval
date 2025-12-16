
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..catalog import Catalog
from ..retrieval.system import RetrievalSystem
from .metrics_accuracy import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k
from .metrics_beyond import coverage_at_k, pop_at_k

def _build_genre_inverted_index(catalog: Catalog) -> Dict[str, List[int]]:
    """
    genre -> list of track indices having that genre
    """
    inv: Dict[str, List[int]] = {}
    if catalog.genres is None:
        return inv
    for i, gset in enumerate(catalog.genres):
        for g in gset:
            inv.setdefault(g, []).append(i)
    return inv

def _total_relevant_for_query(inv: Dict[str, List[int]], q_genres: set, qidx: int) -> int:
    """
    Total relevant tracks in whole catalog under genre-overlap definition
    (excluding query itself).
    """
    if not q_genres:
        return 0
    union = set()
    for g in q_genres:
        union.update(inv.get(g, []))
    if qidx in union:
        union.remove(qidx)
    return len(union)

def _binary_rels_for_retrieved(catalog: Catalog, qidx: int, ranked_ids: List[str], k: int) -> List[int]:
    """
    For top-k retrieved ids, compute binary relevance by genre overlap.
    """
    gq = catalog.genres[qidx] if catalog.genres is not None else set()
    rels = []
    for tid in ranked_ids[:k]:
        tidx = catalog.id_to_idx[tid]
        gt = catalog.genres[tidx] if catalog.genres is not None else set()
        rels.append(1 if (gq and gt and len(gq.intersection(gt)) > 0) else 0)
    return rels

def evaluate_algorithms(
    system: RetrievalSystem,
    algos: List[str],
    k_values: List[int],
    query_ids: List[str],
    out_dir: Path,
    store_lists: bool = True,
) -> pd.DataFrame:
    """
    Runs evaluation for multiple algorithms and k values.

    Writes:
    - outputs/results/metrics.csv
    - outputs/retrieval_lists/<algo>_top<maxK>.json (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_lists").mkdir(parents=True, exist_ok=True)

    catalog = system.catalog
    inv = _build_genre_inverted_index(catalog)

    maxK = max(k_values)
    rows = []

    for algo in algos:
        # 1) retrieve top maxK for each query once
        retrieval_lists: Dict[str, List[str]] = {}
        for qi, qid in enumerate(query_ids, start=1):
            res = system.retrieve(qid, k=maxK, algo=algo)
            retrieval_lists[qid] = res.ranked_ids

            # lightweight progress
            if qi % 500 == 0:
                print(f"[{algo}] processed {qi}/{len(query_ids)} queries")

        if store_lists:
            path = out_dir / "retrieval_lists" / f"{algo}_top{maxK}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(retrieval_lists, f)

        # 2) compute metrics for each k
        N = len(catalog.ids)
        for k in k_values:
            p_list, r_list, mrr_list, ndcg_list = [], [], [], []

            for qid in query_ids:
                qidx = catalog.id_to_idx[qid]
                gq = catalog.genres[qidx] if catalog.genres is not None else set()
                total_rel = _total_relevant_for_query(inv, gq, qidx)

                rels = _binary_rels_for_retrieved(catalog, qidx, retrieval_lists[qid], k)

                p_list.append(precision_at_k(rels, k))
                r_list.append(recall_at_k(rels, total_rel, k))
                mrr_list.append(mrr_at_k(rels, k))
                ndcg_list.append(ndcg_at_k(rels, total_rel, k))

            cov = coverage_at_k(retrieval_lists, k=k, N=N)
            pop = pop_at_k(catalog, retrieval_lists, k=k)

            rows.append({
                "algo": algo,
                "k": k,
                "precision": float(np.mean(p_list)) if p_list else 0.0,
                "recall": float(np.mean(r_list)) if r_list else 0.0,
                "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
                "ndcg": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
                "coverage": float(cov),
                "pop": (None if pop is None else float(pop)),
                "num_queries": len(query_ids),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    return df
