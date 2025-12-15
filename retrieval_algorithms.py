from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Small data container
# -----------------------------
@dataclass
class RetrievalOutput:
    ids: List[str]
    scores: Optional[List[float]]  # None for random


# -----------------------------
# Utilities
# -----------------------------
def _safe_parse_list(x) -> List[str]:
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(t) for t in v]
            if isinstance(v, str):
                return [v]
        except Exception:
            return []
    return []


def build_genres_dict(df_genres: pd.DataFrame) -> Dict[str, List[str]]:
    # expects columns: id + genre (or genres/genre_list)
    genre_col = None
    for col in df_genres.columns:
        if col.lower() in ["genre", "genres", "genre_list"]:
            genre_col = col
            break
    if genre_col is None:
        raise ValueError("Could not find genre column in genres dataframe.")

    parsed = df_genres.copy()
    parsed[genre_col] = parsed[genre_col].apply(_safe_parse_list)
    return dict(zip(parsed["id"].astype(str), parsed[genre_col]))


def _id_to_idx(ids: List[str]) -> Dict[str, int]:
    return {tid: i for i, tid in enumerate(ids)}


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _topk_cosine(qidx: int, X_norm: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = X_norm @ X_norm[qidx]
    sims[qidx] = -np.inf
    if k >= len(sims) - 1:
        idx = np.argsort(sims)[::-1]
    else:
        part = np.argpartition(sims, -k)[-k:]
        idx = part[np.argsort(sims[part])[::-1]]
    return idx[:k], sims[idx[:k]]


# -----------------------------
# Feature loading (cached by UI if you want)
# -----------------------------
def load_feature_matrix_tsv(path: str, ids: List[str]) -> np.ndarray:
    """
    TSV expected: column 'id' + feature columns (numeric).
    Returns matrix aligned to `ids` order. Missing ids -> zero row.
    """
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        raise ValueError(f"{path} must contain an 'id' column")

    feat_df = df.drop(columns=["id"])
    feats = feat_df.to_numpy(dtype=np.float32, copy=False)

    id2i = _id_to_idx(ids)
    X = np.zeros((len(ids), feats.shape[1]), dtype=np.float32)

    for row_i, tid in enumerate(df["id"].astype(str).tolist()):
        j = id2i.get(tid)
        if j is not None:
            X[j] = feats[row_i]

    return X


# -----------------------------
# Algorithms
# -----------------------------
def retrieve_random(query_id: str, ids: List[str], k: int, seed: Optional[int] = None) -> RetrievalOutput:
    rng = np.random.default_rng(seed)
    candidates = [tid for tid in ids if tid != query_id]
    rng.shuffle(candidates)
    out = candidates[:k]
    return RetrievalOutput(ids=out, scores=None)


def retrieve_unimodal_cosine(
    query_id: str,
    ids: List[str],
    X_norm: np.ndarray,
    k: int
) -> RetrievalOutput:
    id2i = _id_to_idx(ids)
    qidx = id2i[query_id]
    idx, scores = _topk_cosine(qidx, X_norm, k)
    out_ids = [ids[i] for i in idx]
    return RetrievalOutput(ids=out_ids, scores=scores.tolist())


def retrieve_late_fusion(
    query_id: str,
    ids: List[str],
    X_norm_list: List[np.ndarray],
    k: int,
    weights: Optional[List[float]] = None
) -> RetrievalOutput:
    """
    Late fusion over cosine similarity scores. Uses min-max normalization per modality per query.
    """
    id2i = _id_to_idx(ids)
    qidx = id2i[query_id]

    m = len(X_norm_list)
    if weights is None:
        weights = [1.0 / m] * m
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    fused = None

    for w, Xn in zip(weights, X_norm_list):
        sims = Xn @ Xn[qidx]
        sims[qidx] = -np.inf

        # min-max normalization ignoring -inf
        finite = np.isfinite(sims)
        if finite.any():
            smin, smax = sims[finite].min(), sims[finite].max()
            if smax > smin:
                sims = (sims - smin) / (smax - smin)
            else:
                sims = np.zeros_like(sims)
                sims[qidx] = -np.inf
        else:
            sims = np.zeros_like(sims)
            sims[qidx] = -np.inf

        fused = sims * w if fused is None else fused + sims * w

    fused[qidx] = -np.inf
    if k >= len(fused) - 1:
        idx = np.argsort(fused)[::-1][:k]
    else:
        part = np.argpartition(fused, -k)[-k:]
        idx = part[np.argsort(fused[part])[::-1]]

    out_ids = [ids[i] for i in idx]
    out_scores = fused[idx].tolist()
    return RetrievalOutput(ids=out_ids, scores=out_scores)


def build_early_fusion_matrix(X_norm_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Early fusion = concatenate modality vectors (optionally weighted), then renormalize.
    """
    m = len(X_norm_list)
    if weights is None:
        weights = [1.0] * m
    weights = np.asarray(weights, dtype=np.float32)

    blocks = []
    for w, Xn in zip(weights, X_norm_list):
        blocks.append(Xn * w)
    X_cat = np.concatenate(blocks, axis=1)
    return _l2_normalize_rows(X_cat)


# -----------------------------
# Minimal metrics (optional; UI can keep placeholders for now)
# -----------------------------
def genre_overlap_relevant(genres_dict: Dict[str, List[str]], qid: str, tid: str) -> bool:
    gq = set(genres_dict.get(qid, []))
    gt = set(genres_dict.get(tid, []))
    return len(gq.intersection(gt)) > 0
