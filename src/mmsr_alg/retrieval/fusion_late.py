# src/mmsr_alg/retrieval/fusion_late.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from typing import List  # add at top if not present


from .system import RetrievalResult
from ..catalog import Catalog

def _get_matrix(cat: Catalog, name: str) -> np.ndarray:
    if name == "text":
        X = cat.X_lyrics
    elif name == "audio":
        X = cat.X_audio
    elif name == "video":
        X = cat.X_video
    else:
        raise ValueError(f"Unknown modality '{name}'")
    if X is None:
        raise ValueError(f"Matrix for '{name}' not loaded")
    return X

def _cosine_scores(qidx: int, X: np.ndarray) -> np.ndarray:
    s = X @ X[qidx]
    s[qidx] = -np.inf
    return s

def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    out = scores.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros_like(out)
    mn = out[finite].min()
    mx = out[finite].max()
    if mx - mn < 1e-12:
        out[finite] = 0.0
        out[~finite] = -np.inf
        return out
    out[finite] = (out[finite] - mn) / (mx - mn)
    out[~finite] = -np.inf
    return out

def late_fusion_combo_algo(
    cat: Catalog,
    qidx: int,
    k: int,
    combo: Tuple[str, ...],
    seed: Optional[int] = None,
    weights: Optional[Tuple[float, ...]] = None,
    normalize: bool = True,
) -> RetrievalResult:
    if weights is None:
        weights = tuple([1.0 / len(combo)] * len(combo))
    if len(weights) != len(combo):
        raise ValueError("weights length must match combo length")

    fused = None
    for m, w in zip(combo, weights):
        X = _get_matrix(cat, m)
        s = _cosine_scores(qidx, X)
        if normalize:
            s = _minmax_norm(s)
        fused = (w * s) if fused is None else (fused + w * s)

    fused[qidx] = -np.inf

    # top-k
    if k >= len(fused) - 1:
        idx = np.argsort(fused)[::-1]
    else:
        part = np.argpartition(fused, -k)[-k:]
        idx = part[np.argsort(fused[part])[::-1]]
    idx = idx[:k]

    return RetrievalResult(
        query_id=cat.ids[qidx],
        algo=f"late_{'-'.join(combo)}",
        k=k,
        ranked_ids=[cat.ids[i] for i in idx],
        scores=fused[idx].tolist(),
    )


def late_fusion_algo(
    catalog: Catalog,
    qidx: int,
    k: int,
    seed: Optional[int] = None,
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    normalize: bool = True,
) -> RetrievalResult:
    """
    Backwards-compatible wrapper: late fusion over ALL 3 modalities (text, audio, video).
    Keeps old registry imports working.
    """
    return late_fusion_combo_algo(
        catalog,
        qidx,
        k,
        combo=("text", "audio", "video"),
        weights=weights,
        normalize=normalize,
    )

def late_fusion_custom(
    catalog: Catalog,
    qidx: int,
    k: int,
    matrices: List[np.ndarray],
    weights: List[float],
    seed: Optional[int] = None,
    normalize: bool = True,
) -> RetrievalResult:
    """
    Generic late fusion over arbitrary matrices (used for NN late fusion in your registry).
    """
    if len(matrices) != len(weights):
        raise ValueError("matrices and weights must have same length")
    if len(matrices) == 0:
        raise ValueError("need at least one matrix")

    fused = None
    for X, w in zip(matrices, weights):
        s = _cosine_scores(qidx, X)
        if normalize:
            s = _minmax_norm(s)
        fused = (w * s) if fused is None else (fused + w * s)

    fused[qidx] = -np.inf

    # top-k
    if k >= len(fused) - 1:
        idx = np.argsort(fused)[::-1]
    else:
        part = np.argpartition(fused, -k)[-k:]
        idx = part[np.argsort(fused[part])[::-1]]
    idx = idx[:k]

    return RetrievalResult(
        query_id=catalog.ids[qidx],
        algo="neural_late_fusion",
        k=k,
        ranked_ids=[catalog.ids[i] for i in idx],
        scores=fused[idx].tolist(),
    )
