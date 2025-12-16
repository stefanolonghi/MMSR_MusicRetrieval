
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .system import RetrievalResult
from ..catalog import Catalog

def _cosine_scores(qidx: int, X: np.ndarray) -> np.ndarray:
    # X must already be L2-normalized row-wise
    s = X @ X[qidx]         # (N,)
    s[qidx] = -np.inf
    return s

def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    """
    Per-query min-max normalize to [0,1], ignoring -inf entries.
    If all finite scores equal -> returns zeros for finite entries.
    """
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

def late_fusion_algo(
    catalog: Catalog,
    qidx: int,
    k: int,
    seed: Optional[int] = None,
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    normalize: bool = True,
) -> RetrievalResult:
    """
    Late fusion over (lyrics, audio, video) similarity scores.
    """
    if catalog.X_lyrics is None or catalog.X_audio is None or catalog.X_video is None:
        raise ValueError("late_fusion requires X_lyrics, X_audio, X_video to be loaded.")

    wL, wA, wV = weights

    sL = _cosine_scores(qidx, catalog.X_lyrics)
    sA = _cosine_scores(qidx, catalog.X_audio)
    sV = _cosine_scores(qidx, catalog.X_video)

    if normalize:
        sL = _minmax_norm(sL)
        sA = _minmax_norm(sA)
        sV = _minmax_norm(sV)

    fused = wL * sL + wA * sA + wV * sV
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
        algo="late_fusion",
        k=k,
        ranked_ids=[catalog.ids[i] for i in idx],
        scores=fused[idx].tolist(),
    )
