
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .system import RetrievalResult
from .cosine import topk_cosine
from ..catalog import Catalog
from ..features import l2_normalize

def build_early_fusion_matrix(
    X_lyrics: np.ndarray,
    X_audio: np.ndarray,
    X_video: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """
    Weighted concatenation:
    - assumes X_* are already L2-normalized row-wise
    - scales each block by sqrt(weight)
    - concatenates and L2-normalizes again
    """
    wL, wA, wV = weights
    XL = X_lyrics * np.sqrt(wL)
    XA = X_audio  * np.sqrt(wA)
    XV = X_video  * np.sqrt(wV)
    X = np.concatenate([XL, XA, XV], axis=1)
    return l2_normalize(X)

def early_fusion_algo(
    catalog: Catalog,
    qidx: int,
    k: int,
    seed: Optional[int] = None,
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
) -> RetrievalResult:
    if catalog.X_lyrics is None or catalog.X_audio is None or catalog.X_video is None:
        raise ValueError("early_fusion requires X_lyrics, X_audio, X_video to be loaded.")

    # If precomputed exists, trust it (assumes it matches the intended weights)
    if catalog.X_early is None:
        catalog.X_early = build_early_fusion_matrix(catalog.X_lyrics, catalog.X_audio, catalog.X_video, weights)

    idx, scores = topk_cosine(qidx, catalog.X_early, k)
    return RetrievalResult(
        query_id=catalog.ids[qidx],
        algo="early_fusion",
        k=k,
        ranked_ids=[catalog.ids[i] for i in idx],
        scores=scores.tolist(),
    )

