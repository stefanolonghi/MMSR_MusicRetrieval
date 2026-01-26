# src/mmsr_alg/retrieval/fusion_early.py
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np

from .system import RetrievalResult
from .cosine import topk_cosine
from ..catalog import Catalog
from ..features import l2_normalize

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

def build_early_fusion_from_blocks(blocks: List[np.ndarray], weights: Tuple[float, ...]) -> np.ndarray:
    if len(blocks) != len(weights):
        raise ValueError("blocks and weights must have same length")
    if len(blocks) < 2:
        raise ValueError("early fusion requires at least 2 modalities")
    w = np.array(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    scaled = [X * np.sqrt(wi) for X, wi in zip(blocks, w)]
    Xf = np.concatenate(scaled, axis=1)
    return l2_normalize(Xf)

def early_fusion_combo_algo(
    cat: Catalog,
    qidx: int,
    k: int,
    combo: Tuple[str, ...],
    seed: Optional[int] = None,
    weights: Optional[Tuple[float, ...]] = None,
) -> RetrievalResult:
    # default equal weights
    if weights is None:
        weights = tuple([1.0 / len(combo)] * len(combo))
    if len(weights) != len(combo):
        raise ValueError("weights length must match combo length")

    blocks = [_get_matrix(cat, m) for m in combo]

    # cache per (combo, weights)
    if not hasattr(cat, "X_early_cache") or cat.X_early_cache is None:
        cat.X_early_cache = {}

    key = (combo, tuple(map(float, weights)))
    if key not in cat.X_early_cache:
        cat.X_early_cache[key] = build_early_fusion_from_blocks(blocks, key[1])

    X = cat.X_early_cache[key]
    idx, scores = topk_cosine(qidx, X, k)

    return RetrievalResult(
        query_id=cat.ids[qidx],
        algo=f"early_{'-'.join(combo)}",
        k=k,
        ranked_ids=[cat.ids[i] for i in idx],
        scores=scores.tolist(),
    )


def early_fusion_algo(
    catalog: Catalog,
    qidx: int,
    k: int,
    seed: Optional[int] = None,
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
) -> RetrievalResult:
    """
    Backwards-compatible wrapper: early fusion over ALL 3 modalities (audio, text, video).
    Keeps old registry imports working.
    """
    return early_fusion_combo_algo(
        catalog,
        qidx,
        k,
        combo=("audio", "text", "video"),
        weights=weights,
        seed=seed,
    )
