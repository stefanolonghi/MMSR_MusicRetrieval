
from __future__ import annotations
from typing import Dict, List, Optional, Set
import numpy as np
from ..catalog import Catalog

def coverage_at_k(all_ranked_ids: Dict[str, List[str]], k: int, N: int) -> float:
    """
    Coverage@k = (# unique retrieved tracks across all queries) / N
    """
    seen: Set[str] = set()
    for rids in all_ranked_ids.values():
        seen.update(rids[:k])
    return len(seen) / max(N, 1)

def pop_at_k(catalog: Catalog, all_ranked_ids: Dict[str, List[str]], k: int) -> Optional[float]:
    """
    Pop@k = macro-average popularity of retrieved tracks in top-k.
    Uses nanmean so missing popularity doesn't blank out results.
    Returns None if popularity isn't loaded.
    """
    if catalog.popularity is None:
        return None

    per_query_means = []
    for rids in all_ranked_ids.values():
        idxs = [catalog.id_to_idx[tid] for tid in rids[:k]]
        vals = catalog.popularity[idxs]
        m = float(np.nanmean(vals))
        if not np.isnan(m):
            per_query_means.append(m)

    if not per_query_means:
        return None
    return float(np.mean(per_query_means))

