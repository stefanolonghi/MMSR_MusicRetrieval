import numpy as np
from .system import RetrievalResult

def random_algo(catalog, qidx, k, seed=None):
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(catalog.ids))
    candidates = candidates[candidates != qidx]
    rng.shuffle(candidates)

    idx = candidates[:k]
    return RetrievalResult(
        query_id=catalog.ids[qidx],
        algo="random",
        k=k,
        ranked_ids=[catalog.ids[i] for i in idx],
        scores=None
    )
