import numpy as np

def topk_cosine(qidx: int, X: np.ndarray, k: int):
    sims = X @ X[qidx]
    sims[qidx] = -np.inf
    idx = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]
