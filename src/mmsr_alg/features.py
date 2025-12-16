from pathlib import Path
import numpy as np
import pandas as pd

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)

def load_feature_matrix(path: Path, id_to_idx: dict) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        raise ValueError(f"{path.name} must contain an 'id' column")

    feats = df.drop(columns=["id"]).to_numpy(dtype=np.float32)
    X = np.zeros((len(id_to_idx), feats.shape[1]), dtype=np.float32)

    for i, tid in enumerate(df["id"].astype(str)):
        if tid in id_to_idx:
            X[id_to_idx[tid]] = feats[i]

    return X
