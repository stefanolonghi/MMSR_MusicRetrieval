from pathlib import Path
import numpy as np
from mmsr_alg.features import load_feature_matrix, l2_normalize

def load_and_normalize_split(parts, id_to_idx):
    """
    Carica feature divise in più file TSV, le combina correttamente e normalizza.
    
    parts: lista di Path o stringhe dei file part1, part2, ...
    id_to_idx: dizionario id -> indice nel catalogo
    """
    X_full = None

    for part_file in parts:
        X_part = load_feature_matrix(Path(part_file), id_to_idx)
        if X_full is None:
            X_full = X_part
        else:
            X_full += X_part  # somma le feature per non duplicare righe

    return l2_normalize(X_full)