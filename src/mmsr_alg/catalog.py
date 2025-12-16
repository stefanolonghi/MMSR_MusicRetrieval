from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class Catalog:
    tracks: pd.DataFrame
    ids: List[str]
    id_to_idx: Dict[str, int]

    X_lyrics: Optional[np.ndarray] = None
    X_audio: Optional[np.ndarray] = None
    X_video: Optional[np.ndarray] = None
    X_early: Optional[np.ndarray] = None

    genres: Optional[List[set]] = None
    popularity: Optional[np.ndarray] = None
