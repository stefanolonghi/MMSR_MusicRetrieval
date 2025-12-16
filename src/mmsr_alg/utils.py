
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from .catalog import Catalog

def get_track_row(catalog: Catalog, tid: str) -> Dict[str, Any]:
    # fast lookup by idx -> row
    idx = catalog.id_to_idx[tid]
    row = catalog.tracks.iloc[idx]
    return {
        "id": tid,
        "artist": row.get("artist", None),
        "song": row.get("song", None),
        "album_name": row.get("album_name", None),
        "url": row.get("url", None),
        "genres": list(catalog.genres[idx]) if catalog.genres is not None else None,
    }

def decorate_result(catalog: Catalog, ranked_ids: List[str]) -> List[Dict[str, Any]]:
    return [get_track_row(catalog, tid) for tid in ranked_ids]
