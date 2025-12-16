
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from .catalog import Catalog

def _read_tsv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)

def _parse_genre(cell: str):
    if cell is None or cell == "" or str(cell).lower() == "nan":
        return []
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return [str(x).strip() for x in val]
    except Exception:
        pass
    return []

def load_catalog(retrieval_dir: Path) -> Catalog:
    info = _read_tsv_str(retrieval_dir / "id_information_mmsr.tsv")
    genres = _read_tsv_str(retrieval_dir / "id_genres_mmsr.tsv")
    urls = _read_tsv_str(retrieval_dir / "id_url_mmsr.tsv")

    genres["genre_list"] = genres["genre"].apply(_parse_genre)

    tracks = (
        info
        .merge(genres[["id", "genre_list"]], on="id", how="left")
        .merge(urls, on="id", how="left")
    )

    # ---- Load popularity from metadata if available ----
    meta_path = retrieval_dir / "id_metadata_mmsr.tsv"
    if meta_path.exists():
        # read without dtype=str so popularity parses as numeric
        meta = pd.read_csv(meta_path, sep="\t")
        if "id" in meta.columns and "popularity" in meta.columns:
            meta = meta[["id", "popularity"]].copy()
            meta["id"] = meta["id"].astype(str)
            meta["popularity"] = pd.to_numeric(meta["popularity"], errors="coerce")
            tracks = tracks.merge(meta, on="id", how="left")

    tracks = tracks.drop_duplicates(subset=["id"]).reset_index(drop=True)

    ids = tracks["id"].astype(str).tolist()
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    genre_sets = [set(g) if isinstance(g, list) else set() for g in tracks["genre_list"]]

    popularity = None
    if "popularity" in tracks.columns:
        popularity = tracks["popularity"].to_numpy(dtype=float)

    return Catalog(
        tracks=tracks,
        ids=ids,
        id_to_idx=id_to_idx,
        genres=genre_sets,
        popularity=popularity
        )
