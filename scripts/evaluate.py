
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import ALGORITHMS
from mmsr_alg.retrieval.fusion_early import build_early_fusion_matrix
from mmsr_alg.eval.batch_runner import evaluate_algorithms

DATA = Path("data/retrieval")
OUT  = Path("outputs/results")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_queries", type=int, default=0,
                    help="0 = all queries, else evaluate first N queries (useful for quick tests).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cat = load_catalog(DATA)
    print("Columns in tracks:", list(cat.tracks.columns))
    print("Popularity loaded?", cat.popularity is not None)
    if cat.popularity is not None:
        print("popularity dtype:", cat.popularity.dtype)
        print("pop non-NaN count:", int(np.sum(~np.isnan(cat.popularity))))
        print("pop sample:", cat.popularity[:10])

    # Load + normalize feature matrices
    cat.X_lyrics = l2_normalize(load_feature_matrix(DATA / "id_lyrics_bert_mmsr.tsv", cat.id_to_idx))
    cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))
    cat.X_video  = l2_normalize(load_feature_matrix(DATA / "id_vgg19_mmsr.tsv", cat.id_to_idx))

    # Precompute early fusion
    cat.X_early = build_early_fusion_matrix(cat.X_lyrics, cat.X_audio, cat.X_video, (1/3, 1/3, 1/3))

    system = RetrievalSystem(cat, ALGORITHMS)

    # Query set
    query_ids = cat.ids
    if args.max_queries and args.max_queries > 0:
        query_ids = query_ids[:args.max_queries]

    algos = ["random", "lyrics", "audio", "video", "late_fusion", "early_fusion"]
    k_values = [5, 10, 20, 50, 100, 200]

    df = evaluate_algorithms(
        system=system,
        algos=algos,
        k_values=k_values,
        query_ids=query_ids,
        out_dir=OUT,
        store_lists=True
    )

    # Print a compact view
    print("\nSaved:", (OUT / "metrics.csv"))
    print(df.sort_values(["k", "algo"]).head(20).to_string(index=False))

if __name__ == "__main__":
    main()
