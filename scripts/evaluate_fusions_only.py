from __future__ import annotations
from pathlib import Path
import argparse

from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import get_combined_registry
from mmsr_alg.eval.batch_runner import evaluate_algorithms

DATA = Path("src/data/retrieval")
OUT  = Path("outputs/fusions_only")  # <- separate output folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_queries", type=int, default=0,
                    help="0 = all queries, else evaluate first N queries (quick test).")
    args = ap.parse_args()

    # Load catalog (tracks/ids/genres/popularity)
    cat = load_catalog(DATA)

    # Load + normalize raw feature matrices
    import numpy as np  # add at top

    # lyrics bert = part1 + part2
    X_lyrics_p1 = load_feature_matrix(DATA / "id_lyrics_bert_mmsr_part1.tsv", cat.id_to_idx)
    X_lyrics_p2 = load_feature_matrix(DATA / "id_lyrics_bert_mmsr_part2.tsv", cat.id_to_idx)
    cat.X_lyrics = l2_normalize(np.concatenate([X_lyrics_p1, X_lyrics_p2], axis=1))

    # mfcc bow = single file
    cat.X_audio = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))

    # vgg19 = part1..part5
    X_vgg = []
    for i in range(1, 6):
        X_vgg.append(load_feature_matrix(DATA / f"id_vgg19_mmsr_part{i}.tsv", cat.id_to_idx))
    cat.X_video = l2_normalize(np.concatenate(X_vgg, axis=1))


    # Build registry (this must contain early_AT, late_AT, etc.)
    registry = get_combined_registry(cat)
    system = RetrievalSystem(cat, registry)

    # Queries
    query_ids = cat.ids
    if args.max_queries and args.max_queries > 0:
        query_ids = query_ids[:args.max_queries]

    # ONLY the 8 required fusion systems
    algos = [
        "early_AT", "late_AT",
        "early_AV", "late_AV",
        "early_TV", "late_TV",
        "early_ATV", "late_ATV",
    ]

    # Safety: keep only those actually registered (prevents crashing if a name is missing)
    missing = [a for a in algos if a not in registry]
    if missing:
        raise RuntimeError(
            "These fusion algorithms are missing from the registry:\n"
            + "\n".join(missing)
            + "\n\nFix registry.py registration first."
        )

    # Default cutoff required by course
    k_values = [10]
    # Optional sweep (if you want later):
    # k_values = [1, 5, 10, 100]

    df = evaluate_algorithms(
        system=system,
        algos=algos,
        k_values=k_values,
        query_ids=query_ids,
        out_dir=OUT,
        store_lists=False,  # set True only if you want to debug retrieved lists
    )

    print("\nSaved:", (OUT / "metrics.csv"))
    print(df.sort_values(["k", "algo"]).to_string(index=False))


if __name__ == "__main__":
    main()
