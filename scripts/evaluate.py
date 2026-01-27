from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import get_combined_registry
from mmsr_alg.retrieval.fusion_early import build_early_fusion_from_blocks
from mmsr_alg.eval.runner import evaluate_one_query, evaluate_algorithm_global
from feature_utils import load_and_normalize_split 

HERE = Path(__file__).parent
DATA = Path("src/data/retrieval")
OUT  = Path("outputs/results")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_queries", type=int, default=0,
                    help="0 = all queries, else evaluate first N queries (useful for quick tests).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- Load catalog ---
    cat = load_catalog(DATA)

    # --- Load + normalize features ---
    cat.X_lyrics = load_and_normalize_split([
            DATA / "id_lyrics_bert_mmsr_part1.tsv",
            DATA / "id_lyrics_bert_mmsr_part2.tsv"
        ], cat.id_to_idx)    
    
    cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))
    
    cat.X_video = load_and_normalize_split([
        DATA / "id_vgg19_mmsr_part1.tsv",
        DATA / "id_vgg19_mmsr_part2.tsv",
        DATA / "id_vgg19_mmsr_part3.tsv",
        DATA / "id_vgg19_mmsr_part4.tsv",
        DATA / "id_vgg19_mmsr_part5.tsv"
    ], cat.id_to_idx)

    # --- Early fusion ---
    cat.X_early = build_early_fusion_from_blocks(
        blocks=[cat.X_lyrics, cat.X_audio, cat.X_video],
        weights=(1/3, 1/3, 1/3),
    )

    cat.nn_matrices = {}

    cat.nn_matrices["lyrics_lyrics"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/lyrics_bert_lyrics_bert_padding/f1_lyrics_bert_f2_lyrics_bert_padding/f2_lyrics_bert.tsv",
            cat.id_to_idx
        )
    )

    cat.nn_matrices["video_audio"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/vgg19_mfcc_bow_padding/f1_vgg19_f2_mfcc_bow_padding/f2_vgg19.tsv",
            cat.id_to_idx
        )
    )

    cat.nn_matrices["lyrics_audio"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/lyrics_bert_mfcc_bow_padding/f1_lyrics_bert_f2_mfcc_bow_padding/f2_lyrics_bert.tsv",
            cat.id_to_idx
        )
    )

    cat.nn_matrices["video_lyrics"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/vgg19_lyrics_bert_padding/f1_vgg19_f2_lyrics_bert_padding/f2_vgg19.tsv",
            cat.id_to_idx
        )
    )

    cat.nn_matrices["video_video"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/vgg19_vgg19_padding/f1_vgg19_f2_vgg19_padding/f2_vgg19.tsv",
            cat.id_to_idx
        )
    )

    cat.nn_matrices["audio_audio"] = l2_normalize(
        load_feature_matrix(
            DATA / "nn_models/mfcc_bow_mfcc_bow_padding/f1_mfcc_bow_f2_mfcc_bow_padding/f2_mfcc_bow.tsv",
            cat.id_to_idx
        )
    )

    # import function that joins simple altigorithms and neural networks algorithms
    from mmsr_alg.retrieval.registry import get_combined_registry

    all_algos = get_combined_registry(cat)
    system = RetrievalSystem(cat, all_algos)

    # --- Queries ---
    query_ids = cat.ids
    if args.max_queries > 0:
        query_ids = query_ids[:args.max_queries]

    # --- k values ---
    k_values = list(range(1, 21))  # 1..20

    # --- Storage for metrics ---
    rows = []

    print("Precomputing per-query metrics...")
    for algo in tqdm(all_algos):
        for qid in tqdm(query_ids, leave=False):
            # retrieve top max(k_values) once
            max_k = max(k_values)
            res = system.retrieve(query_id=qid, k=max_k, algo=algo)
            ranked_ids = res.ranked_ids

            # compute all metrics for each k
            for k in k_values:
                metrics = evaluate_one_query(cat, qid, ranked_ids, k)
                for metric_name, value in metrics.items():
                    rows.append({
                        "QueryID": qid,
                        "Algorithm": algo,
                        "Metric": metric_name,
                        "k": k,
                        "Value": value
                    })

    # --- Global metrics ---
    print("Precomputing global metrics (Coverage@k, Pop@k)...")
    for algo in tqdm(all_algos):
        for k in k_values:
            global_metrics = evaluate_algorithm_global(cat, system, algo, k)
            for metric_name, value in global_metrics.items():
                rows.append({
                    "QueryID": "GLOBAL",
                    "Algorithm": algo,
                    "Metric": metric_name,
                    "k": k,
                    "Value": value
                })

    # --- Save everything in one TSV ---
    out_file = OUT / "metrics_all.tsv"
    df = pd.DataFrame(rows)
    df.to_csv(out_file, sep="\t", index=False)
    print(f"✅ Saved all metrics to {out_file}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
