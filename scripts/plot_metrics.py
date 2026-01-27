import os
import csv
from pathlib import Path
from feature_utils import load_and_normalize_split
from loader import load_data, load_genres
from mmsr_alg.eval.metrics_beyond import coverage_at_k, pop_at_k
from mmsr_alg.eval.runner import evaluate_one_query
from mmsr_alg.features import l2_normalize, load_feature_matrix
from mmsr_alg.retrieval.fusion_early import build_early_fusion_from_blocks
from mmsr_alg.retrieval.registry import get_combined_registry
import matplotlib.pyplot as plt

from mmsr_alg.io import load_catalog
from mmsr_alg.retrieval.system import RetrievalSystem

# =========================
# CONFIG
# =========================
K_VALUES = range(1, 21)
OUTPUT_DIR = "results"

METRICS = [
    "Precision@k",
    "Recall@k",
    "MRR@k",
    "nDCG@k",
    "Coverage@k",
    "Pop@k",
]

HERE = Path(__file__).parent
DATA = Path("src/data/retrieval")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_catalog_and_system():
    cat = load_catalog(DATA)

    #cat.X_lyrics = l2_normalize(load_feature_matrix(DATA / "id_lyrics_bert_mmsr.tsv", cat.id_to_idx))
    cat.X_lyrics = load_and_normalize_split([
        DATA / "id_lyrics_bert_mmsr_part1.tsv",
        DATA / "id_lyrics_bert_mmsr_part2.tsv"
    ], cat.id_to_idx)

    cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))

    #cat.X_video  = l2_normalize(load_feature_matrix(DATA / "id_vgg19_mmsr.tsv", cat.id_to_idx))
    cat.X_video = load_and_normalize_split([
        DATA / "id_vgg19_mmsr_part1.tsv",
        DATA / "id_vgg19_mmsr_part2.tsv",
        DATA / "id_vgg19_mmsr_part3.tsv",
        DATA / "id_vgg19_mmsr_part4.tsv",
        DATA / "id_vgg19_mmsr_part5.tsv"
    ], cat.id_to_idx)

    cat.X_early = build_early_fusion_from_blocks(
        blocks=[cat.X_lyrics, cat.X_audio, cat.X_video],
        weights=(1/3, 1/3, 1/3),
    )



    #neural  network
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

    retrieval_system = RetrievalSystem(cat, all_algos)
    return cat, retrieval_system


def compute_metrics_from_rankings(cat, all_ranked_ids, k):
    """
    Calcola tutte le metriche principali (Precision, Recall, MRR, nDCG)
    a partire dai ranked_ids, usando evaluate_one_query.
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    for qid, ranked_ids in all_ranked_ids.items():
        eval_dict = evaluate_one_query(cat, qid, ranked_ids, k)
        precisions.append(eval_dict[f"precision@{k}"])
        recalls.append(eval_dict[f"recall@{k}"])
        mrrs.append(eval_dict[f"mrr@{k}"])
        ndcgs.append(eval_dict[f"ndcg@{k}"])

    return {
        "Precision@k": float(sum(precisions)/len(precisions)),
        "Recall@k": float(sum(recalls)/len(recalls)),
        "MRR@k": float(sum(mrrs)/len(mrrs)),
        "nDCG@k": float(sum(ndcgs)/len(ndcgs)),
    }



# =========================
# CORE EVALUATION
# =========================
def compute_all_metrics(cat, retrieval_system, algos):
    results = {}
    MAX_K = max(K_VALUES)

    for algo_name in algos:
        print(f"\nEvaluating {algo_name}...")
        results[algo_name] = {m: {} for m in METRICS}

        # Precompute ranking massimo per tutte le query
        all_ranked_ids_full = {}
        failed_queries_full = []

        for qid in cat.id_to_idx.keys():
            try:
                out = retrieval_system.retrieve(query_id=qid, k=MAX_K, algo=algo_name)
                all_ranked_ids_full[qid] = out.ranked_ids
            except Exception as e:
                print(f"  [ERROR] algo={algo_name}, query={qid}, k={MAX_K}: {e}")
                failed_queries_full.append(qid)

        if not all_ranked_ids_full:
            print(f"  [WARNING] No valid rankings for algo {algo_name}, skipping all metrics.")
            for k in K_VALUES:
                for metric in METRICS:
                    results[algo_name][metric][k] = None
            continue

        for k in K_VALUES:
            # Taglia i ranking precomputati
            all_ranked_ids = {qid: ids[:k] for qid, ids in all_ranked_ids_full.items()}

            # Calcola metriche principali
            try:
                metrics_basic = compute_metrics_from_rankings(cat, all_ranked_ids, k)
                for m in ["Precision@k", "Recall@k", "MRR@k", "nDCG@k"]:
                    results[algo_name][m][k] = metrics_basic[m]
            except Exception as e:
                print(f"  [ERROR] computing basic metrics for algo={algo_name}, k={k}: {e}")
                for m in ["Precision@k", "Recall@k", "MRR@k", "nDCG@k"]:
                    results[algo_name][m][k] = None

            # Coverage e Pop
            try:
                results[algo_name]["Coverage@k"][k] = coverage_at_k(all_ranked_ids, k, N=len(cat.id_to_idx))
            except Exception as e:
                print(f"  [ERROR] computing Coverage@k for algo={algo_name}, k={k}: {e}")
                results[algo_name]["Coverage@k"][k] = None

            try:
                results[algo_name]["Pop@k"][k] = pop_at_k(cat, all_ranked_ids, k)
            except Exception as e:
                print(f"  [ERROR] computing Pop@k for algo={algo_name}, k={k}: {e}")
                results[algo_name]["Pop@k"][k] = None

            if failed_queries_full:
                print(f"  [INFO] {len(failed_queries_full)} queries failed for algo={algo_name}")

    return results



# =========================
# SAVE CSV
# =========================
def save_metrics_csv(results, path, k_for_report):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algo",
            "k",
            "precision",
            "recall",
            "mrr",
            "ndcg",
            "coverage",
            "pop",
            "num_queries"
        ])

        for algo, metrics in results.items():
            k = k_for_report

            # metrics
            precision = metrics["Precision@k"].get(k, None)
            recall = metrics["Recall@k"].get(k, None)
            mrr = metrics["MRR@k"].get(k, None)
            ndcg = metrics["nDCG@k"].get(k, None)
            coverage = metrics["Coverage@k"].get(k, None)
            pop = metrics["Pop@k"].get(k, None)

            # assume that Precision@k has None values for invalid queries
            num_queries = sum(1 for v in metrics["Precision@k"].values() if v is not None)

            writer.writerow([
                algo,
                k,
                precision,
                recall,
                mrr,
                ndcg,
                coverage,
                pop,
                num_queries
            ])


    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Metric", "k", "Value"])

        for algo, metrics in results.items():
            for metric, values in metrics.items():
                for k, v in values.items():
                    writer.writerow([algo, metric, k, v])

# =========================
# PLOTTING
# =========================
def plot_metrics(results):
    for metric in METRICS:
        plt.figure()
        for algo in results:
            ks = list(results[algo][metric].keys())
            vals = list(results[algo][metric].values())
            plt.plot(ks, vals, label=algo)

        plt.xlabel("k")
        plt.ylabel(metric)
        plt.title(metric)

        plt.xticks(list(K_VALUES))   # force integer k only
        plt.xlim(min(K_VALUES), max(K_VALUES))

        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}.png"))
        plt.close()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    K = 10 # test for k=10
    cat, retrieval_system = init_catalog_and_system()
    ALGORITHMS = retrieval_system.algorithms
    all_algos = get_combined_registry(cat)
    
    print("X_audio:", cat.X_audio is not None)
    print("X_video:", cat.X_video is not None)
    print("X_lyrics:", cat.X_lyrics is not None)

    results = compute_all_metrics(cat, retrieval_system, all_algos)
    save_metrics_csv(results, os.path.join(OUTPUT_DIR, "metrics.csv"), K)
    plot_metrics(results)

    print("Done. Metrics and plots saved.")