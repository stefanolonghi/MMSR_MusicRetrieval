
from pathlib import Path

from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import ALGORITHMS
from mmsr_alg.utils import decorate_result
from mmsr_alg.eval.runner import evaluate_one_query
from mmsr_alg.retrieval.fusion_early import build_early_fusion_matrix

DATA = Path("data/retrieval")

def main():
    cat = load_catalog(DATA)

    # Load + normalize feature matrices
    cat.X_lyrics = l2_normalize(load_feature_matrix(DATA / "id_lyrics_bert_mmsr.tsv", cat.id_to_idx))
    cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))
    cat.X_video  = l2_normalize(load_feature_matrix(DATA / "id_vgg19_mmsr.tsv", cat.id_to_idx))

    # Precompute early-fusion matrix once (equal weights)
    cat.X_early = build_early_fusion_matrix(cat.X_lyrics, cat.X_audio, cat.X_video, (1/3, 1/3, 1/3))

    sys = RetrievalSystem(cat, ALGORITHMS)

    # pick a query track (0th) – you can change this to any id
    qid = cat.ids[0]
    qmeta = decorate_result(cat, [qid])[0]
    print("\nQUERY:", qmeta["artist"], "—", qmeta["song"], "| genres:", qmeta["genres"], "\n")

    algos = ["random", "lyrics", "audio", "video", "late_fusion", "early_fusion"]
    k = 5

    for algo in algos:
        res = sys.retrieve(qid, k=k, algo=algo)

        print(algo, "→")
        for item in decorate_result(cat, res.ranked_ids):
            top_genres = (item["genres"] or [])[:3]
            print("  -", item["artist"], "—", item["song"], "|", top_genres, "|", item["url"])

        metrics = evaluate_one_query(cat, qid, res.ranked_ids, k=k)
        print("METRICS", algo, metrics)
        print()

if __name__ == "__main__":
    main()

