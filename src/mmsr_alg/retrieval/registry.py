from .random_baseline import random_algo
from .unimodal import lyrics_algo, audio_algo, video_algo
from .fusion_late import late_fusion_algo, late_fusion_custom, late_fusion_combo_algo
from .fusion_early import early_fusion_algo, early_fusion_combo_algo
from .system import RetrievalResult
from .cosine import topk_cosine

def get_combined_registry(catalog):

    algos = {
        # ---- baseline / classic ----
        "random": random_algo,
        "lyrics": lyrics_algo,
        "audio": audio_algo,
        "video": video_algo,

        #"early_fusion": early_fusion_algo,
        #"late_fusion": late_fusion_algo,
    }

    # ---- REQUIRED multimodal combinations (4 combos x early/late) ----
    COMBOS = {
        "AT":  ("audio", "text"),
        "AV":  ("audio", "video"),
        "TV":  ("text",  "video"),
        "ATV": ("audio", "text", "video"),
    }

    for key, combo in COMBOS.items():
        algos[f"early_{key}"] = (lambda combo=combo: (lambda cat, qidx, k, seed=None:
            early_fusion_combo_algo(cat, qidx, k, combo=combo)
        ))()

        algos[f"late_{key}"] = (lambda combo=combo: (lambda cat, qidx, k, seed=None:
            late_fusion_combo_algo(cat, qidx, k, combo=combo, normalize=True)
        ))()

    # ---- Neural Network retrieval (STATIC) ----
    def nn_algo(matrix_key):
        def fn(cat, qidx, k, seed=None):
            X = cat.nn_matrices.get(matrix_key)
            if X is None:
                raise ValueError(f"NN matrix '{matrix_key}' not loaded")

            idx, scores = topk_cosine(qidx, X, k)
            return RetrievalResult(
                query_id=cat.ids[qidx],
                algo=matrix_key,
                k=k,
                ranked_ids=[cat.ids[i] for i in idx],
                scores=scores.tolist(),
            )
        return fn

    NN_ALGOS = {
        "lyrics_lyrics": "lyrics_lyrics",
        "audio_audio": "audio_audio",
        "video_lyrics": "video_lyrics",
        "lyrics_audio": "lyrics_audio",
        "video_audio": "video_audio",
    }

    for algo_name, matrix_key in NN_ALGOS.items():
        if matrix_key in catalog.nn_matrices:
            algos[algo_name] = nn_algo(matrix_key)

    # ---- Neural Late Fusion (explicit) ----
    if all(k in catalog.nn_matrices for k in [
        "audio_audio", "video_lyrics", "lyrics_lyrics"
    ]):
        def nn_late_fusion(cat, qidx, k, seed=None):
            return late_fusion_custom(
                cat, qidx, k,
                matrices=[
                    catalog.nn_matrices["audio_audio"],
                    catalog.nn_matrices["video_lyrics"],
                    catalog.nn_matrices["lyrics_lyrics"],
                ],
                weights=[1/3, 1/3, 1/3],
            )

        algos["nn_late_fusion"] = nn_late_fusion

    return algos

ALGORITHMS = {}
