
from .random_baseline import random_algo
from .unimodal import lyrics_algo, audio_algo, video_algo
from .fusion_late import late_fusion_algo
from .fusion_early import early_fusion_algo
from .system import RetrievalResult
from .cosine import topk_cosine

def get_combined_registry(catalog):
    # standard algorithms
    algos = {
        "random": random_algo,
        "lyrics": lyrics_algo,
        "audio": audio_algo,
        "video": video_algo,
        "late_fusion": late_fusion_algo,
        "early_fusion": early_fusion_algo,
    }

    # add Neural Network algorithms
    if hasattr(catalog, "nn_matrices"):
        for key in catalog.nn_matrices.keys():
            # create retrieval function for each neural network matrix
            def make_neural_fn(matrix_key):
                def fn(cat, qidx, k, seed=None):
                    X = cat.nn_matrices[matrix_key]
                    idx, scores = topk_cosine(qidx, X, k)
                    return RetrievalResult(
                        query_id=cat.ids[qidx],
                        algo=matrix_key,
                        k=k,
                        ranked_ids=[cat.ids[i] for i in idx],
                        scores=scores.tolist(),
                    )
                return fn
            
            algos[key] = make_neural_fn(key)
            
    return algos


#temporary keep for compatibility
ALGORITHMS = {}