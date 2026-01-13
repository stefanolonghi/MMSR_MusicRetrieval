
from .random_baseline import random_algo
from .unimodal import lyrics_algo, audio_algo, video_algo
from .fusion_late import late_fusion_algo, late_fusion_custom
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
    
    # 2. new: NEURAL LATE FUSION
        # select f1 branches of refined models (autoencoders)
        smart_audio = catalog.nn_matrices.get("mfcc_bow_mfcc_bow_f1")
        smart_video = catalog.nn_matrices.get("vgg19_vgg19_f1")
        smart_lyrics = catalog.nn_matrices.get("lyrics_bert_lyrics_bert_f1")

        if all(m is not None for m in [smart_audio, smart_video, smart_lyrics]):
            def neural_late_fusion_fn(cat, qidx, k, seed=None):
                return late_fusion_custom(
                    cat, qidx, k, 
                    matrices=[smart_audio, smart_video, smart_lyrics],
                    weights=[1/3, 1/3, 1/3]
                )
            algos["neural_late_fusion"] = neural_late_fusion_fn
            
    return algos


#temporary keep for compatibility
ALGORITHMS = {}