from .system import RetrievalResult
from .cosine import topk_cosine

def _cosine_algo(name: str, attr: str):
    def fn(catalog, qidx, k, seed=None):
        X = getattr(catalog, attr)
        idx, scores = topk_cosine(qidx, X, k)
        return RetrievalResult(
            query_id=catalog.ids[qidx],
            algo=name,
            k=k,
            ranked_ids=[catalog.ids[i] for i in idx],
            scores=scores.tolist()
        )
    return fn

lyrics_algo = _cosine_algo("lyrics", "X_lyrics")
audio_algo  = _cosine_algo("audio",  "X_audio")
video_algo  = _cosine_algo("video",  "X_video")
