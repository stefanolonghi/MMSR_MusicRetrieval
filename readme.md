
## What the UI imports and calls

### 1) Build the system once at app startup

```python
from pathlib import Path
from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import ALGORITHMS
from mmsr_alg.retrieval.fusion_early import build_early_fusion_matrix

DATA = Path("data/retrieval")

cat = load_catalog(DATA)
cat.X_lyrics = l2_normalize(load_feature_matrix(DATA / "id_lyrics_bert_mmsr.tsv", cat.id_to_idx))
cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))
cat.X_video  = l2_normalize(load_feature_matrix(DATA / "id_vgg19_mmsr.tsv", cat.id_to_idx))
cat.X_early  = build_early_fusion_matrix(cat.X_lyrics, cat.X_audio, cat.X_video, (1/3, 1/3, 1/3))

retrieval_system = RetrievalSystem(cat, ALGORITHMS)
```

The UI does this **once** (global singleton).

---

### 2) On user query, call `retrieve(query_id, k, algo)`

```python
res = retrieval_system.retrieve(
    query_id="01rMxQv6vhyE1oQX",
    k=10,
    algo="late_fusion"
)

# res.ranked_ids -> list[str] of retrieved track ids
# res.scores     -> optional list[float]
```

That’s the only algorithm call the UI needs.

---

## How the UI gets metadata for display

`Catalog` already contains the merged track table (`cat.tracks`) and cached genres. The UI can “decorate” results like this:

```python
from mmsr_alg.utils import decorate_result

query_card = decorate_result(cat, [res.query_id])[0]
retrieved_cards = decorate_result(cat, res.ranked_ids)
```

Each card has:

* artist, song, album_name
* genres
* url (YouTube)

So the UI can render:

* query track header (artist/song/genres + YouTube link)
* list of retrieved tracks with metadata + YouTube links
* plus optional metric display (next section)

---

## How the UI shows “accuracy” for the retrieved list

For the selected query + algorithm output:

```python
from mmsr_alg.eval.runner import evaluate_one_query

metrics = evaluate_one_query(cat, res.query_id, res.ranked_ids, k=10)
# metrics: {'precision@10': ..., 'recall@10': ..., 'mrr@10': ..., 'ndcg@10': ...}
```

UI displays those numbers next to the results list.

---

## Minimal interface summary 

**Inputs from UI**

* `query_id` (selected by typing artist/song/album in their UI search)
* `k` (slider/dropdown)
* `algo` (dropdown)

**Outputs from your module**

* `RetrievalResult`:

  * `ranked_ids` (for results)
  * `scores` (optional)
* plus `decorate_result()` for metadata
* plus `evaluate_one_query()` for accuracy metrics
##  Core return type from the algorithms

algorithms already return this dataclass:

```python
RetrievalResult(
    query_id: str,
    algo: str,
    k: int,
    ranked_ids: List[str],
    scores: Optional[List[float]]
)
```

### Raw example (algorithm output)

```python
RetrievalResult(
    query_id="01rMxQv6vhyE1oQX",
    algo="late_fusion",
    k=5,
    ranked_ids=[
        "ReikSabesID",
        "ParamoreBrokenID",
        "FMStaticMomentID",
        "DeltaGoodremID",
        "EnriqueMaybeID"
    ],
    scores=[0.91, 0.88, 0.85, 0.81, 0.79]
)
```


---

##  UI-facing format (what the UI actually consumes)

The UI should receive a **fully decorated dictionary** like this:

###  Final UI payload (recommended)

```json
{
  "query": {
    "id": "01rMxQv6vhyE1oQX",
    "artist": "Against the Current",
    "song": "Chasing Ghosts",
    "album": "In Our Bones",
    "genres": ["rock", "pop punk"],
    "youtube_url": "https://www.youtube.com/watch?v=XXXX"
  },
  "algorithm": "late_fusion",
  "k": 5,
  "results": [
    {
      "rank": 1,
      "id": "ReikSabesID",
      "artist": "Reik",
      "song": "Sabes",
      "album": "Des/Amor",
      "genres": ["rock", "latin pop"],
      "youtube_url": "https://www.youtube.com/watch?v=ephwPrwaY20",
      "score": 0.91,
      "relevant": true
    },
    {
      "rank": 2,
      "id": "ParamoreBrokenID",
      "artist": "Paramore",
      "song": "We Are Broken",
      "album": "Riot!",
      "genres": ["pop punk", "emo"],
      "youtube_url": "https://www.youtube.com/watch?v=Dnqu632aynU",
      "score": 0.88,
      "relevant": true
    }
  ],
  "metrics": {
    "precision@5": 0.6,
    "recall@5": 0.0015,
    "mrr@5": 1.0,
    "ndcg@5": 0.62
  }
}
```


* everything pre-joined
* no extra lookups
* easy to render

---

## Adapter function YOU should provide (important)

Create **one function** that converts algorithm output → UI payload.

### `src/mmsr_alg/ui_adapter.py`

```python
from typing import Dict, Any
from .utils import decorate_result
from .eval.runner import evaluate_one_query

def retrieve_for_ui(system, catalog, query_id: str, k: int, algo: str) -> Dict[str, Any]:
    res = system.retrieve(query_id=query_id, k=k, algo=algo)

    query_meta = decorate_result(catalog, [query_id])[0]
    results_meta = decorate_result(catalog, res.ranked_ids)

    metrics = evaluate_one_query(catalog, query_id, res.ranked_ids, k)

    ui_results = []
    for i, item in enumerate(results_meta):
        ui_results.append({
            "rank": i + 1,
            "id": item["id"],
            "artist": item["artist"],
            "song": item["song"],
            "album": item["album_name"],
            "genres": item["genres"],
            "youtube_url": item["url"],
            "score": None if res.scores is None else res.scores[i],
            "relevant": (
                bool(
                    set(query_meta["genres"])
                    & set(item["genres"])
                )
                if query_meta["genres"] and item["genres"]
                else False
            )
        })

    return {
        "query": query_meta,
        "algorithm": algo,
        "k": k,
        "results": ui_results,
        "metrics": metrics
    }
```

---

##  What the UI calls

```python
payload = retrieve_for_ui(system, cat, query_id, k=10, algo="early_fusion")
```

The UI **never touches**:

* feature matrices
* similarity
* evaluation logic

---
