# MMSR – Multimodal Music Retrieval System (Group G)
**Group components:**  Nishant Sunil Kale, Filip Kitic, Stefano Longhi, Alvaro Montes Sanchez, Maximiliano Paulino Rodriguez

This repository contains the implementation of a content-based music retrieval system developed for the **Multimedia Search and Retrieval** practical project (2025/26). The framework integrates acoustic, textual, and visual signals to capture the multifaceted nature of musical similarity.

*   **Live Application:** [mmsrmusicretrieval-groupg.streamlit.app](https://mmsrmusicretrieval-groupg.streamlit.app/)
*   **Source Code:** [github.com/stefanolonghii/MMSR_MusicRetrieval](https://github.com/stefanolonghii/MMSR_MusicRetrieval)

## Core Features
- **Multimodal Integration:** Unimodal retrieval (Audio, Lyrics, Video), Feature-level Early Fusion, and Decision-level Late Fusion (with Min-Max normalization).
- **Neural Retrieval:** Six variants of Two-Tower neural architectures for cross-modal latent space alignment.
- **Interactive Dashboard:** Side-by-side algorithm comparison with real-time accuracy and beyond-accuracy metrics.
- **Multimedia Integration:** Dynamic YouTube embedding for qualitative audio-visual verification of results.

---

## 1. System Architecture
The system is designed using a **Registry Pattern**, which decouples the retrieval logic from the presentation layer. At startup, the system initializes a global `Catalog` and registers all available algorithms.

### System Initialization
```python
from pathlib import Path
from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import get_combined_registry

DATA = Path("src/data/retrieval")

# 1. Load Metadata and Handcrafted Features
cat = load_catalog(DATA)
cat.X_lyrics = l2_normalize(load_feature_matrix(DATA / "id_lyrics_bert_mmsr.tsv", cat.id_to_idx))
cat.X_audio  = l2_normalize(load_feature_matrix(DATA / "id_mfcc_bow_mmsr.tsv", cat.id_to_idx))
cat.X_video  = l2_normalize(load_feature_matrix(DATA / "id_vgg19_mmsr.tsv", cat.id_to_idx))

# 2. Initialize Neural Embedding Towers
cat.nn_matrices = {
    "lyrics_lyrics": l2_normalize(load_feature_matrix(DATA / "nn_models/.../f2_lyrics_bert.tsv", cat.id_to_idx)),
    "video_audio": l2_normalize(load_feature_matrix(DATA / "nn_models/.../f2_vgg19.tsv", cat.id_to_idx)),
    # ... additional towers
}

# 3. Build Registry and Engine
all_algos = get_combined_registry(cat)
retrieval_system = RetrievalSystem(cat, all_algos)
```
## 2. Retrieval Modalities
The interface allows users to perform queries using various strategies:
### Unimodal: 
Audio (MFCC BoW), Text (BERT), or Video (VGG19) via Cosine Similarity.
### Early Fusion: 
Feature-level concatenation with weighted modality scaling.
### Late Fusion: 
Decision-level aggregation with Min-Max score normalization.
### Neural Models: 
Learned representations from Two-Tower architectures designed to bridge the "semantic gap."

```python
# Execution example used by the UI
result = retrieval_system.retrieve(
    query_id="01rMxQv6vhyE1oQX",
    k=10,
    algo="late_ATV"
)
```

## 3. Evaluation Framework
The system shows performance metrics based on a genre-overlap relevance proxy:
- Accuracy Metrics: Precision@k, Recall@k, MRR, and nDCG.
- Beyond-Accuracy Metrics: Catalog Coverage and Popularity.

## 4. Deployment and Verification
The system is deployed via Streamlit Cloud, establishing a Continuous Deployment (CD) pipeline linked to this GitHub repository. The application dynamically provisions the environment based on the provided requirements.txt.
The interface facilitates human-in-the-loop verification by integrating YouTube iframe embeds for every retrieved track, allowing users to assess the perceptual success of the mathematical similarity models.

## Project Structure
- src/mmsr_alg/retrieval/: Algorithmic implementations (Cosine, Fusion, Neural).
- src/mmsr_alg/eval/: Quantitative evaluation modules.
- src/ui.py: Streamlit dashboard and visual components.
- scripts/: Offline evaluation and plotting utilities.
