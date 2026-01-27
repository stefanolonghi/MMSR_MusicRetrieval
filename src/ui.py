import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from loader import load_data, load_genres, precompute_all_system_metrics
from feature_utils import load_and_normalize_split 

# --- app startup ---
from pathlib import Path
from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.fusion_late import late_fusion_custom
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import ALGORITHMS
from mmsr_alg.retrieval.fusion_early import build_early_fusion_from_blocks
from mmsr_alg.utils import decorate_result
from mmsr_alg.eval.runner import evaluate_one_query

HERE = Path(__file__).parent
DATA = HERE/"data/retrieval"
METRICS_BEYOND_FILE = HERE/"data/metrics_beyond.tsv"

@st.cache_resource
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

cat, retrieval_system = init_catalog_and_system()
ALGORITHMS = retrieval_system.algorithms

# --- CSS Font Awesome ---
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">""", unsafe_allow_html=True)

# --- Load data ---
df = load_data()
genres_dict = load_genres()
 # -- Load precomputed metrics ---
metrics_df = pd.read_csv(METRICS_BEYOND_FILE, sep="\t")

# --- Unique lists for dropdowns ---
all_artists = sorted(df["artist"].dropna().unique().tolist())
all_tracks = sorted(df["song"].dropna().unique().tolist())
all_albums = sorted(df["album_name"].dropna().unique().tolist())

# --- Page config ---
st.set_page_config(page_title="MMSR – Music Retrieval System", layout="wide")
st.markdown("<h1 style='text-align: center;'>MMSR – Music Retrieval System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type or select a query artist, track, or album</p>", unsafe_allow_html=True)

# --- Input panel con filtri dipendenti ---
with st.container():
    center_cols = st.columns([1, 4, 1])
    with center_cols[1]:
        input_cols = st.columns([1,1,1])

        # --- Dropdown artist/album/track ---
        query_artist = input_cols[0].selectbox(
            "Artist (optional)", ["(none)"] + all_artists, key="artist_select"
        )

        # Filtra albums in base all'artista
        filtered_df_artist = df if query_artist == "(none)" else df[df["artist"] == query_artist]
        available_albums = sorted(filtered_df_artist["album_name"].dropna().unique().tolist())
        query_album = input_cols[2].selectbox(
            "Album (optional)", ["(none)"] + available_albums, key="album_select"
        )

        # Filtra tracks in base ad artista+album
        filtered_df_album = filtered_df_artist if query_album == "(none)" else filtered_df_artist[filtered_df_artist["album_name"] == query_album]
        available_tracks = sorted(filtered_df_album["song"].dropna().unique().tolist())
        query_track = input_cols[1].selectbox(
            "Track", ["(none)"] + available_tracks, key="track_select"
        )

        # --- Slider e algoritmi ---
        row2 = st.columns([1,2])
        num_results = row2[0].slider("Number of results", 1, 20, 5)
        # old: available_algorithms = list(ALGORITHMS.keys())
        available_algorithms = available_algorithms = sorted(list(retrieval_system.algorithms.keys()))

        algorithms = row2[1].multiselect("Select retrieval algorithms", available_algorithms)

# --- Run algorithms ---
if query_track == "(none)":
    st.warning("⚠️ Please select a track to run the retrieval.")
    st.stop()
    
if not algorithms:
    st.warning("⚠️ Please select at least one retrieval algorithm.")
    st.stop() 

if algorithms:
    matches = cat.tracks[cat.tracks["song"] == query_track]
    
    query_id = matches.iloc[0]["id"]
    
    if query_id not in cat.id_to_idx:
        st.error("❌ Selected track not found in MMSR catalog.")
        st.stop()

    
    results_by_algo = {}

    for algo in algorithms:
        results_by_algo[algo] = retrieval_system.retrieve(
            query_id=query_id,
            k=num_results,
            algo=algo
        )

# --- Tabs ---
tab_objects = st.tabs(algorithms)

for tab_idx, algo in enumerate(algorithms):
    with tab_objects[tab_idx]:
        output = results_by_algo.get(algo)

        if not output:
            st.error("❌ No results.")
            continue

        # --- Retrieval output ---
        ranked_ids = output.ranked_ids
        scores = output.scores

        # --- Decorate ---
        retrieved_meta = decorate_result(cat, ranked_ids)

        # --- Build UI-ready results ---
        ui_results = []
        for idx, item in enumerate(retrieved_meta):
            ui_results.append({
                "track": item["song"],
                "artist": item["artist"],
                "album_name": item["album_name"],
                "genre": item["genres"],
                "url": item["url"],
                "score": None if scores is None else scores[idx]
            })

        # --- Metrics ---
        metrics = evaluate_one_query(
            cat,
            output.query_id,
            ranked_ids,
            k=num_results
        )
        
        global_metrics_for_tab = metrics_df[
            (metrics_df["Algorithm"] == algo) &
            (metrics_df["QueryID"] == "GLOBAL") &
            (metrics_df["k"] == num_results)
        ]
        global_metrics = dict(zip(global_metrics_for_tab["Metric"], global_metrics_for_tab["Value"]))


        if not ui_results:
            st.error("❌ No matching tracks found.")
            continue

        st.success(f"Found **{len(ui_results)}** tracks using `{algo}`.")
        metrics_col, results_col = st.columns([1, 2])

        # --- Metrics ---
        with metrics_col:
            st.markdown("### Evaluation Metrics")
            for k, v in metrics.items():
                st.write(f"**{k}:** {v:.4f}")

            st.markdown("---")
            st.markdown("### Beyond-Accuracy (Global)")
            if not global_metrics:
                st.write("Global metrics not found for this k.")
            else:
                for k_name, v in global_metrics.items():
                    st.write(f"**{k_name}:** {v:.4f}")


        # --- Results ---
        with results_col:
            st.markdown("### Retrieved Tracks")
            for r in ui_results:
                score_val = r.get("score")
                score_display = f"{score_val:.4f}" if score_val is not None else "N/A"
                yt = r.get("url", "")
                video_html = ""

                if yt and "watch?v=" in yt:
                    video_id = yt.split("watch?v=")[1].split("&")[0]
                    embed_url = f"https://www.youtube.com/embed/{video_id}"
                    video_html = f"""
                    <iframe width="100%" height="200"
                        src="{embed_url}" frameborder="0"
                        style="border-radius:10px;"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen></iframe>
                    """

                card_html = f"""
                <div style="
                    padding:16px;
                    border-radius:12px;
                    border:1px solid #ddd;
                    margin-bottom:20px;
                    background-color:white;
                    display:flex;
                    gap:16px;
                    width: 60%;
                    font-family: 'Source Sans Pro', sans-serif;
                    font-size:16px;
                    line-height:1.5;
                    color:#262730;
                ">
                    <div style="flex:1;">
                        <h4 style="margin:0 0 6px 0;">🎵 {r['track']} (Score: {score_display})</h4>
                        <p><strong>Artist:</strong> {r['artist']}</p>
                        <p><strong>Album:</strong> {r['album_name']}</p>
                        <p><strong>Genre:</strong> {r['genre']}</p>
                    </div>
                    <div style="max-width:50%;">{video_html}</div>
                </div>
                """
                components.html(card_html, height=260)