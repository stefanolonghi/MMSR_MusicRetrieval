import streamlit as st
import streamlit.components.v1 as components
from loader import load_data, load_genres
from feature_utils import load_and_normalize_split 

# --- app startup ---
from pathlib import Path
from mmsr_alg.io import load_catalog
from mmsr_alg.features import load_feature_matrix, l2_normalize
from mmsr_alg.retrieval.system import RetrievalSystem
from mmsr_alg.retrieval.registry import ALGORITHMS
from mmsr_alg.retrieval.fusion_early import build_early_fusion_matrix
from mmsr_alg.utils import decorate_result
from mmsr_alg.eval.runner import evaluate_one_query

DATA = Path("data/retrieval")

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

    cat.X_early  = build_early_fusion_matrix(cat.X_lyrics, cat.X_audio, cat.X_video, (1/3, 1/3, 1/3))

    retrieval_system = RetrievalSystem(cat, ALGORITHMS)
    return cat, retrieval_system

cat, retrieval_system = init_catalog_and_system()


# --- CSS Font Awesome ---
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">""", unsafe_allow_html=True)

# --- Load data ---
df = load_data()
genres_dict = load_genres()

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
        available_algorithms = list(ALGORITHMS.keys())
        algorithms = row2[1].multiselect("Select retrieval algorithms", available_algorithms, default=["random"])

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

        if not ui_results:
            st.error("❌ No matching tracks found.")
            continue

        st.success(f"Found **{len(ui_results)}** tracks using `{algo}`.")
        metrics_col, results_col = st.columns([1, 2])

        # --- Metrics ---
        with metrics_col:
            st.markdown("### Evaluation Metrics")
            for k, v in metrics.items():
                st.write(f"**{k}:** {v}")

        # --- Results ---
        with results_col:
            st.markdown("### Retrieved Tracks")
            for r in ui_results:
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
                        <h4 style="margin:0 0 6px 0;">🎵 {r['track']} (Score: {r['score']})</h4>
                        <p><strong>Artist:</strong> {r['artist']}</p>
                        <p><strong>Album:</strong> {r['album_name']}</p>
                        <p><strong>Genre:</strong> {r['genre']}</p>
                    </div>
                    <div style="max-width:50%;">{video_html}</div>
                </div>
                """
                components.html(card_html, height=260)