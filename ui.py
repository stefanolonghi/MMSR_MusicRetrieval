import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)


# ===============================
# Load dataset
# ===============================
@st.cache_data
def load_data():
    # Main metadata file
    df_info = pd.read_csv("id_information_mmsr.tsv", sep="\t")

    # YouTube URLs file
    df_urls = pd.read_csv("id_url_mmsr.tsv", sep="\t")

    df_genres = pd.read_csv("id_genres_mmsr.tsv", sep="\t")

    # Merge based on common ID column
    # Use the correct shared key "id"
    df_merged = df_info.merge(df_urls, on="id", how="left")
    df_merged = df_merged.merge(df_genres, on="id", how="left")
    
    return df_merged

@st.cache_data
def load_genres():
    df_gen = pd.read_csv("id_genres_mmsr.tsv", sep="\t")

    # Detect correct column name automatically
    genre_col = None
    for col in df_gen.columns:
        if col.lower() in ["genre", "genres", "genre_list"]:
            genre_col = col
    if genre_col is None:
        st.error("❌ Could not find genre column in id_genres_mmsr.tsv")
        return {}

    # Convert Python-list-like strings safely
    def safe_parse(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return []

    df_gen[genre_col] = df_gen[genre_col].apply(safe_parse)

    return dict(zip(df_gen["id"], df_gen[genre_col]))

genres_dict = load_genres()

df = load_data()

# Extract unique lists
all_artists = sorted(df["artist"].dropna().unique().tolist())
all_tracks = sorted(df["song"].dropna().unique().tolist())
all_albums = sorted(df["album_name"].dropna().unique().tolist())
all_genres = sorted(df["genre"].dropna().unique().tolist())

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="MMSR – Music Retrieval System", layout="wide")

# ===============================
# Centered Query Section
# ===============================
st.markdown("<h1 style='text-align: center;'>MMSR – Music Retrieval System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type or select a query artist, track, or album</p>", unsafe_allow_html=True)

with st.container():
    center_cols = st.columns([1, 4, 1])
    with center_cols[1]:

        # --- Inputs in a single row ---
        input_cols = st.columns([1, 1, 1])
        query_artist = input_cols[0].selectbox("Artist (optional)", ["(none)"] + all_artists)
        query_track = input_cols[1].selectbox("Track (optional)", ["(none)"] + all_tracks)
        query_album = input_cols[2].selectbox("Album (optional)", ["(none)"] + all_albums)

        # --- Slider + Algorithm selection on same row ---
        row2 = st.columns([1, 2])  # algorithms take double space
        num_results = row2[0].slider("Number of results", min_value=1, max_value=20, value=5)

        algorithms = row2[1].multiselect(
            "Select retrieval algorithms",
            ["Random baseline", "Unimodal", "Multimodal", "Neural-network based"],
            default=["Random baseline"]
        )


# ===============================
# Retrieval function (placeholder)
# ===============================
def retrieve_tracks(query_artist, query_track, query_album, algorithm, n):
    filtered = df.copy()
    if query_artist != "(none)":
        filtered = filtered[filtered["artist"] == query_artist]
    if query_track != "(none)":
        filtered = filtered[filtered["song"] == query_track]
    if query_album != "(none)":
        filtered = filtered[filtered["album_name"] == query_album]

    results = []
    for i, row in filtered.head(n).iterrows():
        track_id = row["id"]
        genre_list = genres_dict.get(track_id, [])

        results.append({
            "artist": row["artist"],
            "track": row["song"],
            "album_name": row["album_name"],
            "genre": ", ".join(genre_list) if genre_list else "Unknown",
            "url": row.get("url", ""),
            "score": 1.0
        })
    return results

# ===============================
# Evaluation metrics (placeholder)
# ===============================
def compute_metrics(results, query_genre):
    metrics = {
        "Accuracy": 1.0,
        "Precision@k": 1.0,
        "Recall@k": 1.0,
        "MRR@k": 1.0,
        "nDCG@k": 1.0,
        "Coverage@k": 0.5,
        "Pop@k": 0.7
    }
    return metrics

# ===============================
# Display results
# ===============================
if not algorithms:
    st.warning("⚠️ Please select at least one retrieval algorithm.")
else:
        tab_objects = st.tabs(algorithms)
        for idx, algo in enumerate(algorithms):
            with tab_objects[idx]:
                results = retrieve_tracks(query_artist, query_track, query_album, algo, num_results)
                
                if not results:
                    st.error("❌ No matching tracks found.")
                else:
                    st.success(f"Found **{len(results)}** tracks using {algo}.")

                    metrics_col, results_col = st.columns([1, 2])
                    with metrics_col:
                        # Get genre from our genre dictionary instead of df (ignore if missing)
                        if query_track != "(none)":
                            try:
                                # find ID of the selected track
                                track_id = df[(df["artist"] == query_artist) & (df["song"] == query_track)]["id"].iloc[0]
                                query_genre = genres_dict.get(track_id, [])
                            except:
                                query_genre = []
                        else:
                            query_genre = []

                        metrics = compute_metrics(results, query_genre)
                        st.markdown("### Evaluation Metrics")
                        for k, v in metrics.items():
                            st.write(f"**{k}:** {v}")

                    with results_col:
                        st.markdown("### Retrieved Tracks")
                        for r in results:
                            yt = r["url"]
                            video_html = ""

                            if yt and isinstance(yt, str) and "watch?v=" in yt:
                                try:
                                    video_id = yt.split("watch?v=")[1].split("&")[0]
                                    embed_url = f"https://www.youtube.com/embed/{video_id}"

                                    video_html = f"""
                                        <iframe width="100%" height="200"
                                        src="{embed_url}"
                                        frameborder="0"
                                        style="border-radius:10px;"
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                        allowfullscreen></iframe>
                                    """
                                except:
                                    video_html = "<p style='color:orange;'>⚠️ Could not embed video.</p>"

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
                                    font-size:16px;        /* stessa dimensione di Streamlit markdown */
                                    font-weight:400;       /* stesso peso */
                                    line-height:1.5;       /* stessa altezza di linea */
                                    color:#262730;         /* stesso colore testo */
                                ">
                                    <!-- Left column with metadata -->
                                    <div style="flex:1;">
                                        <h4 style="margin:0 0 6px 0;">🎵 {r['track']}</h4>

                                        <p style="margin:4px 0;"><strong>Artist:</strong> {r['artist']}</p>
                                        <p style="margin:4px 0;"><strong>Album:</strong> {r['album_name']}</p>
                                        <p style="margin:4px 0;"><strong>Genre:</strong> {r['genre']}</p>
                                        <p style="margin:4px 0;"><strong>Score:</strong> {r['score']}</p>
                                    </div>

                                    <!-- Right column with video -->
                                    <div style="flex:1; max-width: 50%;">
                                        {video_html}
                                    </div>
                                </div>
                            """

                            components.html(card_html, height=250)