import streamlit as st
import streamlit.components.v1 as components
from loader import load_data, load_genres
from dispatcher import retrieve_and_compute

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

# --- Input panel ---


# --- Input panel con filtri dipendenti ---
with st.container():
    center_cols = st.columns([1, 4, 1])
    with center_cols[1]:
        input_cols = st.columns([1,1,1])

        # --- Dropdown artista ---
        query_artist = input_cols[0].selectbox("Artist (optional)", ["(none)"] + all_artists)

        # --- Filtra dataframe in base all'artista ---
        if query_artist != "(none)":
            filtered_df_artist = df[df["artist"] == query_artist]
        else:
            filtered_df_artist = df

        # --- Dropdown album filtrato dall'artista ---
        available_albums = sorted(filtered_df_artist["album_name"].dropna().unique().tolist())
        query_album = input_cols[2].selectbox("Album (optional)", ["(none)"] + available_albums)

        # --- Filtra dataframe in base all'album selezionato ---
        filtered_df_album = filtered_df_artist
        if query_album != "(none)":
            filtered_df_album = filtered_df_album[filtered_df_album["album_name"] == query_album]

        # --- Dropdown track filtrato da artista + album ---
        available_tracks = sorted(filtered_df_album["song"].dropna().unique().tolist())
        query_track = input_cols[1].selectbox("Track (optional)", ["(none)"] + available_tracks)

        # --- Slider e algoritmi ---
        row2 = st.columns([1,2])
        num_results = row2[0].slider("Number of results", 1, 20, 5)
        algorithms = row2[1].multiselect("Select retrieval algorithms", ["Mockup", "Mockup2"], default=["Mockup"])


# --- Run algorithms ---
if algorithms:
    results_and_metrics = retrieve_and_compute(df, genres_dict, query_artist, query_track, query_album, algorithms, num_results)

    tab_objects = st.tabs(algorithms)
    for idx, algo in enumerate(algorithms):
        with tab_objects[idx]:
            output = results_and_metrics[algo]
            results = output["results"]
            metrics = output["metrics"]

            if not results:
                st.error("❌ No matching tracks found.")
                continue

            st.success(f"Found **{len(results)}** tracks using {algo}.")
            metrics_col, results_col = st.columns([1,2])

            # --- Metrics ---
            with metrics_col:
                st.markdown("### Evaluation Metrics")
                for k,v in metrics.items():
                    st.write(f"**{k}:** {v}")

            # --- Tracks cards ---
            with results_col:
                st.markdown("### Retrieved Tracks")
                for r in results:
                    yt = r.get("url","")
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
                        font-weight:400;
                        line-height:1.5;
                        color:#262730;
                    ">
                        <div style="flex:1;">
                            <h4 style="margin:0 0 6px 0;">🎵 {r['track']} (Score: {r['score']})</h4>
                            <p style="margin:4px 0;"><strong>Artist:</strong> {r['artist']}</p>
                            <p style="margin:4px 0;"><strong>Album:</strong> {r['album_name']}</p>
                            <p style="margin:4px 0;"><strong>Genre:</strong> {r['genre']}</p>
                        </div>
                        <div style="max-width:50%;">{video_html}</div>
                    </div>
                    """
                    components.html(card_html, height=260)
