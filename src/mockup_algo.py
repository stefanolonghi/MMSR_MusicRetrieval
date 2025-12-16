### This algorithm is just for testing and acts as a search over the dataset.

def run(df, genres_dict, query_artist, query_track, query_album, n):
    # Filters all data based on selected values
    filtered = df.copy()
    if query_artist != "(none)":
        filtered = filtered[filtered["artist"] == query_artist]
    if query_track != "(none)":
        filtered = filtered[filtered["song"] == query_track]
    if query_album != "(none)":
        filtered = filtered[filtered["album_name"] == query_album]

    # Keep only first n results
    filtered = filtered.head(n)

    results = []
    for _, row in filtered.iterrows():
        track_id = str(row["id"])
        genre_list = genres_dict.get(track_id, [])

        results.append({
            "track": row["song"],
            "artist": row["artist"],
            "album_name": row["album_name"],
            "genre": ", ".join(genre_list) if genre_list else "Unknown",
            "score": 1.0,
            "url": row.get("url", "")
        })

    # Default return value
    if not results:
        results = [{
            "track": "No match found",
            "artist": "",
            "album_name": "",
            "genre": "",
            "score": 0.0,
            "url": ""
        }]

    metrics = {
        "Accuracy": 1.0,
        "Precision@k": 1.0,
        "Recall@k": 1.0,
        "MRR@k": 1.0,
        "nDCG@k": 1.0,
        "Coverage@k": 0.5,
        "Pop@k": 0.7
    }

    return {"results": results, "metrics": metrics}