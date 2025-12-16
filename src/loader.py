import pandas as pd
import ast
import streamlit as st
from pathlib import Path

@st.cache_data
def load_data():
    HERE = Path(__file__).parent       # cartella dove si trova questo script
    DATA = HERE / "data"               
    df_info = pd.read_csv(DATA / "id_information_mmsr.tsv", sep="\t")
    df_urls = pd.read_csv(DATA / "id_url_mmsr.tsv", sep="\t")
    df_genres = pd.read_csv(DATA / "id_genres_mmsr.tsv", sep="\t")
    df_merged = df_info.merge(df_urls, on="id", how="left").merge(df_genres, on="id", how="left")
    return df_merged

@st.cache_data
def load_genres():
    df_gen = pd.read_csv("data/id_genres_mmsr.tsv", sep="\t")
    genre_col = next((c for c in df_gen.columns if c.lower() in ["genre", "genres", "genre_list"]), None)
    if genre_col is None:
        st.error("❌ Could not find genre column")
        return {}
    def safe_parse(x):
        try: return ast.literal_eval(x) if isinstance(x, str) else []
        except: return []
    df_gen[genre_col] = df_gen[genre_col].apply(safe_parse)
    return dict(zip(df_gen["id"].astype(str), df_gen[genre_col]))