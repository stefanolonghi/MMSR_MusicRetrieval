import pandas as pd
import ast
from mmsr_alg.eval.runner import evaluate_algorithm_global
import streamlit as st
from pathlib import Path

HERE = Path(__file__).parent  

@st.cache_data
def load_data():
    DATA = HERE / "data"               
    df_info = pd.read_csv(DATA / "id_information_mmsr.tsv", sep="\t")
    df_urls = pd.read_csv(DATA / "id_url_mmsr.tsv", sep="\t")
    df_genres = pd.read_csv(DATA / "id_genres_mmsr.tsv", sep="\t")
    df_merged = df_info.merge(df_urls, on="id", how="left").merge(df_genres, on="id", how="left")
    return df_merged

@st.cache_data
def load_genres():
    DATA = HERE / "data" 
    df_gen = pd.read_csv(DATA / "id_genres_mmsr.tsv", sep="\t")
    genre_col = next((c for c in df_gen.columns if c.lower() in ["genre", "genres", "genre_list"]), None)
    if genre_col is None:
        st.error("❌ Could not find genre column")
        return {}
    def safe_parse(x):
        try: return ast.literal_eval(x) if isinstance(x, str) else []
        except: return []
    df_gen[genre_col] = df_gen[genre_col].apply(safe_parse)
    return dict(zip(df_gen["id"].astype(str), df_gen[genre_col]))

@st.cache_data
def precompute_all_system_metrics(cat, _retrieval_system, _algos, max_k=20):
    """
    Precompute Coverage@k and Pop@k for all algorithms and k=1..max_k.
    
    Returns:
        dict: {algo_name: {k: {"Coverage@k": ..., "Pop@k": ...}}}
    """
    all_metrics = {}

    for algo_name in _algos.keys():
        print(f"Precomputing system metrics for {algo_name}...")
        algo_metrics = {}
        for k in range(1, max_k+1):
            cov, pop = evaluate_algorithm_global(cat, _retrieval_system, algo_name, k)
            algo_metrics[k] = {
                "Coverage@k": cov,
                "Pop@k": pop
            }
        all_metrics[algo_name] = algo_metrics

    return all_metrics