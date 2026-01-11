import ast
import os
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Can be a local path or an HTTP(S) URL. Set these as env vars on Render.
MOVIES_CSV = os.environ.get("MOVIES_CSV", os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv"))
CREDITS_CSV = os.environ.get("CREDITS_CSV", os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv"))
OMDB_KEY = os.environ.get("OMDB_KEY", "52e11bb4")

# Caches for lazy loading
_df = None
_cosine_sim = None
_indices = None
_title_col = None

def _is_url(path):
    return isinstance(path, str) and path.startswith(("http://", "https://"))

def _download_to_temp(url):
    """Download URL to a temp file and return the local path. Reuse if exists."""
    tmpdir = tempfile.gettempdir()
    fname = os.path.basename(url.split("?")[0]) or "data.csv"
    local_path = os.path.join(tmpdir, fname)
    if os.path.exists(local_path):
        return local_path
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_path

def _resolve_path(source):
    """If source is a URL, download and return local path, else return source as-is."""
    if _is_url(source):
        return _download_to_temp(source)
    return source

def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def get_director(crew):
    for m in crew:
        if m.get("job") == "Director":
            return m.get("name")
    return ""

def get_names(obj_list, key='name', top_n=None):
    names = []
    for i, obj in enumerate(obj_list):
        if top_n and i >= top_n:
            break
        val = obj.get(key)
        if val:
            names.append(val.replace(" ", ""))
    return " ".join(names)

def _load_and_prepare():
    movies_path = _resolve_path(MOVIES_CSV)
    credits_path = _resolve_path(CREDITS_CSV)

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id")

    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].fillna("[]").apply(safe_literal_eval)

    df["director"] = df["crew"].apply(get_director)
    df["genres_list"] = df["genres"].apply(lambda x: get_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: get_names(x, top_n=15))
    df["cast_list"] = df["cast"].apply(lambda x: get_names(x, top_n=5))
    df["overview"] = df["overview"].fillna("")

    df["soup"] = (
        df["genres_list"] + " " +
        df["keywords_list"] + " " +
        df["cast_list"] + " " +
        df["director"].str.replace(" ", "")
    )

    df = df.reset_index(drop=True)

    possible_title_cols = ["title", "original_title", "name"]
    title_col = None
    for col in possible_title_cols:
        if col in df.columns:
            title_col = col
            break
    if title_col is None:
        raise Exception("No valid title column found in dataset")

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df.index, index=df[title_col].str.lower()).drop_duplicates()

    return df, cosine_sim, indices, title_col

def _ensure_loaded():
    global _df, _cosine_sim, _indices, _title_col
    if _df is None:
        _df, _cosine_sim, _indices, _title_col = _load_and_prepare()

def fetch_movie_data_omdb(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_KEY}"
        response = requests.get(url, timeout=5).json()
        poster = response.get("Poster") if response.get("Poster") and response["Poster"] != "N/A" else None
        imdb_id = response.get("imdbID")
        return poster, imdb_id
    except Exception:
        return None, None

def recommend_movies(title, n=5):
    _ensure_loaded()
    title = title.lower().strip()

    if title in _indices:
        idx = _indices[title]
    else:
        matches = _df[_title_col].str.lower().str.contains(title, na=False)
        matches = _df[matches]
        if len(matches) == 0:
            return []
        idx = matches.index[0]

    sim_scores = list(enumerate(_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    results = []
    for i, _ in sim_scores:
        movie_title = _df.loc[i, _title_col]
        poster_url, imdb_id = fetch_movie_data_omdb(movie_title)
        results.append({
            "title": movie_title,
            "overview": _df.loc[i, "overview"],
            "poster": poster_url,
            "imdb": imdb_id
        })
    return results
