import ast
import os
import re
import requests
import pandas as pd
import numpy as np
import tempfile
import threading
import time
import tracemalloc
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_CSV = os.environ.get(
    "MOVIES_CSV",
    os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
)

CREDITS_CSV = os.environ.get(
    "CREDITS_CSV",
    os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")
)

OMDB_KEY = os.environ.get("OMDB_KEY", "52e11bb4")

EMBEDDING_FILE = os.path.join(BASE_DIR, "movie_embeddings.npy")

# lazy caches
_df = None
_indices = None
_title_col = None
_model = None
_embeddings = None
_load_lock = threading.Lock()

_profile = {}

logger = logging.getLogger(__name__)


def normalize_title(value):
    """Normalize user-entered titles for forgiving, punctuation-safe matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def safe_literal_eval(x):
    try:
        value = ast.literal_eval(x) if isinstance(x, str) else x
        return value if isinstance(value, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def get_director(crew):
    for m in crew:
        if m.get("job") == "Director":
            return m.get("name")
    return ""


def get_names(obj_list, key="name", top_n=None):
    names = []

    for i, obj in enumerate(obj_list):
        if top_n and i >= top_n:
            break

        val = obj.get(key)

        if val:
            names.append(val.replace(" ", ""))

    return " ".join(names)


def _load_and_prepare():

    global _model, _embeddings

    t0 = time.perf_counter()

    logger.info("Loading datasets...")

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

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
        df["genres_list"]
        + " "
        + df["keywords_list"]
        + " "
        + df["cast_list"]
        + " "
        + df["director"].str.replace(" ", "")
    )

    df = df.reset_index(drop=True)

    possible_title_cols = ["title", "original_title", "name"]

    title_col = None

    for col in possible_title_cols:
        if col in df.columns:
            title_col = col
            break

    if title_col is None:
        raise Exception("No valid title column found")

    logger.info("Loading transformer model...")

    _model  = None

    def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

    # EMBEDDING CACHE
    if os.path.exists(EMBEDDING_FILE):

        logger.info("Loading cached embeddings...")

        _embeddings = np.load(EMBEDDING_FILE)

    else:

        logger.info("Generating embeddings...")

        _embeddings = get_model().encode(
            df["soup"].tolist(),
            show_progress_bar=True
        )

        np.save(EMBEDDING_FILE, _embeddings)

        logger.info("Embeddings saved to disk")

    indices = pd.Series(
        df.index,
        index=df[title_col].map(normalize_title)
    )

    elapsed = time.perf_counter() - t0

    logger.info("Dataset ready in %.2f seconds", elapsed)

    return df, indices, title_col


def _ensure_loaded():

    global _df, _indices, _title_col

    if _df is None:
        with _load_lock:
            if _df is None:
                _df, _indices, _title_col = _load_and_prepare()


def fetch_movie_data_omdb(title):

    try:

        response = requests.get(
            "http://www.omdbapi.com/",
            params={"t": title, "apikey": OMDB_KEY},
            timeout=5,
        ).json()

        poster = (
            response.get("Poster")
            if response.get("Poster") and response["Poster"] != "N/A"
            else None
        )

        imdb_id = response.get("imdbID")

        return poster, imdb_id

    except Exception:

        return None, None


fetch_movie_data_omdb = lru_cache(maxsize=512)(fetch_movie_data_omdb)


def recommend_movies(title, n=5):

    _ensure_loaded()

    title = normalize_title(title)

    if not title:
        return []

    if title in _indices:
        matching_index = _indices[title]
        idx = int(matching_index.iloc[0] if isinstance(matching_index, pd.Series) else matching_index)

    else:
        normalized_titles = _df[_title_col].map(normalize_title)
        matches = _df[normalized_titles.str.contains(title, regex=False, na=False)]

        if len(matches) == 0:
            return []

        idx = int(matches.index[0])

    global _embeddings

    query_vec = _embeddings[idx].reshape(1, -1)

    similarities = cosine_similarity(query_vec, _embeddings).flatten()

    similar_indices = similarities.argsort()[::-1]

    result_indices = [nbr for nbr in similar_indices if nbr != idx][:n]
    movie_titles = [_df.loc[nbr, _title_col] for nbr in result_indices]

    # OMDB is optional enrichment. Fetch it concurrently so a slow external
    # service cannot add one timeout per recommendation to the request.
    if movie_titles:
        with ThreadPoolExecutor(max_workers=min(len(movie_titles), 8)) as executor:
            omdb_data = list(executor.map(fetch_movie_data_omdb, movie_titles))
    else:
        omdb_data = []

    results = []
    for nbr, movie_title, (poster_url, imdb_id) in zip(
        result_indices, movie_titles, omdb_data
    ):
        results.append(
            {
                "title": movie_title,
                "overview": _df.loc[nbr, "overview"],
                "poster": poster_url,
                "imdb": imdb_id,
                "score": float(similarities[nbr]),
            }
        )

    return results


def get_profile():

    return dict(_profile)