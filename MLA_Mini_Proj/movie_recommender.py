import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")
OMDB_KEY = "52e11bb4"


# -----------------------------
# Helper Functions
# -----------------------------
def safe_literal_eval(x):
    """Safely convert stringified lists/dicts into Python objects."""
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def get_director(crew):
    """Extract director's name from crew data."""
    for m in crew:
        if m.get("job") == "Director":
            return m.get("name")
    return ""


def get_names(obj_list, key='name', top_n=None):
    """Extract top N names from lists like genres, keywords, cast."""
    names = []
    for i, obj in enumerate(obj_list):
        if top_n and i >= top_n:
            break
        val = obj.get(key)
        if val:
            names.append(val.replace(" ", ""))
    return " ".join(names)


# -----------------------------
# Load & Prepare Dataset
# -----------------------------
def load_and_prepare():
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
        df["genres_list"] + " " +
        df["keywords_list"] + " " +
        df["cast_list"] + " " +
        df["director"].str.replace(" ", "")
    )

    df = df.reset_index(drop=True)
    return df


def fetch_movie_data_omdb(title):
    """Fetch poster + IMDb ID using OMDb."""
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_KEY}"
        response = requests.get(url).json()

        poster = None
        imdb_id = None

        if response.get("Poster") and response["Poster"] != "N/A":
            poster = response["Poster"]

        if response.get("imdbID"):
            imdb_id = response["imdbID"]

        return poster, imdb_id

    except Exception as e:
        print("OMDb error:", e)
        return None, None


# -----------------------------
# Load Data NOW
# -----------------------------
print("Loading dataset... Please wait.")
df = load_and_prepare()

# Determine title column
possible_title_cols = ["title", "original_title", "name"]
title_col = None

for col in possible_title_cols:
    if col in df.columns:
        title_col = col
        break

if title_col is None:
    raise Exception("No valid title column found in dataset")


# -----------------------------
# Build similarity matrix
# -----------------------------
count = CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df["soup"])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Fast lookup
indices = pd.Series(df.index, index=df[title_col].str.lower()).drop_duplicates()


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_movies(title, n=5):
    title = title.lower().strip()

    # Exact match
    if title in indices:
        idx = indices[title]
    else:
        matches = df[df[title_col].str.lower().str.contains(title, na=False)]
        if len(matches) == 0:
            return []
        idx = matches.index[0]

    # Similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    results = []
    for i, _ in sim_scores:
        movie_title = df.loc[i, title_col]

        # Fetch poster + imdb ID
        poster_url, imdb_id = fetch_movie_data_omdb(movie_title)

        results.append({
            "title": movie_title,
            "overview": df.loc[i, "overview"],
            "poster": poster_url,
            "imdb": imdb_id
        })

    return results
