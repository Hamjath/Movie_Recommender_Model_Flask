# views.py
from flask import render_template, request, jsonify, current_app, url_for
from MLA_Mini_Proj import app
from . import movie_recommender

# Convenience references to the recommender's df and helpers
df = movie_recommender.df
recommend_movies = movie_recommender.recommend_movies
title_col = movie_recommender.title_col

@app.route("/")
def home():
    # sample titles for dropdown (kept small and stable order)
    sample_titles = df[title_col].dropna().sample(30, random_state=42).sort_values().tolist()
    return render_template("index.html", sample_titles=sample_titles)


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    # Accept both POST (form) and GET (link from "More like this")
    if request.method == "POST":
        movie_name = request.form.get("movie", "").strip()
    else:
        movie_name = request.args.get("movie", "").strip()

    if not movie_name:
        return render_template("index.html", error="Please enter a movie name")

    recs = recommend_movies(movie_name, n=8)  # show more on page
    return render_template("results.html", movie=movie_name, recommendations=recs)


@app.route("/api/suggest")
def api_suggest():
    """Return up to 10 titles that contain the query (case-insensitive)."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])

    # Use the title column from the recommender
    titles = df[title_col].dropna().astype(str)
    mask = titles.str.lower().str.contains(q, na=False)
    matches = titles[mask].drop_duplicates().head(10).tolist()
    return jsonify(matches)


# Optional: return JSON recommendations for programmatic use
@app.route("/api/recommend")
def api_recommend():
    q = request.args.get("movie", "").strip()
    if not q:
        return jsonify([])
    recs = recommend_movies(q, n=8)
    return jsonify(recs)
