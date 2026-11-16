# views.py
from flask import render_template, request, jsonify, current_app, url_for
from MLA_Mini_Proj import app
from . import movie_recommender
import time

#    Keep only the callable reference (recommender handles lazy-load)
recommend_movies = movie_recommender.recommend_movies

@app.route("/")
def home():
    sample_titles = [
        "(500) Days of Summer",
        "AVP: Alien vs. Predator",
        "Away We Go",
        "Bandits",
    ]

    return render_template(
        "index.html",
        sample_titles=sample_titles,
    )
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    # Accept both POST (form) and GET (link from "More like this")
    if request.method == "POST":
        movie_name = request.form.get("movie", "").strip()
    else:
        movie_name = request.args.get("movie", "").strip()

    if not movie_name:
        return render_template("index.html", error="Please enter a movie name")

    try:
        t0 = time.perf_counter()
        recs = recommend_movies(movie_name, n=8)  # show more on page
        elapsed = time.perf_counter() - t0
        return render_template("results.html", movie=movie_name, recommendations=recs, recommend_time=elapsed)
    except Exception:
        app.logger.exception("Recommendation failed for title %r", movie_name)
        return render_template(
            "results.html",
            movie=movie_name,
            recommendations=[],
            error="We couldn't process that search right now. Try another movie title.",
        ), 503

@app.route("/api/suggest")
def api_suggest():
    """Return up to 10 titles that contain the query (case-insensitive)."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])

    t0 = time.perf_counter()
    movie_recommender._ensure_loaded()
    df = movie_recommender._df
    title_col = movie_recommender._title_col

    # Use the title column from the recommender
    titles = df[title_col].dropna().astype(str)
    q = movie_recommender.normalize_title(q)
    if not q:
        return jsonify([])
    mask = titles.map(movie_recommender.normalize_title).str.contains(q, regex=False, na=False)
    matches = titles[mask].drop_duplicates().head(10).tolist()
    elapsed = time.perf_counter() - t0
    # include timing in response headers for quick diagnostics
    resp = jsonify(matches)
    resp.headers['X-Suggest-Time'] = f"{elapsed:.4f}s"
    return resp


# Optional: return JSON recommendations for programmatic use
@app.route("/api/recommend")
def api_recommend():
    q = request.args.get("movie", "").strip()
    if not q:
        return jsonify([])
    t0 = time.perf_counter()
    try:
        recs = recommend_movies(q, n=8)
        elapsed = time.perf_counter() - t0
        resp = jsonify(recs)
        resp.headers['X-Recommend-Time'] = f"{elapsed:.4f}s"
        return resp
    except Exception:
        app.logger.exception("API recommendation failed for title %r", q)
        return jsonify({"error": "Unable to process that movie search."}), 503


# Debug route to fetch last profiling stats
@app.route("/debug/profile")
def debug_profile():
    profile = movie_recommender.get_profile()
    return jsonify(profile)
