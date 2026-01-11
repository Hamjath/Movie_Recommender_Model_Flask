# Movie Recommender (Flask)

A lightweight content‑based movie recommender web app built with Flask, pandas and scikit-learn.  
Features lazy dataset loading, a memory‑efficient nearest‑neighbors index, a simple web UI and JSON APIs.

Live demo: https://movie-recommender-model-flask.onrender.com  
Repo: https://github.com/Hamjath/Movie_Recommender_Model_Flask

## Tech stack
- Python 3.11+
- Flask
- pandas, numpy
- scikit-learn (`CountVectorizer`, `NearestNeighbors`)
- gunicorn (production WSGI server)

## Quickstart — local

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python runserver.py
