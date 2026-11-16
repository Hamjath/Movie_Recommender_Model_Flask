# MovieMatch — Movie Recommendation App

MovieMatch is a Flask-based content recommendation application. Users can search for a movie, view detailed information, and discover related movies using cached movie embeddings and cosine similarity.

**Live demo:** https://movie-recommender-model-flask.onrender.com  
**Repository:** https://github.com/Hamjath/Movie_Recommender_Model_Flask

## Features

- Movie title search with autocomplete suggestions
- Content-based movie recommendations
- Selected movie details, including poster, plot, genre, director, cast, year, and IMDb rating
- Direct navigation to movie pages on IMDb
- Related movie cards with similarity scores
- Explore links for discovering more related movies
- Responsive dark-themed interface
- Cached OMDb responses and precomputed movie embeddings
- Docker and Gunicorn deployment

## How it works

1. Movie metadata is converted into embedding vectors using Sentence Transformers.
2. The selected movie vector is compared with the cached movie vectors using cosine similarity.
3. The most similar movies are selected as recommendations.
4. OMDb supplies posters, IMDb IDs, plot details, cast, director, genre, and ratings.
5. Flask renders the selected movie details and related recommendations using Jinja templates.

The production loading path uses `movie_embeddings.npy` and loads only the movie fields needed by the results page. This avoids parsing the large credits dataset for every request and reduces memory usage on limited hosting plans.

## Technology stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- OMDb API
- Jinja templates
- Bootstrap Icons
- Docker
- Gunicorn
- Render

## Project structure

```text
MLA_Mini_Proj/
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── static/
│   ├── content/site.css
│   └── scripts/autocomplete.js
├── templates/
│   ├── layout.html
│   ├── index.html
│   └── results.html
├── movie_embeddings.npy
├── movie_recommender.py
├── views.py
└── __init__.py
```

## Run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the OMDb API key

Create an API key at https://www.omdbapi.com/apikey.aspx and set it as an environment variable.

PowerShell:

```powershell
$env:OMDB_KEY = "your-omdb-api-key"
```

Command Prompt:

```cmd
set OMDB_KEY=your-omdb-api-key
```

Do not commit API keys to GitHub.

### 3. Start the application

```bash
python runserver.py
```

Open http://localhost:7860 in a browser.

## API endpoints

- `GET /` — MovieMatch homepage
- `GET|POST /recommend` — Movie details and related recommendations
- `GET /api/suggest?q=batman` — Up to ten matching movie titles
- `GET /api/recommend?movie=Batman` — JSON recommendation results

## Deployment

The project includes a `Dockerfile` and can be deployed as a Docker web service.

The production container runs Gunicorn and listens on port `7860`. For Render deployment:

1. Connect the GitHub repository to a Render Web Service.
2. Select the Docker runtime.
3. Add `OMDB_KEY` as a secret environment variable.
4. Deploy the `main` branch.

Free hosting plans may have limited memory and may sleep when inactive. The cached embedding and lightweight loading path help reduce startup memory usage.

## Data and external services

The recommendation dataset is based on TMDB movie and credits data. Poster images and additional movie details are retrieved from OMDb. The project is not affiliated with or sponsored by TMDB, OMDb, or IMDb.
