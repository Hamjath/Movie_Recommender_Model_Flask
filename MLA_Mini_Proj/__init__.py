from flask import Flask
import threading
import logging

app = Flask(__name__)

_loader_started = False

def _background_load():
    try:
        # import here to avoid circular imports at module import time
        from . import movie_recommender
        app.logger.info("Starting background dataset load")
        movie_recommender._ensure_loaded()
        app.logger.info("Background dataset load complete")
    except Exception:
        app.logger.exception("Background dataset load failed")

def start_background_loader():
    global _loader_started
    if not _loader_started:
        _loader_started = True
        threading.Thread(target=_background_load, daemon=True).start()

# Start loader in the worker on the first incoming request (works if before_first_request is unavailable)
@app.before_request
def _start_loader_on_first_request():
    start_background_loader()

import MLA_Mini_Proj.views
