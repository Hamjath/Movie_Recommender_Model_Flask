from flask import Flask
import threading
import logging

app = Flask(__name__)

def _background_load():
    try:
        # import here to avoid circular imports at module import time
        from . import movie_recommender
        app.logger.info("Starting background dataset load")
        movie_recommender._ensure_loaded()
        app.logger.info("Background dataset load complete")
    except Exception:
        app.logger.exception("Background dataset load failed")

# Start loader thread (daemon so it won't block process exit)
threading.Thread(target=_background_load, daemon=True).start()

import MLA_Mini_Proj.views
