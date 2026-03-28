"""
server.py — Analyso Flask Backend
──────────────────────────────────
Local dev : python server.py  (port 5001)
Production: gunicorn server:app  (Vercel via api/index.py)
"""

import os
import nltk

# ── CPU thread limits (set before importing numpy/sklearn) ──
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = os.environ.get(_k, "1")

# ── NLTK: use /tmp on serverless, home dir locally ──────────
_NLTK_DIR = os.environ.get("NLTK_DATA",
    os.path.join(os.path.expanduser("~"), "nltk_data"))
nltk.data.path.insert(0, _NLTK_DIR)

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    SentimentIntensityAnalyzer()          # probe — raises LookupError if missing
except LookupError:
    print("⬇️  Downloading NLTK vader_lexicon to", _NLTK_DIR)
    nltk.download("vader_lexicon", download_dir=_NLTK_DIR, quiet=True)

from flask import Flask, jsonify
from flask_cors import CORS
from routes import register_routes

# ── App ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../frontend", static_url_path="")

# ── CORS: read allowed origins from env (comma-separated) ───
_raw = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()] or ["*"]
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "alive",
        "message": "Analyso backend is running"
    }), 200

@app.route("/login")
def login_page():
    return app.send_static_file("login.html")

@app.route("/globe.html")
def globe_page():
    return app.send_static_file("globe.html")

@app.route("/robots.txt")
def robots():
    return app.send_static_file("robots.txt")

@app.route("/sitemap.xml")
def sitemap():
    return app.send_static_file("sitemap.xml")

register_routes(app)

# ── Local entry point ────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Backend → http://127.0.0.1:{port}")
    print("📊 Ready to scan stocks!")
    app.run(host="0.0.0.0", port=port, debug=False)