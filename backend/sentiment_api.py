"""
sentiment_api.py — 3D Globe News Sentiment Engine
═══════════════════════════════════════════════════
ADDITIVE MODULE: This file is brand new and does NOT
touch any existing backend logic, databases, or signals.

Flow: RSS Feeds → VADER Sentiment → JSON for Globe.gl
"""

import time
import hashlib
from datetime import datetime

# ── NLTK VADER (already loaded by server.py) ──────────────────
from nltk.sentiment.vader import SentimentIntensityAnalyzer
_sia = SentimentIntensityAnalyzer()

# ── Indian Financial Hub Coordinates ──────────────────────────
INDIAN_HUBS = {
    "Mumbai":    {"lat": 19.076,  "lng": 72.878,  "label": "Mumbai — BSE/NSE HQ"},
    "Delhi":     {"lat": 28.614,  "lng": 77.209,  "label": "Delhi — Policy Hub"},
    "Bengaluru": {"lat": 12.972,  "lng": 77.595,  "label": "Bengaluru — Tech/IT"},
    "Chennai":   {"lat": 13.083,  "lng": 80.271,  "label": "Chennai — Industrial"},
    "Hyderabad": {"lat": 17.385,  "lng": 78.487,  "label": "Hyderabad — Pharma/IT"},
    "Kolkata":   {"lat": 22.573,  "lng": 88.364,  "label": "Kolkata — Eastern Trade"},
}

# ── City-keyword map: route headlines to the right hub ────────
CITY_KEYWORDS = {
    "Mumbai":    ["bse", "nse", "sensex", "nifty", "rbi", "sebi", "mumbai", "reliance", "tata", "hdfc", "icici", "bajaj"],
    "Delhi":     ["delhi", "government", "policy", "budget", "modi", "parliament", "rbi", "tax", "gst", "fiscal"],
    "Bengaluru": ["bengaluru", "bangalore", "infosys", "wipro", "tech", "startup", "it sector", "software"],
    "Chennai":   ["chennai", "auto", "automobile", "ashok leyland", "tvs", "titan", "manufacturing"],
    "Hyderabad": ["hyderabad", "pharma", "biotech", "dr reddy", "cipla", "healthcare", "generic"],
    "Kolkata":   ["kolkata", "coal", "itc", "eastern", "calcutta", "bandhan", "emami"],
}

# ── RSS Feed Sources ──────────────────────────────────────────
RSS_FEEDS = [
    {"url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "source": "Economic Times"},
    {"url": "https://www.moneycontrol.com/rss/MCtopnews.xml", "source": "MoneyControl"},
    {"url": "https://www.livemint.com/rss/markets", "source": "Livemint"},
]

# ── In-memory cache (no database needed) ─────────────────────
_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 300  # 5 minutes


def _fetch_rss_headlines():
    """Scrape headlines from Indian financial RSS feeds."""
    import feedparser

    headlines = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:8]:  # Top 8 per source
                title = entry.get("title", "").strip()
                link = entry.get("link", "#")
                pub_date = entry.get("published", "")

                if title:
                    headlines.append({
                        "title": title,
                        "link": link,
                        "source": feed_info["source"],
                        "published": pub_date,
                    })
        except Exception as e:
            print(f"⚠️ RSS fetch error ({feed_info['source']}): {e}")
            continue

    return headlines


def _analyze_sentiment(text):
    """Run VADER sentiment analysis on a headline. Returns score + label."""
    scores = _sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.15:
        return {"score": round(compound, 3), "label": "Bullish", "color": "#22c55e"}
    elif compound <= -0.15:
        return {"score": round(compound, 3), "label": "Bearish", "color": "#ef4444"}
    else:
        return {"score": round(compound, 3), "label": "Neutral", "color": "#f59e0b"}


def _assign_city(headline_text):
    """Map a headline to an Indian financial hub based on keywords."""
    text_lower = headline_text.lower()
    for city, keywords in CITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return city
    # Default: Mumbai (financial capital, catches general market news)
    return "Mumbai"


def get_globe_data():
    """
    Main entry point. Returns JSON-ready data for the 3D Globe.

    Response format:
    {
        "timestamp": "2026-03-14T20:00:00",
        "hubs": [...],
        "news_points": [...],
        "summary": { "bullish": N, "bearish": N, "neutral": N }
    }
    """
    now = time.time()

    # Return cached data if fresh
    if _cache["data"] and (now - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]

    # 1. Fetch headlines from RSS
    headlines = _fetch_rss_headlines()

    # 2. Analyze sentiment + assign cities
    news_points = []
    summary = {"bullish": 0, "bearish": 0, "neutral": 0}

    for h in headlines:
        sentiment = _analyze_sentiment(h["title"])
        city = _assign_city(h["title"])
        hub = INDIAN_HUBS[city]

        # Unique ID for deduplication
        uid = hashlib.md5(h["title"].encode()).hexdigest()[:8]

        news_points.append({
            "id": uid,
            "title": h["title"],
            "source": h["source"],
            "link": h["link"],
            "published": h["published"],
            "city": city,
            "lat": hub["lat"] + (hash(uid) % 100 - 50) * 0.005,  # Slight jitter
            "lng": hub["lng"] + (hash(uid) % 100 - 50) * 0.005,
            "sentiment": sentiment,
        })

        # Count
        key = sentiment["label"].lower()
        summary[key] = summary.get(key, 0) + 1

    # 3. Build hub points
    hubs = []
    for name, coords in INDIAN_HUBS.items():
        # Aggregate sentiment for this hub
        hub_news = [n for n in news_points if n["city"] == name]
        if hub_news:
            avg_score = sum(n["sentiment"]["score"] for n in hub_news) / len(hub_news)
            if avg_score >= 0.1:
                hub_color = "#22c55e"
            elif avg_score <= -0.1:
                hub_color = "#ef4444"
            else:
                hub_color = "#f59e0b"
        else:
            avg_score = 0
            hub_color = "#64748b"

        hubs.append({
            "name": name,
            "label": coords["label"],
            "lat": coords["lat"],
            "lng": coords["lng"],
            "color": hub_color,
            "news_count": len(hub_news),
            "avg_sentiment": round(avg_score, 3),
        })

    # 4. Build response
    response = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hubs": hubs,
        "news_points": news_points,
        "summary": summary,
        "total_headlines": len(news_points),
    }

    # 5. Cache it
    _cache["data"] = response
    _cache["timestamp"] = now

    return response
