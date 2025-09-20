# app.py
import os
import re
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from transformers import pipeline

# Optional: make logs less noisy on Render
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finbert-app")

# ------------------------------------------------------------------------------
# Flask + DB config
# ------------------------------------------------------------------------------
app = Flask(__name__)

# Render sometimes gives DATABASE_URL starting with postgres:// (old style).
db_url = os.getenv("DATABASE_URL", "sqlite:///sentiment.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class Article(db.Model):
    __tablename__ = "article"
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50), nullable=False)            # e.g., GoogleNews, Reddit
    instrument = db.Column(db.String(100), nullable=False)       # search term
    headline = db.Column(db.Text, nullable=False)
    url = db.Column(db.String(500), unique=True)                 # avoid duplicates
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # One-to-one (one sentiment per article)
    sentiments = db.relationship("SentimentScore", backref="article", lazy=True, uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "instrument": self.instrument,
            "headline": self.headline,
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
            "sentiment": self.sentiments.sentiment_label if self.sentiments else None
        }

class SentimentScore(db.Model):
    __tablename__ = "sentiment_score"
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    sentiment_label = db.Column(db.String(20), nullable=False)   # positive|negative|neutral
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# ------------------------------------------------------------------------------
# External deps (FinBERT, Google News, Reddit)
# ------------------------------------------------------------------------------
logger.info("Loading FinBERT model (ProsusAI/finbert)...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    logger.info("FinBERT loaded.")
except Exception as e:
    logger.exception("Failed to load FinBERT. Falling back to neutral outputs.")
    sentiment_pipeline = None

# Reddit creds from env (don’t hardcode secrets)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "python:finbert-sentiment-api:1.0 (by u/yourname)"
)

# Lazy imports so app can boot even if these libs hiccup during cold start
def _lazy_import_news_clients():
    from pygooglenews import GoogleNews
    import praw
    return GoogleNews, praw

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def analyze_sentiment(text: str) -> str:
    """Return 'positive' | 'negative' | 'neutral'."""
    try:
        if not text or not isinstance(text, str):
            return "neutral"
        if sentiment_pipeline is None:
            return "neutral"
        # FinBERT is trained on finance; cap length for speed
        result = sentiment_pipeline(text[:512])[0]
        label = str(result.get("label", "")).lower()
        # Some transformers return uppercase; normalize
        if label in {"positive", "negative", "neutral"}:
            return label
        # Map any unexpected labels safely
        if "pos" in label:
            return "positive"
        if "neg" in label:
            return "negative"
        return "neutral"
    except Exception:
        return "neutral"

def fetch_and_analyze(security_name: str, force_refresh: bool = False):
    """
    Fetch articles (Google News + Reddit), store in DB, analyze sentiment,
    and return a list of dicts ready for aggregation.
    """
    logger.info(f"Request: '{security_name}', refresh={force_refresh}")

    # 1) Use cached (recent) articles if not forcing refresh
    if not force_refresh:
        cutoff = datetime.utcnow() - timedelta(minutes=30)
        recent = Article.query.filter(
            Article.instrument.ilike(f"%{security_name}%"),
            Article.fetched_at > cutoff
        ).all()

        if recent:
            # Ensure sentiments exist
            to_analyze = [a for a in recent if not a.sentiments]
            for art in to_analyze:
                label = analyze_sentiment(art.headline)
                db.session.add(SentimentScore(article_id=art.id, sentiment_label=label))
            if to_analyze:
                db.session.commit()

            return [a.to_dict() for a in recent]

    # 2) Fetch fresh from sources
    articles_created = []
    all_items = []

    GoogleNews, praw = _lazy_import_news_clients()
    # A) Google News
    try:
        gn = GoogleNews(lang="en", country="US")
        resp = gn.search(f"{security_name} stock", when="7d")  # last 7 days
        for entry in (resp or {}).get("entries", []):
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", None)
            if not url or not title:
                continue
            # Skip if exists
            if Article.query.filter_by(url=url).first():
                continue

            art = Article(
                source="GoogleNews",
                instrument=security_name,
                headline=title,
                url=url
            )
            db.session.add(art)
            try:
                db.session.commit()
                articles_created.append(art)
                all_items.append({"source": "GoogleNews", "headline": title, "url": url, "instrument": security_name})
            except IntegrityError:
                db.session.rollback()
                # Duplicate raced in; ignore
    except Exception as e:
        logger.warning(f"Google News fetch failed: {e}")

    # B) Reddit (optional — only if creds provided)
    try:
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            subreddit = reddit.subreddit("investing+stocks+wallstreetbets")
            for submission in subreddit.search(security_name, sort="new", limit=30):
                url = getattr(submission, "url", None)
                title = getattr(submission, "title", None)
                if not url or not title:
                    continue
                if Article.query.filter_by(url=url).first():
                    continue

                art = Article(
                    source="Reddit",
                    instrument=security_name,
                    headline=title,
                    url=url
                )
                db.session.add(art)
                try:
                    db.session.commit()
                    articles_created.append(art)
                    all_items.append({"source": "Reddit", "headline": title, "url": url, "instrument": security_name})
                except IntegrityError:
                    db.session.rollback()
        else:
            logger.info("Reddit credentials not set; skipping Reddit fetch.")
    except Exception as e:
        logger.warning(f"Reddit fetch failed: {e}")

    # 3) Sentiment for new articles
    if articles_created:
        for art in articles_created:
            label = analyze_sentiment(art.headline)
            db.session.add(SentimentScore(article_id=art.id, sentiment_label=label))
        db.session.commit()

    # 4) Attach sentiment to all_items (using DB rows)
    for item in all_items:
        db_art = Article.query.filter_by(url=item["url"]).first()
        if db_art and db_art.sentiments:
            item["sentiment"] = db_art.sentiments.sentiment_label

    return all_items

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/")
def root():
    return jsonify({"status": "ok", "message": "FinBERT Sentiment API running."})

@app.route("/init_db")
def init_db():
    try:
        db.create_all()
        return jsonify({"message": "Database initialized."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sentiment", methods=["GET"])
def api_sentiment():
    security_name = request.args.get("q")
    if not security_name or not security_name.strip():
        return jsonify({"error": "Provide query parameter ?q=<security name>"}), 400

    force_refresh = request.args.get("refresh", "false").lower() == "true"
    analyzed = fetch_and_analyze(security_name.strip(), force_refresh=force_refresh)

    if not analyzed:
        return jsonify({"error": f"No data found for '{security_name}'."}), 404

    # Keep only rows that have a sentiment
    analyzed = [d for d in analyzed if d.get("sentiment")]
    if not analyzed:
        return jsonify({"error": f"Sentiment not available for '{security_name}' yet."}), 500

    df = pd.DataFrame(analyzed)
    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()

    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral", 0)

    # Rating: map [-1..+1] -> [1..5] centered at 3
    if total > 0:
        normalized = (pos - neg) / total
        rating = 2 * normalized + 3
    else:
        rating = 3.0

    if rating >= 4.0:
        label = "Very Positive"
    elif rating >= 3.5:
        label = "Positive"
    elif rating > 2.5:
        label = "Neutral"
    elif rating > 1.5:
        label = "Negative"
    else:
        label = "Very Negative"

    percentages = {
        "positive": round((pos / total) * 100, 1) if total else 0,
        "negative": round((neg / total) * 100, 1) if total else 0,
        "neutral": round((neu / total) * 100, 1) if total else 0,
    }

    # Breakdown by source
    breakdown = {}
    for src in df["source"].unique():
        sdf = df[df["source"] == src]
        sc = sdf["sentiment"].value_counts().to_dict()
        breakdown[src] = {
            "totalMentions": len(sdf),
            "sentimentCounts": {
                "positive": sc.get("positive", 0),
                "negative": sc.get("negative", 0),
                "neutral": sc.get("neutral", 0),
            }
        }

    return jsonify({
        "security": security_name,
        "analysisTimestampUTC": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "sentimentRating": round(float(rating), 2),
            "sentimentLabel": label,
            "totalMentions": int(total),
            "sentimentCounts": {"positive": int(pos), "negative": int(neg), "neutral": int(neu)},
            "sentimentPercentages": percentages
        },
        "sourceBreakdown": breakdown,
        "feed": analyzed[:25],
    })

# ------------------------------------------------------------------------------
# Local run
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # For local dev only; on Render we use gunicorn via Procfile
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
