from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import pandas as pd
import praw
import numpy as np
import re
from collections import Counter

app = Flask(__name__)
CORS(app)

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'ow_part1_box1_script')

# NLTK for sentiment
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

nlp = None
sia = None

def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded")
    return nlp

def get_sentiment_analyzer():
    global sia
    if sia is None:
        print("Loading VADER sentiment analyzer...")
        sia = SentimentIntensityAnalyzer()
        print("VADER loaded")
    return sia

print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False
)
print("Reddit client ready")

MKT_SEO_SL_SUBS: List[str] = [
    "marketing",
    "SEO",
    "digital_marketing",
    "socialmedia",
    "PPC",
]

bad_org = {
    "Reddit", "YouTube", "Instagram", "TikTok",
    "GOP", "Democrats", "Republicans",
}

# ============================================
# INFLUENCE ALGORITHM (Standardized)
# ============================================

def reddit_influence(row) -> float:
    """
    Standardized influence algorithm using logarithmic scaling.
    Formula: log(1 + upvotes) + 0.5 * log(1 + comments)
    """
    score = max(row.get("score", 0) or 0, 0)
    comments = max(row.get("num_comments", 0) or 0, 0)
    return np.log1p(score) + 0.5 * np.log1p(comments)

def compute_negativity(text: str) -> float:
    """Compute negativity score using VADER sentiment."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    analyzer = get_sentiment_analyzer()
    compound = analyzer.polarity_scores(text)["compound"]
    return max(0.0, -compound)

def compute_negative_impact_score(row) -> float:
    """Final score: negativity * influence"""
    return float(row.get("negativity", 0.0)) * float(row.get("influence_score", 0.0))

# ============================================
# DATA COLLECTION
# ============================================

def fetch_hot_posts(
    subreddits: List[str],
    days: int = 7,
    per_sub_limit: int = 200,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: List[Dict[str, Any]] = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)

            for post in subreddit.hot(limit=per_sub_limit):
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                if created < cutoff:
                    break

                rows.append({
                    "subreddit": sub,
                    "post_id": post.id,
                    "created_utc": post.created_utc,
                    "created_dt": created.isoformat(),
                    "title": (post.title or "").strip(),
                    "selftext": (post.selftext or "").strip(),
                    "url": getattr(post, "url", ""),
                    "permalink": f"https://www.reddit.com{getattr(post, 'permalink', '')}",
                    "score": getattr(post, "score", None),
                    "num_comments": getattr(post, "num_comments", None),
                    "author": str(getattr(post, "author", "")) if getattr(post, "author", None) else None,
                    "is_self": getattr(post, "is_self", None),
                    "over_18": getattr(post, "over_18", None),
                })

                if sleep_seconds:
                    time.sleep(sleep_seconds)

        except Exception as e:
            print(f"[WARN] Subreddit '{sub}' failed: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["full_text"] = (
        df["title"].astype(str).str.strip()
        + "\n"
        + df["selftext"].astype(str).str.strip()
    ).str.strip()

    # Deduplicate by post ID (one post per unique ID)
    initial_count = len(df)
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    final_count = len(df)
    
    print(f"Collected {initial_count} posts, {final_count} after dedup")
    
    df = df.sort_values("created_utc", ascending=False).reset_index(drop=True)

    return df


def extract_orgs(text: str) -> List[str]:
    
    if not isinstance(text, str) or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text[:5000])
    orgs: List[str] = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            name = ent.text.strip()
            name = re.sub(r"\s+", " ", name)
            name = name.strip(".,:;()[]{}\"'")

            if len(name) < 2:
                continue
            if name in bad_org:
                continue
            if name.isupper() and len(name) <= 3:
                continue

            orgs.append(name)
    
    return orgs


def collapse_to_parent(org: str, org_set: set) -> str:
    
    s = re.sub(r"\s+", " ", org).strip().strip(".,:;()[]{}\"'")
    s_cf = s.casefold()
    parts = s.split()

    if len(parts) == 1:
        return s_cf

    for k in range(len(parts), 0, -1):
        prefix = " ".join(parts[:k]).casefold()
        if prefix in org_set:
            return prefix

    return parts[0].casefold()


def analyze_companies(days: int = 7, per_sub_limit: int = 200):
    """
    Main analysis function with influence algorithm.
    """
    
    df_mkt = fetch_hot_posts(
        MKT_SEO_SL_SUBS, 
        days=days, 
        per_sub_limit=per_sub_limit, 
        sleep_seconds=0.0
    )
    df_mkt["bucket"] = "marketing"

    if df_mkt.empty:
        return {
            "error": "No posts collected",
            "message": "Try increasing per_sub_limit or add more subreddits"
        }

    # Calculate influence scores
    print("Calculating influence scores...")
    df_mkt["influence_score"] = df_mkt.apply(reddit_influence, axis=1)
    
    # Calculate negativity
    print("Analyzing sentiment...")
    df_mkt["negativity"] = df_mkt["full_text"].apply(compute_negativity)
    
    # Calculate negative impact score
    df_mkt["negative_impact_score"] = df_mkt.apply(compute_negative_impact_score, axis=1)

    # Extract organizations
    print("Extracting organizations...")
    df_mkt["orgs"] = df_mkt["full_text"].apply(extract_orgs)

    exploded = df_mkt.explode("orgs").dropna(subset=["orgs"]).copy()

    if exploded.empty:
        return {
            "error": "No organizations found",
            "message": "No ORG entities found. Try increasing PER_SUB_LIMIT."
        }

    exploded["org_clean"] = (
        exploded["orgs"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.strip(".,:;()[]{}\"'")
    )

    org_set = set(exploded["org_clean"].str.casefold().tolist())

    exploded["company_key"] = exploded["org_clean"].apply(
        lambda org: collapse_to_parent(org, org_set)
    )

    display_map = (
        exploded.assign(company_key=exploded["company_key"])
        .groupby(["company_key", "org_clean"])
        .size()
        .reset_index(name="n")
        .sort_values(["company_key", "n"], ascending=[True, False])
        .drop_duplicates("company_key")
        .set_index("company_key")["org_clean"]
        .to_dict()
    )

    # Aggregate by company using negative_impact_score
    company_stats = (
        exploded.groupby("company_key")
        .agg(
            negative_impact_score=("negative_impact_score", "sum"),
            mentions=("post_id", "count"),
            avg_influence=("influence_score", "mean")
        )
        .sort_values("negative_impact_score", ascending=False)
    )

    top10 = company_stats.head(10).reset_index()
    top10["company"] = top10["company_key"].map(display_map).fillna(top10["company_key"])
    
    print(f"Analysis complete. Found {len(top10)} companies")
    
    return top10


@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Feature 1 - Companies Needing Social Listening",
        "endpoints": {
            "/": "Health check",
            "/analyze": "Run analysis (GET)",
            "/health": "Service health"
        },
        "note": "Using influence algorithm (log scaling) + VADER sentiment"
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 50))  # Lower default for free tier
        
        print(f"Starting analysis (days={days}, limit={limit})")
        
        result = analyze_companies(days=days, per_sub_limit=limit)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        ranked_companies = []
        for idx, row in result.iterrows():
            ranked_companies.append({
                "rank": int(idx + 1),
                "company": {
                    "id": f"cmp_{row['company'].lower().replace(' ', '_')}",
                    "name": row['company']
                },
                "volume": {
                    "mentions": int(row['mentions']),
                    "negative_impact_score": round(float(row['negative_impact_score']), 2),
                    "avg_influence": round(float(row['avg_influence']), 2)
                }
            })
        
        response = {
            "data": {
                "ranked_companies": ranked_companies
            },
            "meta": {
                "total": len(ranked_companies),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range_days": days,
                "posts_per_subreddit": limit,
                "scoring": "influence algorithm (log scaling) + VADER sentiment"
            }
        }
        
        print(f"Analysis complete. Returning {len(ranked_companies)} companies")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "tip": "Try reducing the 'limit' parameter"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
