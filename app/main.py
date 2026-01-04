from fastapi import FastAPI, Query
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

# =========================
# App Config
# =========================
app = FastAPI(
    title="EduRecSys Recommendation API",
    description="Hybrid + Subject-Aware Recommendation System",
    version="1.1.0"
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# =========================
# Load Data
# =========================
content_df = pd.read_csv(DATA_DIR / "content_en.csv")

# =========================
# Load Light Model
# =========================
tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

# =========================
# Lazy Loading for Heavy Model
# =========================
content_similarity = None

def get_content_similarity():
    global content_similarity
    if content_similarity is None:
        print("🔄 Loading content_similarity.pkl ...")
        content_similarity = joblib.load(MODELS_DIR / "content_similarity.pkl")
        print("✅ content_similarity loaded")
    return content_similarity

# =========================
# Cold Start Recommendation
# =========================
def recommend_cold_start(
    subject: Optional[str] = None,
    level: Optional[str] = None,
    k: int = 5
):
    df = content_df.copy()

    if subject:
        df = df[df["subject"].str.lower() == subject.lower()]

    if level:
        level_lower = level.lower()
        df["level_match"] = df["level"].str.lower().apply(
            lambda x: 1 if level_lower in x else 0
        )
        df = df.sort_values("level_match", ascending=False)

    if len(df) == 0:
        df = content_df.copy()

    return df.head(k)[
        ["content_id", "title", "subject", "level"]
    ].to_dict(orient="records")

# =========================
# Hybrid Recommendation
# =========================
def recommend_hybrid(user_id: str, k: int = 5):
    similarity_model = get_content_similarity()
    user_index = int(user_id) % similarity_model.shape[0]

    similarity_scores = similarity_model[user_index]
    top_indices = similarity_scores.argsort()[::-1][:k]

    return content_df.iloc[top_indices][
        ["content_id", "title", "subject", "level"]
    ].to_dict(orient="records")

# =========================
# Subject-Aware Filtering (NEW)
# =========================
def filter_by_subject_priority(recommendations, subject, k):
    if not subject:
        return recommendations[:k]

    subject_lower = subject.lower()

    matched = [
        r for r in recommendations
        if subject_lower in r["subject"].lower()
    ]

    not_matched = [
        r for r in recommendations
        if subject_lower not in r["subject"].lower()
    ]

    return (matched + not_matched)[:k]

# =========================
# API Routes
# =========================
@app.get("/")
def root():
    return {"message": "EduRecSys API is running 🚀"}

@app.get("/recommend")
def recommend(
    user_id: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    k: int = 5
):
    # Cold Start
    if user_id is None:
        return {
            "mode": "cold_start",
            "recommendations": recommend_cold_start(
                subject=subject,
                level=level,
                k=k
            )
        }

    # Hybrid
    hybrid_recs = recommend_hybrid(user_id, k * 2)

    # Subject-aware Hybrid
    final_recs = filter_by_subject_priority(
        recommendations=hybrid_recs,
        subject=subject,
        k=k
    )

    return {
        "mode": "hybrid_subject_aware",
        "user_id": user_id,
        "recommendations": final_recs
    }
