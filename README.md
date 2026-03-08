# EduRecSys – Hybrid Course Recommendation System

EduRecSys is a **hybrid educational recommendation system** designed to simulate
real-world recommendation pipelines used in online learning platforms.

The project combines:
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation Logic
- Cold-Start Handling
- Production-ready API using FastAPI

It emphasizes **system design, scalability, and deployment-readiness**, not just
model accuracy.

---

## Project Overview

This system recommends educational courses based on:
- User behavior (when available)
- Content similarity
- Explicit user intent (subject & level)
- Cold-start fallback strategies

It supports:
- New users (no history)
- Returning users (personalized recommendations)
- Subject-aware hybrid recommendations

---

## Recommendation Strategies

### 1️⃣ Cold Start Recommendation
Used when **no user history is available**.

- Filters content by:
  - Subject
  - Level
- Returns top-K relevant courses
- Prevents empty or irrelevant recommendations

---

### 2️⃣ Content-Based Filtering
- Uses TF-IDF vectorization
- Computes similarity between course descriptions
- Recommends items similar to user-consumed content

---

### 3️⃣ Collaborative Filtering
- Learns patterns from user-item interactions
- Captures implicit user preferences
- Simulates real-world user behavior signals

---

### 4️⃣ Hybrid Recommendation System
Combines:
- Content-based similarity
- Collaborative signals

Improves recommendation quality by leveraging both:
- User behavior
- Content semantics

---

### 5️⃣ Subject-Aware Hybrid (Smart Upgrade)
If a **returning user specifies a subject**:
- Recommendations matching the subject are prioritized
- Falls back to pure hybrid when no strong subject match exists

This balances:
- Explicit user intent
- Implicit behavioral preferences

---

## System Architecture

- Heavy similarity models are **lazy-loaded**
- API starts instantly without blocking
- Large models load only when required

This ensures:
- Faster startup time
- Lower memory usage
- Production stability

---

## Language Support

- Current version supports **English educational content**
- Arabic support is **planned** and requires:
  - Arabic text preprocessing
  - Dedicated vectorization
  - Separate similarity modeling

The system is **language-agnostic by design** and ready for multilingual expansion.

---

## Notebooks (Research & Development)

All experiments and model building are documented through Jupyter notebooks:
```text
notebooks/
├── edurecsys-data-exploration.ipynb
├── edurecsys-data-preprocessing-unified-schema.ipynb
├── edurecsys-content-based-recommendation-system.ipynb
├── edurecsys-collaborative-filtering.ipynb
├── edurecsys-hybrid-recommendation-system.ipynb
├── edurecsys-evaluation-analysis.ipynb
├── edurecsys-deployment.ipynb

```

---

## API (FastAPI)

### Run the API locally
```bash
uvicorn app.main:app --reload
```
Then visit:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs


## Project Structure

```text
EduRecSys/
│
├── app/
│   └── main.py
│
├── data/
│   └── content_en.csv
│
├── models/
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── edurecsys-data-exploration.ipynb
│   ├── edurecsys-data-preprocessing-unified-schema.ipynb
│   ├── edurecsys-content-based-recommendation-system.ipynb
│   ├── edurecsys-collaborative-filtering.ipynb
│   ├── edurecsys-hybrid-recommendation-system.ipynb
│   ├── edurecsys-evaluation-analysis.ipynb
│   └── edurecsys-deployment.ipynb
│
├── README.md
├── requirements.txt
```

## Model Files Note

The file 'content_similarity.pkl' is not included in this repository due to its
large size.

It can be regenerated using the provided notebooks or replaced with any compatible
similarity matrix following the same interface.

This decision reflects real-world production constraints.

## Reproducing Models

To regenerate the similarity models, run:
edurecsys-hybrid-recommendation-system.ipynb


## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Joblib


## Future Enhancements
- Arabic recommendation pipeline
- Multilingual support
- Explainable recommendations
- Frontend integration
