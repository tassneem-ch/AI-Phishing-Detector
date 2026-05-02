from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os
from scipy.sparse import hstack, csr_matrix
from fastapi.middleware.cors import CORSMiddleware

# Import our components
from component2_preprocessing import preprocess_email
from component3_feature_extraction import (
    extract_hand_crafted_features,
    URGENCY_WORDS, THREAT_WORDS, REWARD_WORDS, extract_urls,
    is_suspicious_url, has_sender_mismatch, has_replyto_mismatch,
    count_keyword_hits
)
from component4_classification import predict, THREAT_THRESHOLD

app = FastAPI(title="Phishing Detector API")

# Enable CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the extension ID
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_assets():
    global model, vectorizer
    model_path = "output/models/random_forest.pkl"
    vec_path = "output/models/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("[ERROR] Models not found. Please run 'python main.py' first.")
        return False

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Models loaded successfully.")
    return True

class EmailRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    if not load_assets():
        # We don't exit so the user can see the error in the console
        pass

@app.post("/analyze")
async def analyze_email(request: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Models not loaded on server.")

    raw_text = request.text
    clean_text = preprocess_email(raw_text)

    # 1. Extract hand-crafted features
    urls = extract_urls(raw_text)
    hc_features = {
        "urgency_score":         count_keyword_hits(clean_text, URGENCY_WORDS),
        "threat_score":          count_keyword_hits(clean_text, THREAT_WORDS),
        "reward_score":          count_keyword_hits(clean_text, REWARD_WORDS),
        "link_count":            len(urls),
        "suspicious_link_flag":  int(any(is_suspicious_url(u) for u in urls)),
        "sender_mismatch_flag":  has_sender_mismatch(raw_text),
        "replyto_mismatch_flag": has_replyto_mismatch(raw_text),
    }
    hc_array = np.array(list(hc_features.values()), dtype=np.float32).reshape(1, -1)

    # 2. TF-IDF
    tfidf_array = vectorizer.transform([clean_text])

    # 3. Combine features
    X_input = hstack([csr_matrix(hc_array), tfidf_array])

    # 4. Predict
    proba, pred = predict(model, X_input, THREAT_THRESHOLD)
    threat_score = float(proba[0])
    verdict = "PHISH" if pred[0] == 1 else "HAM"

    print(f"\n[SCAN] Verdict: {verdict} | Threat Score: {threat_score:.2%}")
    print(f"       Indicators: {hc_features}")
    
    return {
        "verdict": verdict,
        "threat_score": round(threat_score, 4),
        "indicators": hc_features,
        "clean_text_snippet": clean_text[:200]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
