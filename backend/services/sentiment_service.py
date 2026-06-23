import numpy as np
from transformers import pipeline

# Load once at startup
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment_score(text):
    """
    Returns a single float:
    Positive sentiment → close to 1.0
    Negative sentiment → close to 0.0
    Empty text        → 0.5 (neutral)
    """
    if not text or text.strip() == "":
        return 0.5  # neutral fallback

    result = sentiment_pipeline(text)[0]
    score = result["score"]  # confidence score

    # Convert to directional score
    if result["label"] == "POSITIVE":
        return round(score, 4)
    else:
        return round(1.0 - score, 4)  # flip for negative
