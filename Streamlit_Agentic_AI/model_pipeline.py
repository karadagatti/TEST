import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    with open(path, 'rb') as f:
        return pickle.load(f)

activity_model = load_model('logistic_activity_model.pkl')
category_model = load_model('rf_category_model.pkl')
tfidf_vectorizer = load_model('tfidf_vectorizer.pkl')
anomaly_detector = load_model('anomaly_detector.pkl')

def predict_expense(texts):
    results = []

    for text in texts:
        clean_text = text.replace("\n", " ").replace("\r", " ").strip()

        # Transform text
        X = tfidf_vectorizer.transform([clean_text])

        # Predictions
        activity = activity_model.predict(X)[0]
        activity_conf = activity_model.predict_proba(X).max()

        category = category_model.predict(X)[0]
        category_conf = category_model.predict_proba(X).max()

        # Anomaly detection
        amount = extract_amount(text)
        amount_arr = np.array([[amount if amount else 0]])
        anomaly = int(anomaly_detector.predict(amount_arr)[0] == -1)

        results.append({
            "Raw Text": clean_text,
            "Predicted Activity": activity,
            "Activity Confidence": round(activity_conf, 2),
            "Predicted Category": category,
            "Category Confidence": round(category_conf, 2),
            "Amount": amount,
            "Anomaly": anomaly,
            "Needs Review": (activity_conf < 0.7 or category_conf < 0.7 or anomaly == 1)
        })

    return pd.DataFrame(results)

def extract_amount(text):
    import re
    amount_pat = r"(?i)(?:total|amount paid|grand total|balance due|fare|amount|total due|paid)[^\d]*(\d+[.,]?\d*)"
    match = re.search(amount_pat, text)
    if match:
        try:
            return float(match.group(1).replace(',', '').replace(' ', ''))
        except:
            return None
    return None
