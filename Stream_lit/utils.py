import pytesseract
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import joblib
import os
import fitz  # PyMuPDF
from dateutil import parser

# Load models and vectorizer from /models folder
model_act = joblib.load("models/logistic_activity_model.pkl")
model_cat = joblib.load("models/rf_category_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
anomaly_model = joblib.load("models/anomaly_detector.pkl")

def extract_text_from_image(image: Image.Image) -> str:
    try:
        gray = image.convert('L')
        sharpened = gray.filter(ImageFilter.SHARPEN)
        inverted = ImageOps.invert(sharpened)
        return pytesseract.image_to_string(inverted)
    except Exception as e:
        print("OCR error:", e)
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print("PDF OCR error:", e)
        return ""

def extract_fields(text: str) -> dict:
    import re
    amount_labels = ["total", "amount paid", "grand total", "balance due", "fare", "amount", "total due", "paid"]
    amount_pat = r"(?i)(?:" + "|".join(amount_labels) + r")[^\d]*(\d+[.,]?\d*)"

    # Amount
    amount_match = re.search(amount_pat, text)
    amount = float(amount_match.group(1).replace(',', '').replace(' ', '')) if amount_match else None

    # Date
    try:
        date = parser.parse(text, fuzzy=True).date()
    except:
        date = None

    # Company
    lines = text.strip().split("\n")
    company = lines[0].strip() if lines else "Unknown"

    return {
        "company": company,
        "date": str(date) if date else None,
        "amount": amount,
        "raw_text": text
    }

def run_ocr_and_predict(uploaded_file) -> dict:
    try:
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        text = ""

        if file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            text = extract_text_from_image(image)
        elif file_type == "pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = extract_text_from_pdf("temp.pdf")
            os.remove("temp.pdf")

        fields = extract_fields(text)

        if not fields['amount'] or not fields['raw_text']:
            return {"error": "Missing required information from receipt."}

        # Vectorize text
        text_input = vectorizer.transform([fields['raw_text']])

        # Predictions
        act = model_act.predict(text_input)[0]
        cat = model_cat.predict(text_input)[0]
        conf_act = model_act.predict_proba(text_input).max()
        conf_cat = model_cat.predict_proba(text_input).max()

        # Anomaly detection
        amount_array = np.array([[fields['amount']]])
        anomaly = int(anomaly_model.predict(amount_array)[0] == -1)

        return {
            **fields,
            "predicted_activity": act,
            "confidence_activity": round(conf_act, 3),
            "predicted_category": cat,
            "confidence_category": round(conf_cat, 3),
            "anomaly": anomaly
        }

    except Exception as e:
        print("Prediction error:", e)
        return {"error": "Failed to extract or classify the receipt."}
