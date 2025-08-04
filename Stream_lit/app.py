import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF
import joblib
from datetime import datetime
import io
import re

# Load models and vectorizer
model_act = joblib.load("models/logistic_activity_model.pkl")
model_cat = joblib.load("models/rf_category_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
anomaly_model = joblib.load("models/anomaly_detector.pkl")

# Define field extraction logic
def extract_fields(text):
    vendor = "Unknown"
    date = None
    amount = None

    # Amount
    amount_pat = r'(?i)(?:total|amount paid|grand total|balance due|fare|amount|total due|paid)[^\d]*(\d+[.,]?\d*)'
    amount_match = re.search(amount_pat, text)
    if amount_match:
        try:
            amount = float(amount_match.group(1).replace(",", "").strip())
        except:
            amount = None

    # Date
    date_pat = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})|(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})|(\d{1,2}\s+\w+\s+\d{2,4})|(\w+\s+\d{1,2},\s+\d{4})'
    date_match = re.search(date_pat, text)
    if date_match:
        for g in date_match.groups():
            if g:
                try:
                    date = str(datetime.strptime(g.strip(), "%d %b %Y").date())
                except:
                    date = g.strip()
                break

    # Vendor from first 10 lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    vendor_keywords = ["ltd", "limited", "company", "co.", "llc", "shop", "services", "restaurant", "travel", "market"]
    for line in lines[:10]:
        if not re.search(r'\d', line):
            if any(k in line.lower() for k in vendor_keywords):
                vendor = line.strip()
                break

    return vendor, date, amount

# Define OCR logic
def perform_ocr(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        image = Image.open(file)
        image = image.convert("L")
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageOps.invert(image)
        return pytesseract.image_to_string(image)

# Streamlit app
st.set_page_config(page_title="Agentic AI Receipt App", page_icon="📸", layout="centered")

st.title("📸 Agentic AI Expense Submission")
st.markdown("Upload your receipt below and get structured predictions instantly.")

uploaded_file = st.file_uploader("Upload Receipt Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)

    try:
        ocr_text = perform_ocr(uploaded_file)
        vendor, date, amount = extract_fields(ocr_text)
        cleaned_text = ocr_text.replace("\n", " ").strip()

        # Predictions
        X = vectorizer.transform([cleaned_text])
        pred_act = model_act.predict(X)[0]
        pred_cat = model_cat.predict(X)[0]
        conf_act = np.max(model_act.predict_proba(X))
        conf_cat = np.max(model_cat.predict_proba(X))

        anomaly_flag = 0
        if amount:
            anomaly_flag = 1 if anomaly_model.predict(np.array([[amount]]))[0] == -1 else 0

        # Display results
        st.subheader("📊 Receipt Summary")
        st.write(f"**Vendor:** {vendor}")
        st.write(f"**Date:** {date}")
        st.write(f"**Amount:** £{amount if amount else 'N/A'}")

        st.subheader("🤖 Predictions")
        st.success(f"**Activity:** {pred_act} (Confidence: {conf_act:.2f})")
        st.success(f"**Category:** {pred_cat} (Confidence: {conf_cat:.2f})")

        if anomaly_flag:
            st.warning("⚠️ Amount flagged as potential anomaly.")
        else:
            st.info("✅ Amount appears normal.")

        # Save to CSV (optional logging)
        results = {
            "filename": uploaded_file.name,
            "vendor": vendor,
            "date": date,
            "amount": amount,
            "activity": pred_act,
            "activity_confidence": conf_act,
            "category": pred_cat,
            "category_confidence": conf_cat,
            "anomaly": anomaly_flag,
        }

        df = pd.DataFrame([results])
        df.to_csv("classified_receipt_log.csv", mode="a", header=not bool(pd.read_csv("classified_receipt_log.csv", on_bad_lines='skip').shape[0]), index=False)

    except Exception as e:
        st.error("❌ No predictions made. Please try another file.")
        st.exception(e)
