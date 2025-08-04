import streamlit as st
import requests
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --------------------------------------------
# Load your trained models
model_act = joblib.load("model_activity.pkl")
model_cat = joblib.load("model_category.pkl")
vectorizer_act = joblib.load("vectorizer_activity.pkl")
vectorizer_cat = joblib.load("vectorizer_category.pkl")

# --------------------------------------------
# Your OCR.space API key
OCR_API_KEY = "K89937681088957"

# --------------------------------------------
st.title("📄 Expense Receipt Classifier (Image Upload with OCR.space)")

uploaded_file = st.file_uploader("Upload your receipt image (JPG, PNG)")

if uploaded_file:
    # Send to OCR.space API
    files = {'file': (uploaded_file.name, uploaded_file, "multipart/form-data")}
    payload = {
        'apikey': OCR_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False
    }
    response = requests.post('https://api.ocr.space/parse/image', files=files, data=payload)

    if response.status_code == 200:
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            st.error("OCR.space API error: " + result.get("ErrorMessage", ["Unknown error"])[0])
        else:
            extracted_text = result["ParsedResults"][0]["ParsedText"]

            st.subheader("Extracted Text (OCR):")
            st.write(extracted_text)

            # Vendor guess (first non-empty line)
            vendor = extracted_text.split("\n")[0][:50] if extracted_text else "Unknown"

            # Improved amount extraction logic
            amount_numbers = re.findall(r'[\£\$€]?\s*\d{1,3}(?:[.,]\d{2})', extracted_text)

            amount_values = []
            for amt in amount_numbers:
                amt_clean = amt.replace("£", "").replace("$", "").replace("€", "").replace(" ", "").replace(",", "")
                try:
                    amount_values.append(float(amt_clean))
                except:
                    continue

            if amount_values:
                amount = max(amount_values)
            else:
                amount = 0.0

            combined_text = vendor + " receipt"

            # Vectorize
            X_act = vectorizer_act.transform([combined_text])
            X_cat = vectorizer_cat.transform([combined_text])

            # Predict
            pred_act = model_act.predict(X_act)[0]
            pred_cat = model_cat.predict(X_cat)[0]

            conf_act = model_act.predict_proba(X_act).max()
            conf_cat = model_cat.predict_proba(X_cat).max()

            # Show predictions
            st.subheader("Predicted Classification")
            st.write(f"**Predicted Activity:** {pred_act} (Confidence: {conf_act:.2f})")
            st.write(f"**Predicted Expense Category:** {pred_cat} (Confidence: {conf_cat:.2f})")
            st.write(f"**Amount:** £{amount:.2f}")

            # Human review flag
            needs_review = conf_act < 0.6 or conf_cat < 0.6
            if needs_review:
                st.warning("⚠️ This receipt needs human review (low confidence)")
            else:
                st.success("✅ High confidence prediction")

            # User feedback
            approve = st.radio("Do you approve this classification?", ("Yes", "No"))

            if approve == "Yes":
                st.success("Classification confirmed ✅")
            else:
                st.error("Please flag for manual review or adjust in backend.")

            # Save button
            if st.button("Save Result to File"):
                result_df = pd.DataFrame([{
                    "vendor": vendor,
                    "amount": amount,
                    "predicted_activity": pred_act,
                    "activity_confidence": conf_act,
                    "predicted_category": pred_cat,
                    "category_confidence": conf_cat,
                    "needs_human_review": needs_review
                }])
                result_df.to_csv("classified_receipt.csv", index=False)
                st.success("Result saved to classified_receipt.csv ✅")
    else:
        st.error("OCR.space API call failed. Please check your API key or try again.")
