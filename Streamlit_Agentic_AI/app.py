# app.py (Final Functional Streamlit App)
import streamlit as st
import pandas as pd
from datetime import datetime
from ocr_utils import extract_text_from_file
from model_pipeline import run_pipeline_on_text
import tempfile
import os

st.set_page_config(page_title="Agentic AI Receipt Classifier", layout="wide")
st.title("🧾 Agentic AI - Functional Receipt Classifier")

mode = st.radio("Select Mode", ["Automated", "Manual"], horizontal=True)

# --- AUTOMATED MODE ---
if mode == "Automated":
    uploaded_file = st.file_uploader("Upload Receipt (Image only)", type=["jpg", "jpeg", "png"])

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            # OCR extraction
            extracted_text = extract_text_from_file(file_path)

            # Run ML prediction pipeline
            prediction = run_pipeline_on_text(extracted_text)
            prediction['File'] = uploaded_file.name
            results.append(prediction)

            os.unlink(file_path)

        df = pd.DataFrame(results)
        st.success("✅ Processed all receipts!")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Predictions CSV", csv, file_name="predicted_receipts.csv", mime="text/csv")

# --- MANUAL MODE ---
else:
    st.subheader("✍️ Manual Entry")

    company = st.text_input("Company / Vendor Name")
    date = st.date_input("Date of Receipt", value=datetime.today())
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    activity = st.selectbox("Activity", ["General expenses", "Travel", "Social", "Company meeting", "Conference"])
    category = st.selectbox("Expense Category", ["Miscellaneous", "Taxi", "Train/Bus/Coach", "Hotel", "Food", "Parking Fee"])

    if st.button("📤 Generate Manual CSV Entry"):
        manual_df = pd.DataFrame([{
            "Company": company,
            "Date": date.strftime("%Y-%m-%d"),
            "Amount": amount,
            "Predicted Activity": activity,
            "Predicted Category": category,
            "Confidence": "Manual Entry",
            "Needs Review": False
        }])
        st.success("✅ Entry Created")
        st.dataframe(manual_df)

        csv = manual_df.to_csv(index=False)
        st.download_button("Download Manual Entry CSV", csv, file_name="manual_expense_entry.csv", mime="text/csv")
