from PIL import Image
import streamlit as st
import numpy as np
import joblib
from utils.preprocessing import preprocess_features

# Load model
model = joblib.load("model/svm_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Deteksi Ras Kucing", layout="centered")
st.markdown("<h1 style='text-align:center; color:#6C63FF;'>😺 Deteksi Ras Kucing</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload gambar kucing", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="📷 Gambar Diunggah", use_column_width=True)

    features = preprocess_features(pil_img)
    features_scaled = scaler.transform([features])

    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0].max()
    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"🎉 Ras: {label}")
    st.info(f"🔢 Confidence: {prob*100:.2f}%")
