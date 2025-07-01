import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os
from utils.preprocessing import preprocess_features

# Load model
model = joblib.load("model/svm_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

# UI setup
st.set_page_config(page_title="Deteksi Ras Kucing", layout="centered")
st.markdown("<h1 style='text-align:center; color:#6C63FF;'>😺 Deteksi Ras Kucing</h1>", unsafe_allow_html=True)
st.markdown("Upload gambar kucing (.jpg/.png) untuk mengenali jenis rasnya.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload gambar kucing", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="📷 Gambar yang Diunggah", use_container_width=True)

    with st.spinner("🔍 Mendeteksi..."):
        features = preprocess_features(pil_img)
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0].max()
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"🎉 Prediksi Ras: **{label}**")
        st.info(f"📊 Tingkat Keyakinan: **{prob * 100:.2f}%**")

        if prob < 0.5:
            st.warning("⚠️ Model kurang yakin. Coba gambar lain yang lebih jelas atau sudut berbeda.")

# Tampilkan confusion matrix image
conf_matrix_path = "confusion_matrix.png"
if os.path.exists(conf_matrix_path):
    st.markdown("---")
    st.markdown("📊 <h4>Confusion Matrix Hasil Evaluasi Model</h4>", unsafe_allow_html=True)
    st.image(conf_matrix_path, use_container_width=True)
else:
    st.info("Confusion matrix belum tersedia. Jalankan `model_training.py` untuk membuatnya.")
