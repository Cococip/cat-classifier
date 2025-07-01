import streamlit as st
from PIL import Image
import numpy as np
import joblib
from utils.preprocessing import preprocess_features

st.set_page_config(page_title="Deteksi Ras Kucing", layout="centered")

# Load model & tools
model = joblib.load("model/svm_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

# UI
st.markdown("""
<div style='text-align:center'>
  <h1 style='color:#6C63FF;'>😺 Deteksi Ras Kucing</h1>
  <p>Upload gambar kucing (.jpg/.png) untuk mengenali jenis rasnya.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload gambar kucing", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="📷 Gambar yang Diunggah", use_container_width=True)

    with st.spinner("🔍 Mendeteksi..."):
        features = preprocess_features(pil_img)
        features_scaled = scaler.transform([features])

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba([features_scaled])[0].max()
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"🎉 Prediksi Ras: **{label}**")
        st.info(f"📊 Tingkat Keyakinan: **{prob*100:.2f}%**")

        if prob < 0.5:
            st.warning("⚠️ Model kurang yakin. Coba gambar lain yang lebih jelas atau posisi berbeda.")
