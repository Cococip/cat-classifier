import streamlit as st
import numpy as np
import joblib
from PIL import Image
from utils.preprocessing import preprocess_image_pil

st.set_page_config(page_title="Deteksi Ras Kucing", layout="centered")

model = joblib.load("model/svm_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.markdown("""
<div style='text-align: center'>
    <h1 style='color:#6C63FF;'>😺 Deteksi Ras Kucing</h1>
    <p>Upload gambar kucing (.jpg/.png) untuk mengenali jenis rasnya.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload gambar kucing", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="📷 Gambar yang Diunggah", use_column_width=True)

    with st.spinner("🔍 Mendeteksi..."):
        features = preprocess_image_pil(pil_image)
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0].max()
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"🎉 Prediksi Ras: **{label}**")
        st.info(f"📊 Tingkat Keyakinan: **{prob*100:.2f}%**")
