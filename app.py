import streamlit as st
import cv2
import numpy as np
import joblib
import sys
from PIL import Image
from utils.preprocessing import preprocess_image

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Ras Kucing",
    layout="centered"
)

# Load model
try:
    model = joblib.load('model/svm_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan file .pkl ada di folder 'model/'.")
    st.stop()

# Header
st.markdown("""
    <div style='text-align: center'>
        <h1 style='color:#6C63FF;'>😺 Deteksi Ras Kucing</h1>
        <p style='font-size:18px;'>Upload gambar kucing untuk mengenali jenis rasnya.<br>
        Model berbasis <b>SVM</b> + fitur <b>Local Binary Pattern (LBP)</b></p>
    </div>
""", unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("📁 Upload gambar kucing (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Gambar tidak dapat dibaca. Pastikan file valid dan dalam format .jpg/.png.")
        st.stop()

    # Tampilkan gambar
    st.image(image, caption="📷 Gambar yang Diunggah", use_column_width=True, channels="BGR")

    # Prediksi
    with st.spinner("🔍 Mendeteksi ras kucing..."):
        try:
            feature = preprocess_image(image)
            prediction = model.predict([feature])[0]
            prob = model.predict_proba([feature])[0].max()
            label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"🎉 Prediksi Ras: **{label}**")
            st.info(f"📊 Tingkat Keyakinan: **{prob*100:.2f}%**")

            # Info tambahan ras
            ras_info = {
                "Persian": "🧶 Bulu panjang, wajah pesek, sangat jinak.",
                "Bengal": "🐆 Corak tutul seperti macan tutul, aktif dan cerdas.",
                "Sphynx": "🚿 Tidak berbulu, kulit keriput, sangat unik.",
                "Abyssinian": "🐱 Aktif, penasaran, dan elegan.",
                "Maine_coon": "🦁 Ras besar dan bersahabat.",
                # Tambah info ras lain di sini jika perlu
            }

            if label in ras_info:
                st.markdown(f"**ℹ️ Ciri Khas {label}:** {ras_info[label]}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px; color: gray'>
        Dibuat oleh <b>Mahasiswa Teknik Informatika</b> – Universitas Aisyah Pringsewu<br>
        Powered by Streamlit & Scikit-learn
    </div>
""", unsafe_allow_html=True)