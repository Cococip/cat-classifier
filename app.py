import streamlit as st
import cv2
import numpy as np
import joblib
from utils.preprocessing import preprocess_image
from PIL import Image

# Load model
model = joblib.load('model/svm_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

st.set_page_config(page_title="Deteksi Ras Kucing", layout="centered")

st.title("😺 Deteksi Ras Kucing")
st.markdown("Upload gambar kucing untuk mendeteksi rasnya. Menggunakan SVM + LBP")

uploaded_file = st.file_uploader("Upload gambar kucing (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Gambar yang diunggah', use_column_width=True, channels="BGR")

    with st.spinner('Mendeteksi...'):
        feature = preprocess_image(image)
        prediction = model.predict([feature])[0]
        prob = model.predict_proba([feature])[0].max()
        label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Prediksi: **{label}**")
        st.info(f"Akurasi Keyakinan: **{prob*100:.2f}%**")
