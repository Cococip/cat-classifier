import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

def extract_lbp_features(gray, radius=1, n_points=8):
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = image.convert("HSV").resize((128, 128))
    hist = np.array(hsv.histogram()).astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_aspect_ratio(pil_img):
    w, h = pil_img.size
    return [w / h]

def preprocess_features(pil_img):
    pil_img = pil_img.resize((128, 128))
    gray = pil_img.convert("L")
    gray_np = np.array(gray)

    lbp = extract_lbp_features(gray_np)
    color_hist = extract_color_histogram(pil_img)
    aspect = extract_aspect_ratio(pil_img)

    return np.hstack([lbp, color_hist, aspect])
