import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

def extract_lbp_features(gray, radius=1, n_points=8):
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_histogram(image, bins=(8, 8, 8)):
    image = image.resize((128, 128))
    hsv = image.convert('HSV')
    hist = np.array(hsv.histogram()).astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def preprocess_features(pil_img):
    gray = pil_img.convert("L").resize((128, 128))
    gray_np = np.array(gray)
    lbp = extract_lbp_features(gray_np)

    color_hist = extract_color_histogram(pil_img)

    features = np.hstack([lbp, color_hist])
    return features
