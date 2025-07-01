from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(gray_img, radius=1, n_points=8):
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def preprocess_image_pil(pil_image, size=(128, 128)):
    img = pil_image.convert("L").resize(size)  # Grayscale + Resize
    gray_np = np.array(img)
    return extract_lbp_features(gray_np)
