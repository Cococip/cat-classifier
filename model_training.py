import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils.preprocessing import preprocess_features

# Dataset folder
base_path = 'dataset/12-ras-kucing'
X, y = [], []

# Load + augmentasi data
for label in os.listdir(base_path):
    folder = os.path.join(base_path, label)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, file)
                try:
                    img = Image.open(path).convert("RGB")

                    # Asli
                    features = preprocess_features(img)
                    X.append(features)
                    y.append(label)

                    # Flip horizontal
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    X.append(preprocess_features(flipped))
                    y.append(label)

                    # Rotate 15 deg
                    rotated = img.rotate(15)
                    X.append(preprocess_features(rotated))
                    y.append(label)

                except Exception as e:
                    print(f"❌ Error: {path} -> {e}")

print("Jumlah data per kelas:", Counter(y))

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# (Optional) Tuning SVM dengan GridSearchCV
# param_grid = {
#     'C': [1, 10, 100],
#     'gamma': ['scale', 0.01, 0.001],
#     'kernel': ['rbf']
# }
# grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=2, n_jobs=-1)
# grid.fit(X_train, y_train)
# model = grid.best_estimator_
# print("Best params:", grid.best_params_)

# Train model biasa (tanpa grid search)
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Evaluasi
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
