import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils.preprocessing import preprocess_features

# Load dataset
X, y = [], []
base_path = 'dataset/12-ras-kucing'

for label in os.listdir(base_path):
    folder = os.path.join(base_path, label)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder, file)
                try:
                    img = Image.open(path).convert("RGB")
                    features = preprocess_features(img)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error on {path}: {e}")

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train SVM
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# Simpan model
os.makedirs('model', exist_ok=True)
joblib.dump(clf, 'model/svm_model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Evaluasi
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
