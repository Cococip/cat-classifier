import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load data
X, y = [], []
data_dir = 'dataset/12-ras-kucing'

for label in os.listdir(data_dir):
    class_path = os.path.join(data_dir, label)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img = cv2.imread(path)
            if img is not None:
                feature = preprocess_image(img)
                X.append(feature)
                y.append(label)

X = np.array(X)
y = np.array(y)

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Simpan model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/svm_model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

# Evaluasi model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical')
plt.title("Confusion Matrix - SVM Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
