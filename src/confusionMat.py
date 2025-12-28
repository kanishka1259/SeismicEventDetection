import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

data = np.load("lunar_spectrogram_dataset.npz")
X_spec = data["X_spec"].astype("float32")
y_all = data["y_all"]

# same normalization you used before
X_spec[np.isnan(X_spec)] = 0
mins = X_spec.min(axis=(1,2,3), keepdims=True)
maxs = X_spec.max(axis=(1,2,3), keepdims=True)
X_spec = (X_spec - mins) / (maxs - mins + 1e-8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_spec, y_all, test_size=0.2, stratify=y_all, random_state=42
)

model = load_model("spectrogram_seismic_detector.keras")

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.40).astype(int)

print(confusion_matrix(y_test, y_pred))
