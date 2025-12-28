import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading dataset...")
data = np.load("lunar_spectrogram_dataset.npz")

X_spec = data["X_spec"].astype("float32")
y_all  = data["y_all"]

print("Dataset loaded!")
print("X_spec:", X_spec.shape)
print("y_all :", y_all.shape)


# ======================
# CLEAN BAD VALUES
# ======================
X_spec[np.isnan(X_spec)] = 0.0
X_spec[np.isinf(X_spec)] = 0.0

print("NaNs after cleanup:", np.isnan(X_spec).sum())


# ======================
# NORMALIZE PER SAMPLE
# ======================
mins = X_spec.min(axis=(1,2,3), keepdims=True)
maxs = X_spec.max(axis=(1,2,3), keepdims=True)

X_spec = (X_spec - mins) / (maxs - mins + 1e-8)

print("Min:", X_spec.min())
print("Max:", X_spec.max())
print("Mean:", X_spec.mean())


# ======================
# TRAIN–TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_spec, y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

print("y_test distribution:", dict(zip(*np.unique(y_test, return_counts=True))))


# ======================
# OVERSAMPLE POSITIVES
# ======================
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

pos_idx_over = np.random.choice(pos_idx, size=len(pos_idx)*40, replace=True)

balanced_idx = np.hstack([neg_idx, pos_idx_over])
np.random.shuffle(balanced_idx)

X_train_bal = X_train[balanced_idx]
y_train_bal = y_train[balanced_idx]

print("Balanced training shape:", X_train_bal.shape)
print("Positive ratio:", y_train_bal.mean())


# ======================
# BUILD CNN
# ======================
input_shape = X_train.shape[1:]

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ======================
# TRAIN
# ======================
history = model.fit(
    X_train_bal, y_train_bal,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ======================
# PREDICT
# ======================
y_prob = model.predict(X_test).ravel()

print("Min prob:", y_prob.min())
print("Max prob:", y_prob.max())
print("Mean prob:", y_prob.mean())


# ======================
# THRESHOLD SWEEP
# ======================
def metrics(threshold):
    yp = (y_prob >= threshold).astype(int)

    tp = ((yp==1)&(y_test==1)).sum()
    fp = ((yp==1)&(y_test==0)).sum()
    fn = ((yp==0)&(y_test==1)).sum()

    precision = tp/(tp+fp+1e-8)
    recall    = tp/(tp+fn+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)

    return precision, recall, f1


print("\n===== THRESHOLD SWEEP =====")
best = (0,0,0,0)

for T in [i/10 for i in range(1,10)]:
    p,r,f1 = metrics(T)
    print(f"T={T:.2f}  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    if f1 > best[3]:
        best = (T,p,r,f1)

from sklearn.metrics import confusion_matrix
y_pred = (y_prob >= 0.40).astype(int)
print(confusion_matrix(y_test, y_pred))

print("\n===== BEST OPERATING POINT =====")
print(f"Best Threshold = {best[0]:.2f}")
print(f"Precision = {best[1]:.3f}")
print(f"Recall    = {best[2]:.3f}")
print(f"F1 Score  = {best[3]:.3f}")


# ======================
# SAVE MODEL
# ======================
model.save("spectrogram_seismic_detector.keras")
print("\nModel saved as spectrogram_seismic_detector.keras")
