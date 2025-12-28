import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os

# ========================
# LOAD MODEL
# ========================
model = load_model("rebuilt_detector.keras")

last_conv = model.get_layer("conv2d_2")
print("Using conv layer:", last_conv.name)

# ========================
# LOAD DATA
# ========================
data = np.load("lunar_spectrogram_dataset.npz")
X = data["X_spec"].astype("float32")
y = data["y_all"]

X[np.isnan(X)] = 0

mins = X.min(axis=(1,2,3), keepdims=True)
maxs = X.max(axis=(1,2,3), keepdims=True)
X = (X - mins) / (maxs - mins + 1e-8)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========================
# BUILD GRAD MODEL
# ========================
grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[last_conv.output, model.output]
)

# ========================
# GRAD-CAM FUNCTION
# ========================
def grad_cam(img):
    with tf.GradientTape() as tape:
        conv_maps, preds = grad_model(img)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_maps)
    weights = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(weights[:,None,None,:] * conv_maps, axis=-1)[0]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    return cam, preds.numpy()


# ========================
# OUTPUT FOLDER
# ========================
os.makedirs("gradcam_output", exist_ok=True)

# ========================
# PICK SAMPLES
# ========================
event_ids = np.where(y_test == 1)[0][:10]
nonevent_ids = np.where(y_test == 0)[0][:10]

print("Event samples:", event_ids)
print("Non-event samples:", nonevent_ids)


# ========================
# SAVE FUNCTION
# ========================
def save_cam(idx, label):
    img = X_test[idx:idx+1]
    cam, prob = grad_cam(img)

    spec = img[0,:,:,0]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title(f"{label}\nProb={prob[0][0]:.4f}")
    plt.imshow(spec.T, aspect='auto', origin='lower')

    plt.subplot(1,2,2)
    plt.title("Grad-CAM")
    plt.imshow(spec.T, aspect='auto', origin='lower')
    plt.imshow(cam.T, cmap='jet', alpha=0.5, origin='lower')

    out = f"gradcam_output/{label}_{idx}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)


# ========================
# RUN EVENT SAMPLES
# ========================
for i in event_ids:
    save_cam(i, "EVENT")

# ========================
# RUN NON-EVENT SAMPLES
# ========================
for i in nonevent_ids:
    save_cam(i, "NONEVENT")

print("\nDone 💙 Images saved in folder: gradcam_output/")
