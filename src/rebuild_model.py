import tensorflow as tf
from tensorflow.keras import layers, models

# --- recreate architecture ---
input_layer = layers.Input(shape=(129,32,1))

x = layers.Conv2D(32,(3,3),padding="same",activation="relu")(input_layer)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64,(3,3),padding="same",activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128,(3,3),padding="same",activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64,activation="relu")(x)
output = layers.Dense(1,activation="sigmoid")(x)

model = models.Model(inputs=input_layer, outputs=output)

print("Functional model created!")

# --- load weights from old model ---
old = tf.keras.models.load_model("spectrogram_seismic_detector.keras")

for nw, ow in zip(model.weights, old.weights):
    nw.assign(ow)

print("Weights copied!")

# --- save properly ---
model.save("rebuilt_detector.keras")
print("Saved as rebuilt_detector.keras")
