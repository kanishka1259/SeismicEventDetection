import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("spectrogram_seismic_detector.keras")

# ⭐ force-build model graph
_ = model(tf.zeros((1,129,32,1)))

print("\nLAYER SUMMARY\n")
for i, l in enumerate(model.layers):
    try:
        print(i, l.name, l.output.shape)
    except:
        print(i, l.name, "no shape yet")
