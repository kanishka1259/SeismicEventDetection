from tensorflow.keras.models import load_model
import numpy as np

m = load_model("rebuilt_detector.keras")

print("INPUT =", m.input)
print("OUTPUT =", m.output)

x = np.zeros((1,129,32,1),dtype="float32")
print("Predict =", m.predict(x))
