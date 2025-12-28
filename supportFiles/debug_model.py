import tensorflow as tf
from tensorflow.keras.models import load_model

print("\n===== LOADING MODEL =====")
model = load_model("spectrogram_seismic_detector.keras")
print("Model loaded.")

print("\n===== MODEL TYPE =====")
print(type(model))

print("\n===== MODEL SUMMARY =====")
try:
    model.summary()
except Exception as e:
    print("Summary failed:", e)

print("\n===== MODEL INPUT =====")
try:
    print(model.input)
except Exception as e:
    print("No model.input:", e)

print("\n===== MODEL OUTPUT =====")
try:
    print(model.output)
except Exception as e:
    print("No model.output:", e)

print("\n===== LAYERS =====")
for i, l in enumerate(model.layers):
    print(i, l.name, type(l))

print("\n===== TRY FORCE-CALL =====")
import numpy as np
dummy = np.zeros((1,129,32,1), dtype="float32")
try:
    y = model(dummy)
    print("Call OK. New input:", model.input)
except Exception as e:
    print("Call FAILED:", e)

print("\n===== FINAL INPUT =====")
try:
    print(model.input)
except Exception as e:
    print("Still no input:", e)

print("\n===== DONE =====")
