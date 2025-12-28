from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

m = load_model("spectrogram_seismic_detector.keras")

print("\n===== MODEL SUMMARY =====")
m.summary()

print("\n===== MODEL INPUT =====")
try:
    print(m.input)
except Exception as e:
    print("No model.input:", e)

print("\n===== MODEL OUTPUT =====")
try:
    print(m.output)
except Exception as e:
    print("No model.output:", e)

print("\n===== LAYERS =====")
for i,l in enumerate(m.layers):
    print(i, l.name, type(l))
