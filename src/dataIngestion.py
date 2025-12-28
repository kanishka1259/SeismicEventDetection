# ==============================
# STEP 1: DATA INGESTION
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Load the train catalog
# ------------------------------
catalog_path = "data/lunar/train/lunar_train_catalog.csv"
catalog = pd.read_csv(catalog_path)

print("Catalog loaded successfully")
print("Catalog columns:", catalog.columns.tolist())
print("\nFirst 5 rows:")
print(catalog.head())

# ------------------------------
# 2. Select one seismic event
# ------------------------------
row = catalog.iloc[0]

filename = row["filename"]
event_time_sec = row["time_rel(sec)"]

print("\nSelected file:", filename)
print("Event time (seconds):", event_time_sec)

# ------------------------------
# 3. Load seismic waveform CSV
# ------------------------------
signal_path = os.path.join(
    "data/lunar/train/S12_GradeA",
    filename+".csv"
)
print("Trying to load:", signal_path)

df = pd.read_csv(signal_path)

print("\nSeismic file loaded")
print("Columns:", df.columns.tolist())
print(df.head())

# ------------------------------
# 4. Extract signal and time
# ------------------------------
signal = df["velocity(m/s)"].values
time_rel = df["time_rel(sec)"].values

print("\nSignal length:", len(signal))

# ------------------------------
# 5. Compute sampling rate
# ------------------------------
dt = time_rel[1] - time_rel[0]
sampling_rate = 1 / dt

print("Sampling rate (Hz):", sampling_rate)

# ------------------------------
# 6. Convert event time to index
# ------------------------------
event_index = int(event_time_sec * sampling_rate)

print("Event sample index:", event_index)
print("Signal length:", len(signal))

# ------------------------------
# 7. Plot signal with event marker
# ------------------------------
plt.figure(figsize=(14, 4))
plt.plot(signal, label="Seismic signal")
plt.axvline(event_index, color="red", linestyle="--", label="Event location")
plt.xlabel("Sample index")
plt.ylabel("Velocity (m/s)")
plt.title("Seismic Signal with Event Location")
plt.legend()
plt.show()

# ==============================
# STEP 2: SLIDING WINDOW DATASET
# ==============================

WINDOW_SIZE = 2000       # samples per window
STRIDE = 500             # step size
EVENT_RANGE = 2000       # tolerance around event

X = []
y = []
count = 0
for start in range(0, len(signal) - WINDOW_SIZE, STRIDE):
    end = start + WINDOW_SIZE
    
    window = signal[start:end]
    
    # Label rule:
    # If event index lies inside (start - tolerance, end + tolerance)
    if (event_index >= start - EVENT_RANGE) and (event_index <= end + EVENT_RANGE):
        label = 1
    else:
        label = 0
    
    X.append(window)
    y.append(label)
    X.append(window)
    y.append(label)

    count += 1
    if count % 200 == 0:
        print("Processed windows:", count)

X = np.array(X)
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
y = np.array(y)

print("Total windows created:", len(X))
print("Class distribution (0=no event, 1=event):")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
