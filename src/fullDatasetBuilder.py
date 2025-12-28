import pandas as pd
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
CATALOG_PATH = "data/lunar/train/lunar_train_catalog.csv"
TRAIN_DIR = "data/lunar/train"

WINDOW_SIZE = 2000
STRIDE = 500
EVENT_RANGE = 0  # +/- around event

# ==============================
# LOAD CATALOG
# ==============================
catalog = pd.read_csv(CATALOG_PATH)

print("Catalog size:", len(catalog))

X_all = []
y_all = []

# ==============================
# LOOP THROUGH ALL ROWS
# ==============================
for i, row in catalog.iterrows():

    file_name = row["filename"] + ".csv"
    event_time_sec = row["time_rel(sec)"]

    # find which folder the file is in
    # (S12_GradeA, S15_GradeB, etc)
    found = False
    for folder in os.listdir(TRAIN_DIR):
        folder_path = os.path.join(TRAIN_DIR, folder)
        if os.path.isdir(folder_path):
            fpath = os.path.join(folder_path, file_name)
            if os.path.exists(fpath):
                signal_path = fpath
                found = True
                break

    if not found:
        print("⚠️ Missing:", file_name)
        continue

    df = pd.read_csv(signal_path)

    signal = df["velocity(m/s)"].values
    time = df["time_rel(sec)"].values

    dt = time[1] - time[0]
    sampling_rate = 1 / dt

    event_index = int(event_time_sec * sampling_rate)

    X = []
    y = []

    for start in range(0, len(signal) - WINDOW_SIZE, STRIDE):
        end = start + WINDOW_SIZE
        window = signal[start:end]

        if start <= event_index <= end:
           label = 1
        else:
           label = 0


        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # normalize per window
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    X_all.append(X)
    y_all.append(y)

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(catalog)} files")

# ==============================
# STACK
# ==============================
X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

print("\n🎉 DATASET READY 🎉")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

unique, counts = np.unique(y_all, return_counts=True)
print("Label counts:", dict(zip(unique, counts)))

np.savez("lunar_windows_dataset.npz",
         X_all=X_all,
         y_all=y_all)

print("Saved to lunar_windows_dataset.npz")