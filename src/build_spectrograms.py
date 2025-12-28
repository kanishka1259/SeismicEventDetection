import numpy as np
import librosa
data = np.load("lunar_windows_dataset.npz")

X_all = data["X_all"]
y_all = data["y_all"]

print("Loaded:", X_all.shape, y_all.shape)

# X_all and y_all must already exist in memory

SPEC_LIST = []
for i, x in enumerate(X_all):

    # make sure float32
    x = x.astype(float)

    # STFT → spectrogram
    S = np.abs(librosa.stft(x, n_fft=256, hop_length=64))

    # log scale
    S = librosa.amplitude_to_db(S, ref=np.max)

    # resize to consistent shape
    S = (S - S.min()) / (S.max() - S.min())

    SPEC_LIST.append(S)

    if (i+1) % 5000 == 0:
        print("Processed:", i+1)

X_spec = np.array(SPEC_LIST)
X_spec = X_spec[..., np.newaxis]
print("Spectrogram dataset shape:", X_spec.shape)

np.savez("lunar_spectrogram_dataset.npz",
         X_spec=X_spec,
         y_all=y_all)

print("Saved lunar_spectrogram_dataset.npz")