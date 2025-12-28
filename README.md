# 🛰️ Seismic Event Detection using CNN & Spectrograms

This project detects **seismic events vs non-events** from lunar vibration data.  
Raw time-series windows are converted into **spectrogram images**, then a **Convolutional Neural Network (CNN)** classifies whether an event occurred.

Grad-CAM visualizations are used to understand **which time-frequency regions influenced the prediction**.

---

## ✅ Key Features
- 📊 Convert raw seismic signals → **Mel-spectrograms**
- 🧠 Train a CNN binary classifier
- ⚖ Handle extreme class imbalance
- 📈 Precision / Recall / F1 evaluation
- 🔥 Grad-CAM explainability
- 🔍 Model inspection utilities

> **Dataset is NOT included.** (`data/` is ignored intentionally.)

---

## 📂 Project Structure
<details>
  <summary><code>seismic-event-detection/</code></summary>

  - <code>src/</code>
    - <code>datalngestion.py</code>
    - <code>build_spectrograms.py</code>
    - <code>fullDatasetBuilder.py</code>
    - <code>train_model.py</code>
    - <code>confusionMat.py</code>
    - <code>gradCam_Visualize.py</code>
    - <code>rebuild_model.py</code>

  - <code>supportFiles/</code>
    - <code>inspect_model.py</code>
    - <code>debug_model.py</code>
    - <code>check_rebuilt.py</code>
    - <code>printLayers.py</code>

  - <code>gradcam_output/</code>

  - <code>rebuilt_detector.keras</code>  
  - <code>spectrogram_seismic_detector.keras</code>  
  - <code>spectrogram_seismic_detector.h5</code>  
  - <code>sampling_graph.png</code>  
  - <code>spectrogram_gradCAM.png</code>  
  - <code>README.md</code>

</details>

---

## 🔄 End-to-End Workflow

### 1️⃣ Dataset Loading & Cleaning

**Script:** `src/datalngestion.py`  

- Load raw lunar seismic windows and corresponding labels. 
- Remove NaNs / invalid samples and enforce consistent shapes.  
- Compute label distribution to quantify class imbalance.
- Verify sampling rate and input dimensions for downstream spectrogram generation.

**Main outputs:**

- `X_raw` – cleaned time-series windows  
- `y_all` – binary labels (event = 1, non-event = 0)

---

### 2️⃣ Time-Series → Spectrograms

**Script:** `src/build_spectrograms.py`  

Each time-series window is converted into a **129 × 32 × 1** spectrogram, acting as a time–frequency “image” for the CNN. The pipeline uses mel-scaled or STFT-based spectrograms to encode both spectral content and temporal evolution.

**Key points:**

- Input: `X_raw` windows  
- Output: `X_spec` with shape `(N, 129, 32, 1)`

---

### 3️⃣ Build Final Dataset File

**Script:** `src/fullDatasetBuilder.py`  

This script aggregates all spectrograms and labels into a single compressed dataset file for easy reuse and reproducibility.

It creates:

- `lunar_spectrogram_dataset.npz`

containing:

- `X_spec` – spectrogram images  
- `y_all` – binary labels aligned with each spectrogram

---

### 4️⃣ Train CNN Classifier

**Script:** `src/train_model.py`  

Training uses stratified splitting, imbalance-aware sampling, and threshold tuning to prioritize event detection quality on rare signals.

**Features:**

- ✔ Stratified train/test split to preserve class ratios. 
- ✔ Oversampling of the minority (event) class.
- ✔ Balanced minibatches during training.  
- ✔ Per-sample normalization of spectrograms.  
- ✔ Threshold sweep on validation scores for best Precision / Recall / F1.

**Saved models:**

- `spectrogram_seismic_detector.keras`  
- `rebuilt_detector.keras`

---

### 5️⃣ Evaluate Model Performance

**Script:** `src/confusionMat.py`  

Computes the confusion matrix and standard metrics for a chosen decision threshold, which is crucial under class imbalance.

**Outputs:**

- Confusion matrix:

  - TN, FP  
  - FN, TP  

- Metrics:

  - Precision  
  - Recall  
  - F1-Score  

These metrics are often more informative than plain accuracy for rare-event detection tasks.

---

### 6️⃣ Explainability with Grad-CAM

**Script:** `src/gradCam_Visualize.py`  

Grad-CAM generates **overlay heatmaps** that highlight which time–frequency regions in the spectrogram most influenced an “event” prediction.This helps confirm that the network focuses on physically meaningful seismic patterns instead of artifacts.
**Outputs:**

- Heatmap overlays saved to:

  - `gradcam_output/`

---

## 🧠 CNN Architecture

The CNN is a lightweight 2D convolutional model operating on single-channel spectrograms.

**Layer stack:**

- Conv2D → BatchNorm → MaxPool  
- Conv2D → BatchNorm → MaxPool  
- Conv2D → BatchNorm  
- GlobalAveragePooling  
- Dense(64, relu)  
- Dense(1, sigmoid)

Total parameters are approximately **305K**, making the model suitable for deployment on modest hardware while retaining enough capacity for time–frequency patterns.

---

## ⚖ Handling Class Imbalance

The dataset is strongly skewed:

- **Non-Event ≫ Event**

To mitigate this, the training pipeline adopts standard imbalance-handling strategies.

**Techniques:**

- ✔ Oversampling of positive (event) samples in the training set.
- ✔ Balanced minibatches to avoid majority-class dominance.  
- ✔ Decision threshold tuning to balance precision vs recall.
- ✔ Continuous monitoring of **recall** so genuine events are not missed.

---

## 📊 Example Results

Typical performance (for an example trained model and chosen threshold):

| Metric         | Value  |
|---------------|--------|
| Best Threshold | 0.40  |
| Precision     | ~0.35 |
| Recall        | ~0.36 |
| F1 Score      | ~0.36 |

In highly imbalanced settings, F1 and recall are often more relevant than raw accuracy when the goal is **reliable event detection with minimal false alarms**.

---

## 🔥 Grad-CAM Interpretation

Grad-CAM uses gradients flowing into the last convolutional layer to compute a **class-specific importance map** over the feature maps, which is then upsampled to the input size. For spectrograms, these heatmaps highlight the **time–frequency bands** that contributed most to the “event” decision.

**Useful for:**

- ✔ Scientific interpretability of detected events.  
- ✔ Trust and debugging of the ML model. 
- ✔ Presentation-ready visualizations and reports.

Example:

- Spectrogram + Grad-CAM heat overlay showing active frequency bands around transient energy bursts, making the model’s reasoning visually clear.

---

## 🚀 How To Run

### 1️⃣ Install dependencies
pip install tensorflow numpy matplotlib scikit-learn librosa
### 2️⃣ Build dataset
python src/fullDatasetBuilder.py
### 3️⃣ Train model
python src/train_model.py
### 4️⃣ Evaluate performance
python src/confusionMat.py
### 5️⃣ Generate Grad-CAM visualizations
python src/gradCam_Visualize.py


---

## 🏁 Project Status

- ✔ Dataset processed  
- ✔ CNN trained  
- ✔ Performance evaluated  
- ✔ Explainability added  

🎉 Project completed successfully.

---

## 📌 Notes

- `data/` folder is intentionally excluded from the repository to avoid distributing raw seismic datasets.  
- Codebase is **modular**, separating data processing, model training, evaluation, and visualization.
- Easy to **retrain** on new stations or adapt to other planetary bodies with spectrogram-based seismic data.

---

## 👩‍💻 Author

**Kanishka**  
Seismic Signal Processing • Machine Learning • Explainable AI

