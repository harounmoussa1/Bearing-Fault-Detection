# Bearing Fault Detection – Siamese Networks

> Few-shot anomaly detection on industrial vibration signals using Siamese Neural Networks.

## Overview

This project tackles the classification of industrial bearing faults from raw vibration signals using deep metric learning. Two architectures are compared: a **Conv1D Siamese Network** on raw signals and a **Conv2D Siamese Network** on 2D image encodings (GASF/GADF).

**Dataset:** [SUBF-V2.0 – Bearing Faults Sound Data](https://www.kaggle.com/)  
**Fault classes:** Normal · IRF (Inner Race Fault) · ORF (Outer Race Fault)

---

## Pipeline
Raw Signals (100k pts)
→ Min-Max Normalization (row-wise)
→ PAA Compression (224 / 512 / 1024 pts)
→ GASF / GADF Encoding → 2D Images
→ Conv1D Siamese (Contrastive Loss + Euclidean Distance)
→ Conv2D Siamese (Triplet Loss + Cosine Similarity)
→ Evaluation: Accuracy · F1 · AUC · ROC

---

## Results

| Model | Encoding | Loss | Accuracy | 
|---|---|---|---|
| Conv1D Siamese | Raw PAA signal | Contrastive Loss | **86%** |
| Conv2D Siamese | GASF/GADF 224–1024 | Triplet Loss | **81%** |

---

## Tech Stack

- Python 3.10
- TensorFlow 2.20
- pyts (GASF/GADF encoding)
- scikit-learn
- Google Colab (GPU T4)

---

## Project Structure
├── PS3.ipynb
├── README.md
└── requirements.txt

---

## How to Run

1. Clone the repo
```bash
git clone https://github.com/harounmoussa1/Bearing-Fault-Detection.git
```

2. Upload `archive.zip` (SUBF-V2.0 dataset) to your Google Drive

3. Open `PS3.ipynb` in Google Colab and run all cells

---

## Author

**Haroun Moussa** — ENISo, Applied Computer Science Engineering  
[LinkedIn](https://www.linkedin.com/in/harounmoussa/) · [GitHub](https://github.com/harounmoussa1)
