# 🫁 AI-Powered Pneumonia Detection System

An end-to-end deep learning project that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** categories using Convolutional Neural Networks (CNNs) and transfer learning. The project benchmarks three architectures and integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize image regions driving model predictions.

---

## 📌 Overview

Pneumonia is an inflammatory lung condition detectable via characteristic opacities and infiltrates on chest radiographs. This project implements a complete computer vision pipeline in Python/TensorFlow:
- Dataset acquisition and augmentation
- Addressing class imbalance with weighted loss
- Multi-architecture benchmarking (Custom CNN vs. DenseNet121 vs. EfficientNetB0)
- Interpretability and model explainability using Grad-CAM heatmaps
- Model inference on unseen chest X-rays

---

## 🚀 Key Features

- **Binary Image Classification**: Distinguishes between `NORMAL` and `PNEUMONIA` chest X-rays.
- **Architectures Evaluated**:
  - **Custom CNN**: Baseline neural network trained from scratch.
  - **DenseNet121**: Deep feature-reuse network adapted via transfer learning.
  - **EfficientNetB0**: Lightweight, highly efficient scaled CNN adapted via transfer learning.
- **Class-Weighted Training**: Addresses dataset imbalance by penalizing minority-class misclassifications.
- **Explainable AI (XAI)**: Grad-CAM overlay heatmaps to visually verify diagnostic attention areas.
- **Comprehensive Evaluation**: Metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.
- **Serialized Model**: Ready-to-use exported best model (`.keras`).

---

## 📊 Experimental Results

All models were evaluated under identical validation conditions:

| Metric | Custom CNN | DenseNet121 | EfficientNetB0 (Best) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.21% | 93.87% | **95.15%** |
| **Normal Recall** | 80.60% | 96.02% | **96.02%** |
| **Pneumonia Recall** | 96.22% | 93.13% | **94.85%** |
| **Pneumonia Precision** | 93.49% | 98.55% | **98.57%** |
| **Weighted F1-Score** | 92.09% | 94.01% | **95.23%** |
| **ROC-AUC** | 0.9680 | 0.9895 | **0.9896** |
| **Total Parameters** | **390 K** | 7.04 M | 4.05 M |
| **Inference Time (approx)** | **~13.1 ms** | ~22.1 ms | ~19.7 ms |

### Key Observations
- **EfficientNetB0** achieved the highest overall performance across accuracy (95.15%), weighted F1-score (95.23%), and ROC-AUC (0.9896) with moderate parameter complexity.
- **DenseNet121** demonstrated competitive discriminative capability and high normal-case recall.
- **Custom CNN** provides the fastest inference (~13.1 ms) and smallest footprint (390K params), serving as an effective lightweight baseline.

---

## 🔍 Explainability with Grad-CAM

Deep learning models in medical imaging require interpretability to identify potential shortcuts or spurious correlations. 
- **Grad-CAM** generates localization heatmaps by computing gradients of the target class score with respect to feature maps of the final convolutional layer.
- Overlays highlight pulmonary infiltrates and consolidation patterns influencing predictions.

---

## 📁 Repository Structure

```text
pneumonia-detection/
├── ai_pneumonia_detection_model_comparison (1).ipynb   # Complete workflow notebook
├── pneumonia_detection_best_model.keras               # Serialized best trained model
├── .gitignore                                         # Git ignore patterns
└── README.md                                          # Documentation
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Deep Learning**: TensorFlow, Keras
- **Data & Scientific Computing**: NumPy, Pandas, Pillow
- **Evaluation & Metrics**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Dataset API**: KaggleHub

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/mani1854/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pillow kagglehub
```

### 3. Dataset Setup
The project uses the [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle. You can download it directly via the notebook using `kagglehub` or manually extract it into your workspace.

### 4. Run the Pipeline
Open the notebook in **Jupyter Lab / Notebook** or **Google Colab**:
```bash
jupyter notebook "ai_pneumonia_detection_model_comparison (1).ipynb"
```
Execute cells sequentially to inspect data distribution, model training graphs, comparative metric tables, Grad-CAM visualizations, and sample predictions.

---

## ⚠️ Medical Disclaimer

This project is developed for **academic and educational research purposes only**. The models and predictions should not be used as a clinical diagnostic tool or substitute for professional medical assessment.

---

## 📜 Acknowledgements

- Dataset provided by Paul Mooney via [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- TensorFlow & Keras open-source communities.
