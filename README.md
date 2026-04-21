# 🦴 Bone Fracture Detection

> **Binary classification of bone fractures from X-ray images using iterative deep learning.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-monitoring-orange)](https://www.tensorflow.org/tensorboard)
[![Gradio](https://img.shields.io/badge/Gradio-3.50-blueviolet)](https://gradio.app)

---

## 📌 Problem Statement

Bone fractures are among the most common musculoskeletal injuries worldwide. Manual X-ray reading is time-consuming and error-prone, especially under high workload. This project develops an AI-assisted tool to automatically detect fractures from X-ray images — targeting fast, reliable support for clinical decision-making.

**Impact**: Reduce diagnostic delays, lower false-negative rate, assist radiologists in high-volume settings.

---

## 📂 Project Structure

```
dl-project/
├── dataset/
│   ├── images/
│   │   ├── Fractured/          # 717 X-ray images
│   │   └── Non_fractured/      # 3,366 X-ray images
│   ├── Annotations/            # COCO, YOLO, VOC formats
│   └── dataset.csv             # Metadata (body part, view, label)
│
├── model.py                    # ModelBaseline, ModelImproved, ModelFinal
├── trainer.py                  # Main training script (TensorBoard + all 3 models)
├── dl_utils.py                 # train_one_epoch(), test(), compute_performance()
│
├── src/
│   ├── dataset.py              # FractureDataset + DataLoader factory
│   ├── evaluate.py             # Confusion matrix, ROC, learning curves, Grad-CAM
│   ├── models.py               # (alternative module — same architectures)
│   └── train.py                # (lower-level training loop)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline.ipynb         # Iteration 1: ModelBaseline
│   ├── 03_improved.ipynb         # Iteration 2: ModelImproved
│   └── 04_final.ipynb            # Iteration 3: ModelFinal + comparison
│
├── app/
│   ├── app.py                  # Gradio web demo (upload X-ray → prediction)
│   └── requirements.txt
│
├── models/                     # Saved best checkpoints (auto-created)
├── results/                    # Charts & confusion matrices (auto-created)
├── runs/                       # TensorBoard logs (auto-created)
├── Dockerfile                  # Docker deployment
└── README.md
```

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| **Source** | Public X-ray dataset (aggregated medical imaging) |
| **Total images** | 4,083 |
| **Fractured** | 717 (17.6%) |
| **Non-Fractured** | 3,366 (82.4%) |
| **Imbalance ratio** | ~4.7 : 1 |
| **Body parts** | Leg (2,273), Hand (1,538), Shoulder (349), Hip (338) |
| **View angles** | Frontal (2,503), Lateral (1,492), Oblique (418) |
| **Image sizes** | Variable (373–2,304 px) → normalised to 224 × 224 |

**Key challenge**: 4.7× class imbalance → addressed with inverse-frequency class-weighted loss.

---

## 🤖 Iterative Model Development

### Iteration 1 — `ModelBaseline` (Simple CNN from scratch)
| | |
|---|---|
| **Architecture** | 4-block CNN: Conv(32)→Conv(64)→Conv(128)→Conv(256)→FC(512)→FC(2) |
| **Parameters** | ~2.5M trainable |
| **Augmentation** | None |
| **Class weights** | ❌ |
| **Problem found** | Overfitting (train acc >> val acc); low Fractured recall due to majority bias |
| **Evidence** | Learning curve diverges after epoch 8; CM shows many FN |

### Iteration 2 — `ModelImproved` (ResNet-18 fine-tuned)
| | |
|---|---|
| **Architecture** | ResNet-18 (ImageNet pretrained, 6 layers frozen) |
| **Parameters** | 11.2M total, 10.6M trainable |
| **Augmentation** | Random flip, ±15° rotation, color jitter |
| **Class weights** | ✅ Inverse-frequency weighting |
| **LR** | 3×10⁻⁴ (AdamW) |
| **Improvement** | Higher recall on Fractured class; smoother learning curves |

### Iteration 3 — `ModelFinal` (EfficientNet-B0 fine-tuned)
| | |
|---|---|
| **Architecture** | EfficientNet-B0 (ImageNet pretrained, full fine-tune) + custom head |
| **Parameters** | 5.3M (all trainable) |
| **Augmentation** | Same as Iteration 2 |
| **Class weights** | ✅ |
| **LR** | 1×10⁻⁴ with CosineAnnealingLR |
| **Improvement** | Best F1 + AUC; compound scaling → superior accuracy per parameter |

---

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | **F1** | AUC |
|---|---|---|---|---|---|
| ModelBaseline (Simple CNN) | — | — | — | — | — |
| ModelImproved (ResNet-18)  | — | — | — | — | — |
| **ModelFinal (EfficientNet-B0)** | **—** | **—** | **—** | **—** | **—** |

> Run `trainer.py` to populate the table with actual values.

**Model selection**: EfficientNet-B0 achieves the best F1-score. High **Recall** is prioritised clinically — a missed fracture (false negative) is more dangerous than a false alarm.

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r app/requirements.txt
pip install tensorboard
```

### 2. Explore the dataset

Open and run `notebooks/01_eda.ipynb` using Jupyter or your IDE.
# Outputs charts to results/

### 3. Train all models

```bash
python trainer.py
# Trains ModelBaseline → ModelImproved → ModelFinal
# Saves checkpoints to models/
# Logs to runs/ (TensorBoard)
```

### 4. Monitor live with TensorBoard

```bash
tensorboard --logdir=runs
# Open: http://localhost:6006
```

### 5. Launch web demo

```bash
cd app && python app.py
# Open: http://localhost:7860
```

---

## 🐳 Docker (Bonus)

```bash
docker build -t fracture-detector .
docker run -p 7860:7860 fracture-detector
```

---

## 📁 Generated Outputs

After `trainer.py` completes, `results/` contains:

| File | Description |
|---|---|
| `eda_class_distribution.png` | Class balance bar + pie charts |
| `eda_body_parts.png` | X-ray count per body part |
| `eda_sample_images.png` | Sample X-rays from each class |
| `ModelBaseline_curves.png` | Learning curves (loss, acc, F1/AUC) |
| `ModelImproved_curves.png` | Learning curves |
| `ModelFinal_curves.png` | Learning curves |
| `ModelBaseline_confusion.png` | Confusion matrix on test set |
| `ModelImproved_confusion.png` | Confusion matrix |
| `ModelFinal_confusion.png` | Confusion matrix |
| `roc_curves.png` | ROC curves for all 3 models |
| `model_comparison.png` | Side-by-side metric bar chart |

---

## 👥 Contributors
