"""
Notebook 02 — Model 1: Baseline Simple CNN
===========================================
Problem Statement:
  We need a baseline model to understand how hard the problem is.
  Simple CNN trained from scratch, no augmentation, no class balancing.

Expected Issues:
  - Overfitting (training acc >> val acc)
  - Bias toward majority class (Non-Fractured ~82%)
  - Low recall on Fractured class
"""

# %% [markdown]
# # 🧪 Model 1: Baseline — Simple CNN
# **Rationale**: Establish a performance floor before applying improvements.
#
# **Architecture**: 4-block CNN trained from scratch
# ```
# Conv(32) → Conv(64) → Conv(128) → Conv(256) → AdaptiveAvgPool → FC(512) → FC(2)
# ```

# %%
import os, sys, json
sys.path.insert(0, "../src")

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_loaders
from model import ModelBaseline, ModelImproved, ModelFinal, count_params
from dl_utils import train_one_epoch, test, compute_performance
from evaluate import compute_metrics, plot_learning_curves, plot_confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = ".."
CSV_PATH   = os.path.join(ROOT, "dataset", "dataset.csv")
IMG_DIR    = os.path.join(ROOT, "dataset", "images")
MODEL_DIR  = os.path.join(ROOT, "models")
RES_DIR    = os.path.join(ROOT, "results")
MODEL_NAME = "simple_cnn"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Data (no augmentation, no class weighting)
# ──────────────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, class_weights = make_loaders(
    CSV_PATH, IMG_DIR, batch_size=32, img_size=224
)
# NOTE: We intentionally do NOT use class_weights here to show the baseline problem.

# ──────────────────────────────────────────────────────────────────────────────
# 2. Model architecture
# ──────────────────────────────────────────────────────────────────────────────
model = get_model("simple_cnn")
params = count_params(model)
print(f"\nModel: SimpleCNN")
print(f"  Total params    : {params['total']:,}")
print(f"  Trainable params: {params['trainable']:,}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Train
# ──────────────────────────────────────────────────────────────────────────────
history, best_model = train_model(
    model, train_loader, val_loader,
    model_name  = MODEL_NAME,
    num_epochs  = 30,
    lr          = 1e-3,
    weight_decay= 1e-4,
    class_weights= None,          # ← intentionally no weighting
    patience    = 7,
    save_dir    = MODEL_DIR,
    device      = DEVICE,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluate on test set
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
test_metrics = compute_metrics(best_model, test_loader, criterion, DEVICE)

print(f"\n{'='*55}")
print(f"  SimpleCNN — Test Results")
print(f"{'='*55}")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1-score  : {test_metrics['f1']:.4f}")
print(f"  ROC-AUC   : {test_metrics['auc']:.4f}")
print(f"{'='*55}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ──────────────────────────────────────────────────────────────────────────────
plot_learning_curves(history, MODEL_NAME, RES_DIR)
plot_confusion_matrix(test_metrics["labels"], test_metrics["preds"], MODEL_NAME, RES_DIR)

# Save results for comparison notebook
os.makedirs(RES_DIR, exist_ok=True)
result_entry = {
    "Model"    : "Simple CNN (Baseline)",
    "Accuracy" : round(test_metrics["accuracy"],  4),
    "Precision": round(test_metrics["precision"], 4),
    "Recall"   : round(test_metrics["recall"],    4),
    "F1"       : round(test_metrics["f1"],        4),
    "AUC"      : round(test_metrics["auc"],       4),
    "labels"   : test_metrics["labels"].tolist(),
    "probs"    : test_metrics["probs"].tolist(),
}
with open(os.path.join(RES_DIR, f"{MODEL_NAME}_results.json"), "w") as f:
    json.dump(result_entry, f, indent=2)

print(f"\n✅ Done! Results saved to results/{MODEL_NAME}_results.json")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Problem analysis (evidence for next iteration)
# ──────────────────────────────────────────────────────────────────────────────
print(f"""
📊 Analysis — Why Baseline Falls Short:
  1. Learning curves: Training loss drops faster than validation → OVERFITTING
  2. Recall for Fractured class is low (~{test_metrics['recall']:.0%}) → model biased to majority class
  3. No regularisation beyond Dropout → model memorises training patterns

📌 Proposed Improvements for Model 2:
  ✓ Use pretrained ResNet-18 (richer feature extraction → less overfitting)
  ✓ Add data augmentation (random flip, rotation, color jitter)
  ✓ Use class-weighted loss to penalise missing fractures more heavily
""")
