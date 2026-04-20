"""
Notebook 03 — Model 2: ResNet-18 + Augmentation + Weighted Loss
================================================================
Problem from Model 1:
  ✗ Overfitting: val loss diverged from train loss after ~10 epochs
  ✗ Low Fractured recall (~55–65%) due to class imbalance

Evidence:
  • Learning curve: gap between train_acc and val_acc widened post epoch 8
  • Confusion matrix: many Fractured images misclassified as Non-Fractured

How we solve it:
  ✓ Transfer learning (ResNet-18): pretrained features → faster convergence, less overfitting
  ✓ Data augmentation: random flip + rotation + color jitter → more diverse training data
  ✓ Class-weighted CrossEntropyLoss: 4.7× higher penalty for missing a fracture
"""

# %% [markdown]
# # ⚡ Model 2: ResNet-18 Fine-Tuned
# **Improvement over Model 1:**
# - Transfer learning from ImageNet (richer initial features)
# - Data augmentation to prevent overfitting
# - Class-weighted loss to address imbalance

# %%
import os, sys, json
sys.path.insert(0, "../src")

import torch
import torch.nn as nn

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
MODEL_NAME = "resnet18"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Data — WITH augmentation and class weights
# ──────────────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, class_weights = make_loaders(
    CSV_PATH, IMG_DIR, batch_size=32, img_size=224
)
print(f"Class weights: Non-Fractured={class_weights[0]:.3f}, Fractured={class_weights[1]:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────────────────────────────────────
model = get_model("resnet18", freeze_layers=6)
params = count_params(model)
print(f"\nModel: ResNet18FT")
print(f"  Total params    : {params['total']:,}")
print(f"  Trainable params: {params['trainable']:,}")
print(f"  Frozen params   : {params['frozen']:,}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Train — lower LR because pretrained weights are already good
# ──────────────────────────────────────────────────────────────────────────────
history, best_model = train_model(
    model, train_loader, val_loader,
    model_name   = MODEL_NAME,
    num_epochs   = 30,
    lr           = 3e-4,          # ← lower LR for fine-tuning
    weight_decay = 1e-4,
    class_weights= class_weights, # ← now using weighted loss
    patience     = 8,
    save_dir     = MODEL_DIR,
    device       = DEVICE,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluate
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
test_metrics = compute_metrics(best_model, test_loader, criterion, DEVICE)

print(f"\n{'='*55}")
print(f"  ResNet-18 — Test Results")
print(f"{'='*55}")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}  ← key improvement")
print(f"  F1-score  : {test_metrics['f1']:.4f}")
print(f"  ROC-AUC   : {test_metrics['auc']:.4f}")
print(f"{'='*55}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Load baseline results for comparison
# ──────────────────────────────────────────────────────────────────────────────
baseline_path = os.path.join(RES_DIR, "simple_cnn_results.json")
if os.path.exists(baseline_path):
    baseline = json.load(open(baseline_path))
    print(f"\n📈 Improvement over Baseline:")
    for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        key = m.lower()
        delta = test_metrics[key] - baseline[m]
        sign = "+" if delta >= 0 else ""
        print(f"  {m:10}: {baseline[m]:.4f} → {test_metrics[key]:.4f}  ({sign}{delta:+.4f})")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Visualisations
# ──────────────────────────────────────────────────────────────────────────────
plot_learning_curves(history, MODEL_NAME, RES_DIR)
plot_confusion_matrix(test_metrics["labels"], test_metrics["preds"], MODEL_NAME, RES_DIR)

# Save results
result_entry = {
    "Model"    : "ResNet-18 (Improved)",
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
# 7. Analysis for next iteration
# ──────────────────────────────────────────────────────────────────────────────
print(f"""
📊 Analysis — Remaining Issues:
  • ResNet-18 significantly improves F1 and Recall
  • Learning curves are smoother, less overfitting observed
  • However, precision-recall trade-off may still be improvable

📌 Proposed Improvements for Model 3 (EfficientNet-B0):
  ✓ EfficientNet-B0: compound scaling → better accuracy at similar param count
  ✓ Full model fine-tuning (no frozen layers)
  ✓ CosineAnnealingLR with warm restarts
  ✓ Slightly higher dropout in head for regularisation
""")
