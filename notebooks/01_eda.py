"""
Notebook 01 — Exploratory Data Analysis
========================================
Run this script to explore the bone fracture dataset.
"""

# %% [markdown]
# # 📊 Bone Fracture Detection — EDA
# **Dataset**: 4,083 X-Ray images (Fractured vs Non-Fractured)
# 
# This notebook explores:
# - Class distribution
# - Body part & scan-view distribution
# - Sample images from each class
# - Image size statistics

# %%
import os, sys, csv, random, json
sys.path.insert(0, "../src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = ".."
CSV_PATH = os.path.join(ROOT, "dataset", "dataset.csv")
IMG_DIR  = os.path.join(ROOT, "dataset", "images")
RES_DIR  = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load metadata
# ──────────────────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV_PATH)))
print(f"Total images in CSV: {len(rows)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Class distribution
# ──────────────────────────────────────────────────────────────────────────────
labels = [int(r["fractured"]) for r in rows]
class_names = ["Non-Fractured", "Fractured"]
counts = Counter(labels)
n_nonfrac, n_frac = counts[0], counts[1]
print(f"\nClass Distribution:")
print(f"  Non-Fractured : {n_nonfrac:,}  ({n_nonfrac/len(rows):.1%})")
print(f"  Fractured     : {n_frac:,}   ({n_frac/len(rows):.1%})")
print(f"  Imbalance ratio: {n_nonfrac/n_frac:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0F0F1A")

# Bar chart
ax = axes[0]
ax.set_facecolor("#1A1A2E")
colors = ["#4C9BE8", "#E85C5C"]
bars = ax.bar(class_names, [n_nonfrac, n_frac], color=colors, edgecolor="#FFFFFF22", linewidth=1.5)
for bar, val in zip(bars, [n_nonfrac, n_frac]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}\n({val/len(rows):.1%})", ha="center", color="white", fontsize=12)
ax.set_title("Class Distribution", color="white", fontsize=14, fontweight="bold")
ax.set_ylabel("Count", color="white")
ax.tick_params(colors="white")
ax.spines[:].set_color("#333355")
ax.set_ylim(0, max(n_nonfrac, n_frac) * 1.2)

# Pie chart
ax2 = axes[1]
ax2.set_facecolor("#0F0F1A")
wedges, texts, autotexts = ax2.pie(
    [n_nonfrac, n_frac],
    labels=class_names,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops=dict(edgecolor="#FFFFFF44", linewidth=1.5),
)
for t in texts + autotexts:
    t.set_color("white")
    t.set_fontsize(12)
ax2.set_title("Class Balance", color="white", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "eda_class_distribution.png"),
            dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print(f"\n✅ Saved: eda_class_distribution.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Body part distribution
# ──────────────────────────────────────────────────────────────────────────────
body_cols = ["hand", "leg", "hip", "shoulder"]
body_counts = {col: sum(int(r[col]) for r in rows) for col in body_cols}
print("\nBody Part Distribution:", body_counts)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0F0F1A")
ax.set_facecolor("#1A1A2E")
palette = ["#4C9BE8", "#E85C5C", "#A78BFA", "#34D399"]
bars = ax.bar(list(body_counts.keys()), list(body_counts.values()),
              color=palette, edgecolor="#FFFFFF22")
for bar, val in zip(bars, body_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(val), ha="center", color="white", fontsize=11)
ax.set_title("Images per Body Part", color="white", fontsize=14, fontweight="bold")
ax.set_ylabel("Count", color="white")
ax.tick_params(colors="white")
ax.spines[:].set_color("#333355")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "eda_body_parts.png"),
            dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print("✅ Saved: eda_body_parts.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Scan view distribution
# ──────────────────────────────────────────────────────────────────────────────
view_cols = ["frontal", "lateral", "oblique"]
view_counts = {col: sum(int(r[col]) for r in rows) for col in view_cols}
print("\nView Distribution:", view_counts)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Sample images
# ──────────────────────────────────────────────────────────────────────────────
random.seed(42)
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("Sample X-Ray Images", color="white", fontsize=16, fontweight="bold")

for col, (subdir, cls_name, color) in enumerate([
    ("Non_fractured", "Non-Fractured", "#4C9BE8"),
    ("Fractured",     "Fractured",     "#E85C5C"),
]):
    folder = os.path.join(IMG_DIR, subdir)
    imgs   = random.sample(os.listdir(folder), 5)
    for row, fname in enumerate(imgs):
        ax = fig.add_subplot(2, 5, col * 5 + row + 1)
        ax.set_facecolor("#1A1A2E")
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        ax.imshow(img, cmap="gray")
        ax.set_title(cls_name if row == 2 else "", color=color, fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "eda_sample_images.png"),
            dpi=120, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print("✅ Saved: eda_sample_images.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Image size statistics
# ──────────────────────────────────────────────────────────────────────────────
print("\nSampling image sizes (200 random images)...")
sizes = []
all_imgs = []
for subdir in ["Fractured", "Non_fractured"]:
    folder = os.path.join(IMG_DIR, subdir)
    for f in os.listdir(folder):
        all_imgs.append(os.path.join(folder, f))

sampled = random.sample(all_imgs, min(200, len(all_imgs)))
for path in sampled:
    try:
        w, h = Image.open(path).size
        sizes.append((w, h))
    except:
        pass

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
print(f"Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0F0F1A")
for ax, vals, label, col in zip(
        axes,
        [widths, heights],
        ["Width (px)", "Height (px)"],
        ["#4C9BE8", "#E85C5C"]):
    ax.set_facecolor("#1A1A2E")
    ax.hist(vals, bins=30, color=col, edgecolor="#FFFFFF22", alpha=0.85)
    ax.axvline(np.mean(vals), color="white", linestyle="--", label=f"mean={np.mean(vals):.0f}")
    ax.set_title(f"Image {label} Distribution", color="white", fontsize=13)
    ax.set_xlabel(label, color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "eda_image_sizes.png"),
            dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
plt.close()
print("✅ Saved: eda_image_sizes.png")

print("\n📊 EDA complete! All charts saved to results/")
print("Key findings:")
print(f"  • Class imbalance: {n_nonfrac/n_frac:.1f}x more Non-Fractured → need weighted loss or augmentation")
print(f"  • Most images are leg/hand X-rays")
print(f"  • Highly variable image sizes → normalization to 224×224 required")
