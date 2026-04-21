"""
app.py — Gradio web demo for Bone Fracture Detection.
Compatible with Gradio 3.x

Run: python app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr

from model import ModelBaseline, ModelImproved, ModelFinal

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
CLASSES    = ["Non-Fractured", "Fractured"]
IMG_SIZE   = 224

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

MODEL_CONFIGS = {
    "🧪 Baseline — Simple CNN":   (ModelBaseline,  "ModelBaseline_best_vloss.pth"),
    "⚡ Improved — ResNet-18":     (ModelImproved,  "ModelImproved_best_vloss.pth"),
    "🏆 Final — EfficientNet-B0": (ModelFinal,     "ModelFinal_best_vloss.pth"),
}
_model_cache: dict = {}


def _load_model(model_key: str):
    if model_key in _model_cache:
        return _model_cache[model_key]
    model_cls, filename = MODEL_CONFIGS[model_key]
    ckpt_path = os.path.join(MODEL_DIR, filename)
    model = model_cls()
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"[app] Loaded: {filename}")
    else:
        print(f"[app] WARNING: checkpoint not found → {ckpt_path}")
    model = model.to(DEVICE).eval()
    _model_cache[model_key] = model
    return model


# ── Grad-CAM helper ───────────────────────────────────────────────────────────
def _gradcam_overlay(model, tensor, img_orig, arch):
    """Returns Grad-CAM blended PIL image, or None on failure."""
    try:
        # Pick target layer by architecture
        if "ResNet" in arch:
            target = model.features[-2][-1]
        elif "Efficient" in arch:
            target = model.model.features[-1][0]
        else:
            target = model.features[8]   # last Conv block in ModelBaseline

        feats_h, grads_h = {}, {}

        def fwd(_, __, out): feats_h["f"] = out.detach()
        def bwd(_, __, g):   grads_h["g"] = g[0].detach()

        fh = target.register_forward_hook(fwd)
        bh = target.register_full_backward_hook(bwd)

        t = tensor.clone().requires_grad_(True)
        logits = model(t)
        model.zero_grad()
        logits[0, 1].backward()
        fh.remove(); bh.remove()

        f, g = feats_h.get("f"), grads_h.get("g")
        if f is None or g is None:
            return None

        cam = F.relu((g.mean(dim=[2, 3], keepdim=True) * f).sum(1, keepdim=True))
        cam = F.interpolate(cam, (img_orig.height, img_orig.width),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        import matplotlib.cm as cm
        heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        heat_pil = Image.fromarray(heat).resize(
            (img_orig.width, img_orig.height), Image.LANCZOS)
        return Image.blend(img_orig.convert("RGB"), heat_pil, alpha=0.45)
    except Exception as e:
        print(f"[GradCAM] skipped: {e}")
        return None


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image, model_choice):
    if image is None:
        return {c: 0.0 for c in CLASSES}, "⚠️ Please upload an X-ray image.", None

    model = _load_model(model_choice)
    arch  = model_choice   # used only for Grad-CAM layer selection

    img_rgb  = image.convert("RGB")
    tensor   = TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits   = model(tensor)
        probs    = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        conf     = conf.item()
        pred_cls = CLASSES[idx.item()]

    label_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}

    emoji = "🦴❌" if pred_cls == "Fractured" else "✅"
    bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    info  = f"""## {emoji} Prediction: **{pred_cls}**

| Metric | Value |
|---|---|
| Confidence | **{conf:.1%}** |
| Non-Fractured prob | {probs[0].item():.1%} |
| Fractured prob | {probs[1].item():.1%} |
| Model | {model_choice} |
| Device | {DEVICE.upper()} |

`[{bar}]` {conf:.0%}

> ⚠️ For research purposes only. Consult a qualified radiologist."""

    gradcam = _gradcam_overlay(model, tensor, img_rgb, arch)
    return label_dict, info, gradcam


# ── Example images ────────────────────────────────────────────────────────────
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "images")
EXAMPLES = []
for sub, _ in [("Fractured", 1), ("Non_fractured", 0)]:
    folder = os.path.join(EXAMPLE_DIR, sub)
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder))[:3]:
            EXAMPLES.append([os.path.join(folder, f),
                             list(MODEL_CONFIGS.keys())[2]])


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="🦴 Bone Fracture Detector") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px 0;background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:16px;margin-bottom:16px">
      <h1 style="font-size:2.2em;font-weight:800;color:#C084FC;margin:0">
        🦴 Bone Fracture Detector
      </h1>
      <p style="color:#94A3B8;font-size:1.05em;margin:6px 0 12px 0">
        X-ray fracture classification and detection analysis
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="📤 Upload X-Ray Image")
            model_dd = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=list(MODEL_CONFIGS.keys())[2],
                label="🤖 Select Model",
            )
            btn = gr.Button("🔍  Analyze X-Ray", variant="primary")

        with gr.Column(scale=1):
            label_out   = gr.Label(num_top_classes=2, label="📊 Confidence")
            info_out    = gr.Markdown()

    gradcam_out = gr.Image(type="pil", label="🔥 Grad-CAM Heatmap — Region of Interest")

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[img_in, model_dd],
            label="🖼️ Sample X-Rays (click to load)",
        )

    # Wire up events
    btn.click(predict, [img_in, model_dd], [label_out, info_out, gradcam_out])
    img_in.change(predict, [img_in, model_dd], [label_out, info_out, gradcam_out])

    gr.HTML("""
    <div style="text-align:center;padding:12px;color:#64748B;font-size:.82em;
                border-top:1px solid #1E293B;margin-top:20px">
      Bone Fracture Detection · Dataset: 4,083 X-Rays · 717 Fractured / 3,366 Non-Fractured
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=True)
