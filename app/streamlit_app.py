"""
Streamlit Web App for Bone Fracture Detection.
Deploy this script easily on Streamlit Community Cloud.
"""

import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import streamlit as st

# Ensure root folder is in path so we can import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import ModelBaseline, ModelImproved, ModelFinal

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = "cpu"  # Force CPU for cloud deployment compatibility
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASSES = ["Non-Fractured", "Fractured"]
IMG_SIZE = 224

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

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    model_cls, filename = MODEL_CONFIGS[model_key]
    ckpt_path = os.path.join(MODEL_DIR, filename)
    model = model_cls()
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

def gradcam_overlay(model, tensor, img_orig, arch_name):
    """Returns Grad-CAM blended PIL image, or None."""
    try:
        if "ResNet" in arch_name:
            target = model.features[-2][-1]
        elif "Efficient" in arch_name:
            target = model.model.features[-1][0]
        else:
            target = model.features[8]

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
        if f is None or g is None: return None

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
        return None

# ── UI Layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bone Fracture Detector", page_icon="🦴", layout="wide")

st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:16px;">
  <h1 style="color:#C084FC;margin:0;">🦴 Bone Fracture Detector</h1>
  <p style="color:#94A3B8;font-size:1.1em;margin-top:10px;">AI-powered X-ray analysis</p>
</div>
""", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    model_choice = st.selectbox("🤖 Select Model to Use", list(MODEL_CONFIGS.keys()), index=2)

with col2:
    if uploaded_file is not None:
        st.subheader("📊 Results")
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("Analyzing image..."):
            model = load_model(model_choice)
            tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                conf, idx = probs.max(0)
                pred_cls = CLASSES[idx.item()]
            
            # Show Metrics
            st.success(f"### Prediction: **{pred_cls}** (Confidence: {conf.item():.1%})")
            st.progress(conf.item())
            
            # Show Grad-CAM Overlay
            st.markdown("#### 🔥 Region of Interest (Grad-CAM)")
            heatmap = gradcam_overlay(model, tensor, image, model_choice)
            if heatmap:
                st.image(heatmap, use_container_width=True)
            else:
                st.image(image, caption="Original Image (Heatmap Generation Failed)", use_container_width=True)
    else:
        st.info("Upload an image on the left to see the prediction and Grad-CAM heatmap.")
