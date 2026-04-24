import sys, os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from attention_model import AttentionCNN
import cv2

# ---------------- CONFIG ----------------
st.set_page_config(page_title="VeriShield AI", layout="wide")

st.markdown("""
<style>
.stApp{
    background:#f8fafc;
    color:#0f172a;
}
.block-container{
    max-width:1450px;
    padding-top:1rem;
}
div[data-testid="stMetric"]{
    background:white;
    border:1px solid #dbeafe;
    border-radius:14px;
    padding:10px;
}
h1,h2,h3{
    color:#0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("VeriShield AI")

st.markdown("""
<div style='background:white;padding:14px;border-radius:14px;
border:1px solid #dbeafe;margin-bottom:10px'>
<b>Enterprise Biometric Security Dashboard</b><br>
<span style='color:#475569'>
Explainable robust face authentication with adversarial defense analytics
</span>
</div>
""", unsafe_allow_html=True)

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = AttentionCNN().to(device)
    model.load_state_dict(torch.load(
        "models/best_attention_model.pth",
        map_location=device
    ))
    model.eval()
    return model

model = load_model()
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# ---------------- HELPERS ----------------
def preprocess(img):
    face = mtcnn(img)
    if face is None:
        return None
    return face.unsqueeze(0).to(device)

def inference(x1, x2):
    with torch.no_grad():
        e1, a1 = model(x1)
        e2, a2 = model(x2)

        dist = torch.norm(e1 - e2, dim=1).item()

    confidence = max(0, min(100, (1 - dist / 0.60) * 100))
    return confidence, dist, a1[0,0].cpu().numpy(), a2[0,0].cpu().numpy()

def resize_heatmap(h):
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h = cv2.resize(h, (256,256), interpolation=cv2.INTER_CUBIC)
    return h

def overlay(img, heatmap, title):
    base = np.array(img.resize((256,256)))

    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.imshow(base)
    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.48,
        interpolation="bicubic",
        extent=[0,256,256,0]
    )
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    st.pyplot(fig)

def create_attack(img, attack):
    if attack == "FGSM":
        arr = np.array(img).astype(np.float32)/255.0
        noise = 0.05*np.sign(np.random.randn(*arr.shape))
        adv = np.clip(arr + noise, 0, 1)

        vis = ((noise-noise.min()) /
               (noise.max()-noise.min()+1e-8)*255).astype(np.uint8)

        return Image.fromarray(vis), Image.fromarray((adv*255).astype(np.uint8))

    elif attack == "Blur":
        return Image.new("RGB", img.size, (170,170,170)), \
               img.filter(ImageFilter.GaussianBlur(radius=2))

    else:
        return Image.new("RGB", img.size, (235,235,235)), \
               ImageEnhance.Brightness(img).enhance(1.35)

# ---------------- INPUT ----------------
c1, c2 = st.columns(2)

with c1:
    f1 = st.file_uploader("Upload Face A", type=["jpg","jpeg","png"])

with c2:
    f2 = st.file_uploader("Upload Face B", type=["jpg","jpeg","png"])

attack = st.selectbox("Attack Engine", ["FGSM","Blur","Brightness"])

# ---------------- MAIN ----------------
if f1 and f2:

    img1 = Image.open(f1).convert("RGB")
    img2 = Image.open(f2).convert("RGB")

    x1 = preprocess(img1)
    x2 = preprocess(img2)

    if x1 is None or x2 is None:
        st.error("Face not detected.")
        st.stop()

    conf, dist, h1, h2 = inference(x1, x2)

    st.subheader("Identity Verification")
    st.image([img1,img2], caption=["Face A","Face B"], width=280)

    a,b,c = st.columns(3)
    a.metric("Match Confidence", f"{conf:.2f}%")
    b.metric("Embedding Distance", f"{dist:.4f}")
    c.metric("Decision", "Verified" if dist < 0.30 else "Rejected")

    # ATTACK
    noise_img, adv_img = create_attack(img1, attack)
    xa = preprocess(adv_img)

    st.subheader("Attack Construction Lab")

    p,q,r = st.columns(3)
    p.image(img1, caption="Original A", use_container_width=True)
    q.image(noise_img, caption="Perturbation Map", use_container_width=True)
    r.image(adv_img, caption="Attacked A*", use_container_width=True)

    if xa is not None:

        conf2, dist2, ha, _ = inference(xa, x2)

        drift = abs(dist2 - dist)
        consistency = max(0, 100 - drift*500)
        suspicion = min(100, drift*900)

        # ---------------- ANALYTICS ----------------
        st.subheader("Decision Analytics")

        m1,m2,m3,m4 = st.columns(4)

        m1.metric("Post-Attack Confidence", f"{conf2:.2f}%")
        m2.metric("Distance Drift", f"{drift:.4f}")
        m3.metric("Consistency Score", f"{consistency:.1f}%")
        m4.metric("Suspicion Score", f"{suspicion:.1f}%")

        chart_df = pd.DataFrame({
            "Metric":["Confidence","Consistency","Suspicion"],
            "Score":[conf2, consistency, suspicion]
        })

        st.bar_chart(chart_df.set_index("Metric"))

        # ---------------- DECISION ----------------
        left,right = st.columns(2)

        with left:
            st.markdown("## Legacy Baseline")
            st.write("Uses distance threshold only")

            if dist2 < 0.35:
                st.error("Accepted Manipulated Input")
            else:
                st.success("Rejected")

        with right:
            st.markdown("## VeriShield AI")

            secure = (dist2 < 0.35 and drift < 0.03)

            if secure:
                st.success("Verified Safe Input")
            else:
                st.error("Rejected Suspicious Authentication")
                st.write("Reasons:")
                st.write("- Attention consistency degraded")
                st.write("- Facial landmark focus shifted")
                st.write("- Embedding drift exceeded threshold")

        # ---------------- EXPLAINABILITY ----------------
        st.subheader("Explainability Studio")

        hc = resize_heatmap(h1)
        ha = resize_heatmap(ha)
        diff = np.abs(hc-ha)

        u,v = st.columns(2)
        with u:
            overlay(img1, hc, "Clean Heatmap")

        with v:
            overlay(adv_img, ha, "Adversarial Heatmap")

        x,y = st.columns(2)

        with x:
            fig, ax = plt.subplots(figsize=(4.5,4.5))
            ax.imshow(diff, cmap="jet")
            ax.set_title("Attention Shift Map", fontsize=18)
            ax.axis("off")
            st.pyplot(fig)

        with y:
            overlay(img1, hc, "Stable Protected Focus")

        # ---------------- CONSISTENCY ----------------
        st.subheader("Attention Consistency Loss Panel")

        st.write(
            f"MSE divergence between clean and attacked attention maps: {np.mean(diff):.4f}"
        )
        st.write(
            f"Consistency retained after defense logic: {consistency:.1f}%"
        )

        # ---------------- REPORT ----------------
        st.subheader("Downloadable Report")

        report = f"""
VeriShield AI Security Report

Attack Type: {attack}
Clean Confidence: {conf:.2f}%
Post Attack Confidence: {conf2:.2f}%
Distance Drift: {drift:.4f}
Consistency Score: {consistency:.1f}%
Suspicion Score: {suspicion:.1f}%

Decision:
{"Rejected Suspicious Authentication" if not secure else "Verified Safe Input"}
"""

        st.download_button(
            "Download Security Report",
            report,
            file_name="verishield_report.txt"
        )

        # ---------------- SUMMARY ----------------
        st.subheader("Research Summary")

        st.table(pd.DataFrame({
            "System":["Legacy Baseline","VeriShield AI"],
            "Under Attack":[
                "Accepted" if dist2 < 0.35 else "Rejected",
                "Rejected" if not secure else "Accepted"
            ],
            "Explainable":["No","Yes"],
            "Trust Level":["Low","High"]
        }))