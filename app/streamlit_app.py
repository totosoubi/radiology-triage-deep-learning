from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import streamlit as st
import torch
from PIL import Image

from radiotriage import CHEST_LABELS
from radiotriage.data import TextVocab, image_transform
from radiotriage.models import ConvAutoencoder, MultimodalClassifier, build_supervised_model


@st.cache_resource
def load_supervised():
    path = ROOT / "artifacts" / "best_supervised.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    model = build_supervised_model(
        args["model"],
        len(CHEST_LABELS),
        args["in_channels"],
        args["image_size"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, args


@st.cache_resource
def load_ae():
    path = ROOT / "artifacts" / "best_ae.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    model = ConvAutoencoder(args["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_resource
def load_fusion():
    path = ROOT / "artifacts" / "best_multimodal_fusion.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    vocab = ckpt["vocab"]
    model = MultimodalClassifier(
        mode="fusion",
        vocab_size=len(vocab),
        num_classes=len(CHEST_LABELS),
        in_channels=args["in_channels"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def encode_report(report_text: str, stoi: dict[str, int], max_len: int) -> torch.Tensor:
    ids = [stoi.get(token, 1) for token in TextVocab.tokenize(report_text)[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


st.set_page_config(page_title="Tri radiologique", layout="wide")
st.title("Systeme d'aide au tri radiologique")

uploaded = st.file_uploader("Radiographie thoracique", type=["png", "jpg", "jpeg"])
report = st.text_area("Compte-rendu ou metadonnees optionnelles", height=120)

model, model_args = load_supervised()
ae, ae_ckpt = load_ae()
fusion_model, fusion_ckpt = load_fusion()

if uploaded is None:
    st.info("Chargez une image pour obtenir les predictions et le score d'anomalie.")
    st.stop()

image = Image.open(uploaded).convert("L")
left, right = st.columns([1, 1])
left.image(image, caption="Image chargee", use_container_width=True)

if model is None:
    right.warning("Aucun modele supervise trouve dans artifacts/best_supervised.pt.")
else:
    tf = image_transform(model_args["image_size"], model_args["in_channels"])
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).numpy()
    threshold = float(torch.load(ROOT / "artifacts" / "best_supervised.pt", map_location="cpu").get("threshold", 0.5))
    order = np.argsort(-probs)
    right.subheader("Predictions supervisees")
    right.caption(f"Seuil de decision retenu: {threshold:.2f}")
    for idx in order:
        marker = "positif" if probs[idx] >= threshold else "negatif"
        right.progress(float(probs[idx]), text=f"{CHEST_LABELS[idx]}: {probs[idx]:.3f} ({marker})")

if ae is None:
    st.warning("Aucun autoencodeur trouve dans artifacts/best_ae.pt.")
else:
    tf = image_transform(ae_ckpt["args"]["image_size"], ae_ckpt["args"]["in_channels"])
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        rec = ae(x)
        score = torch.mean((rec - x) ** 2).item()
    threshold = ae_ckpt["threshold"]
    st.metric("Score d'anomalie AE", f"{score:.5f}", delta=f"seuil {threshold:.5f}")
    st.write("Statut:", "atypique" if score >= threshold else "dans la distribution normale apprise")

if report.strip():
    if fusion_model is None:
        st.warning("Aucun modele multimodal fusion trouve dans artifacts/best_multimodal_fusion.pt.")
    else:
        args = fusion_ckpt["args"]
        tf = image_transform(args["image_size"], args["in_channels"])
        x = tf(image).unsqueeze(0)
        tokens = encode_report(report, fusion_ckpt["vocab"], args["max_tokens"])
        with torch.no_grad():
            probs = torch.sigmoid(fusion_model(x, tokens)).squeeze(0).numpy()
        st.subheader("Predictions multimodales image + compte-rendu")
        order = np.argsort(-probs)
        cols = st.columns(2)
        for rank, idx in enumerate(order[:8]):
            cols[rank % 2].progress(float(probs[idx]), text=f"{CHEST_LABELS[idx]}: {probs[idx]:.3f}")
