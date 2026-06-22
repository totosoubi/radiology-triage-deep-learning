from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from radiotriage import CHEST_LABELS
from radiotriage.data import TextVocab, image_transform
from radiotriage.models import ConvAutoencoder, MultimodalClassifier, build_supervised_model


LABEL_FR = {
    "atelectasis": "Atelectasie",
    "cardiomegaly": "Cardiomegalie",
    "effusion": "Epanchement pleural",
    "infiltration": "Infiltration",
    "mass": "Masse",
    "nodule": "Nodule",
    "pneumonia": "Pneumonie",
    "pneumothorax": "Pneumothorax",
    "consolidation": "Consolidation",
    "edema": "Oedeme",
    "emphysema": "Emphyseme",
    "fibrosis": "Fibrose",
    "pleural_thickening": "Epaississement pleural",
    "hernia": "Hernie",
}


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
    return model, ckpt


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


@st.cache_data
def local_openi_images() -> list[str]:
    image_dir = ROOT / "data" / "openi" / "images"
    if not image_dir.exists():
        return []
    return [str(p.relative_to(ROOT)) for p in sorted(image_dir.glob("*.png"))[:80]]


def prediction_frame(probs: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pathologie": CHEST_LABELS,
            "libelle": [LABEL_FR[label] for label in CHEST_LABELS],
            "probabilite": probs,
            "decision": np.where(probs >= threshold, "positif", "negatif"),
        }
    )
    return frame.sort_values("probabilite", ascending=False).reset_index(drop=True)


def run_supervised(image: Image.Image, model, ckpt):
    if model is None:
        return None
    args = ckpt["args"]
    tf = image_transform(args["image_size"], args["in_channels"])
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).numpy()
    threshold = float(ckpt.get("threshold", 0.5))
    return prediction_frame(probs, threshold), threshold


def run_anomaly(image: Image.Image, ae, ckpt):
    if ae is None:
        return None
    args = ckpt["args"]
    tf = image_transform(args["image_size"], args["in_channels"])
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        rec = ae(x)
        score = torch.mean((rec - x) ** 2).item()
    return score, float(ckpt["threshold"])


def run_fusion(image: Image.Image, report: str, model, ckpt):
    if model is None or not report.strip():
        return None
    args = ckpt["args"]
    tf = image_transform(args["image_size"], args["in_channels"])
    x = tf(image).unsqueeze(0)
    tokens = encode_report(report, ckpt["vocab"], args["max_tokens"])
    with torch.no_grad():
        probs = torch.sigmoid(model(x, tokens)).squeeze(0).numpy()
    threshold = float(ckpt.get("threshold", 0.5))
    return prediction_frame(probs, threshold), threshold


def risk_sentence(frame: pd.DataFrame, anomaly_result) -> str:
    positives = frame[frame["decision"].eq("positif")]
    if len(positives) == 0:
        main = "Aucune pathologie ne depasse le seuil retenu."
    else:
        top = positives.iloc[0]
        main = f"Le signal principal est {top['libelle']} avec une probabilite de {top['probabilite']:.2f}."
    if anomaly_result is None:
        return main
    score, threshold = anomaly_result
    if score >= threshold:
        return main + " L'image est aussi consideree atypique par l'autoencodeur."
    return main + " Le score d'anomalie reste sous le seuil appris."


def show_probability_bars(frame: pd.DataFrame, n: int = 8) -> None:
    for _, row in frame.head(n).iterrows():
        st.progress(float(row["probabilite"]), text=f"{row['libelle']} - {row['probabilite']:.3f}")


st.set_page_config(page_title="Tri radiologique", layout="wide")
st.title("Systeme d'aide au tri radiologique")

model, supervised_ckpt = load_supervised()
ae, ae_ckpt = load_ae()
fusion_model, fusion_ckpt = load_fusion()

with st.sidebar:
    st.subheader("Entree")
    sample_images = local_openi_images()
    selected_sample = "Aucune"
    if sample_images:
        selected_sample = st.selectbox("Image OpenI locale", ["Aucune", *sample_images])
    uploaded = st.file_uploader("Importer une radiographie", type=["png", "jpg", "jpeg"])
    report = st.text_area("Compte-rendu", height=150)
    st.caption("Le modele ne remplace pas un avis medical.")

image = None
image_name = None
if uploaded is not None:
    image = Image.open(uploaded).convert("L")
    image_name = uploaded.name
elif selected_sample != "Aucune":
    image = Image.open(ROOT / selected_sample).convert("L")
    image_name = selected_sample

if image is None:
    st.info("Chargez une radiographie ou choisissez une image OpenI locale dans le panneau de gauche.")
    st.stop()

supervised_result = run_supervised(image, model, supervised_ckpt)
anomaly_result = run_anomaly(image, ae, ae_ckpt)
fusion_result = run_fusion(image, report, fusion_model, fusion_ckpt)

if supervised_result is None:
    st.error("Le checkpoint supervise `artifacts/best_supervised.pt` est introuvable.")
    st.stop()

supervised_frame, supervised_threshold = supervised_result

tab_simple, tab_pro, tab_multi, tab_method = st.tabs(
    ["Vue simple", "Vue professionnelle", "Image + texte", "Methodologie"]
)

with tab_simple:
    left, right = st.columns([0.9, 1.1])
    left.image(image, caption=image_name or "Radiographie chargee", use_container_width=True)
    right.subheader("Resume")
    right.write(risk_sentence(supervised_frame, anomaly_result))
    top3 = supervised_frame.head(3)[["libelle", "probabilite", "decision"]].copy()
    top3["probabilite"] = top3["probabilite"].map(lambda x: f"{x:.2f}")
    right.dataframe(top3, hide_index=True, use_container_width=True)
    if anomaly_result is not None:
        score, threshold = anomaly_result
        status = "Atypique" if score >= threshold else "Sous le seuil"
        right.metric("Score d'anomalie", f"{score:.5f}", delta=f"{status} / seuil {threshold:.5f}")
    right.caption("Les probabilites sont des aides au tri et doivent etre relues par un professionnel.")

with tab_pro:
    st.subheader("Predictions supervisees")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modele image", supervised_ckpt["model"])
    c2.metric("Seuil de decision", f"{supervised_threshold:.2f}")
    if anomaly_result is not None:
        c3.metric("MSE anomalie", f"{anomaly_result[0]:.5f}")
    show_probability_bars(supervised_frame, n=14)
    table = supervised_frame.copy()
    table["probabilite"] = table["probabilite"].map(lambda x: f"{x:.4f}")
    st.dataframe(table, hide_index=True, use_container_width=True)

with tab_multi:
    st.subheader("Fusion image + compte-rendu")
    if not report.strip():
        st.info("Ajoutez un compte-rendu dans le panneau de gauche pour lancer la vue multimodale.")
    elif fusion_result is None:
        st.warning("Le checkpoint multimodal `artifacts/best_multimodal_fusion.pt` est introuvable.")
    else:
        fusion_frame, fusion_threshold = fusion_result
        c1, c2 = st.columns([1, 1])
        c1.metric("Seuil fusion", f"{fusion_threshold:.2f}")
        c2.metric("Vocabulaire texte", len(fusion_ckpt["vocab"]))
        show_probability_bars(fusion_frame, n=8)
        compare = supervised_frame[["pathologie", "libelle", "probabilite"]].merge(
            fusion_frame[["pathologie", "probabilite"]],
            on="pathologie",
            suffixes=("_image", "_fusion"),
        )
        compare = compare.sort_values("probabilite_fusion", ascending=False)
        compare["probabilite_image"] = compare["probabilite_image"].map(lambda x: f"{x:.4f}")
        compare["probabilite_fusion"] = compare["probabilite_fusion"].map(lambda x: f"{x:.4f}")
        st.dataframe(compare, hide_index=True, use_container_width=True)

with tab_method:
    st.subheader("Contexte du prototype")
    st.write(
        "Le modele image a ete entraine sur ChestMNIST en classification multi-label. "
        "La detection d'anomalies repose sur un autoencodeur convolutionnel. "
        "La partie multimodale utilise un sous-ensemble OpenI/NLMCXR avec image et compte-rendu."
    )
    st.write(
        "Les scores ne sont pas calibres pour un usage clinique. Ils servent a comparer les briques "
        "du projet et a montrer comment le systeme pourrait aider a prioriser une relecture."
    )
    st.markdown(
        """
        **Artefacts utilises**

        - `artifacts/best_supervised.pt`
        - `artifacts/best_ae.pt`
        - `artifacts/best_multimodal_fusion.pt`
        - `artifacts/mlflow_runs_summary.csv`
        """
    )
