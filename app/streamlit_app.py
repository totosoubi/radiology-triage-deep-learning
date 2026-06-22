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
    manifest = load_openi_manifest()
    if manifest is not None:
        return manifest["image_path"].head(80).tolist()
    image_dir = ROOT / "data" / "openi" / "images"
    if not image_dir.exists():
        return []
    return [str(p.relative_to(ROOT)) for p in sorted(image_dir.glob("*.png"))[:80]]


@st.cache_data
def load_openi_manifest():
    path = ROOT / "data" / "openi" / "manifest.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_mlflow_summary():
    path = ROOT / "artifacts" / "mlflow_runs_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_per_class_metrics(model_name: str):
    path = ROOT / "artifacts" / f"per_class_metrics_{model_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


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
    return score, float(ckpt["threshold"]), tensor_to_image(x[0]), tensor_to_image(rec[0]), error_image(x[0], rec[0])


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


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu()[0].numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


def error_image(x: torch.Tensor, rec: torch.Tensor) -> Image.Image:
    err = torch.abs(x.detach().cpu()[0] - rec.detach().cpu()[0]).numpy()
    if err.max() > 0:
        err = err / err.max()
    arr = (err * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


def risk_sentence(frame: pd.DataFrame, anomaly_result) -> str:
    positives = frame[frame["decision"].eq("positif")]
    if len(positives) == 0:
        main = "Aucune pathologie ne depasse le seuil retenu."
    else:
        top = positives.iloc[0]
        main = f"Le signal principal est {top['libelle']} avec une probabilite de {top['probabilite']:.2f}."
    if anomaly_result is None:
        return main
    score, threshold = anomaly_result[:2]
    if score >= threshold:
        return main + " L'image est aussi consideree atypique par l'autoencodeur."
    return main + " Le score d'anomalie reste sous le seuil appris."


def show_probability_bars(frame: pd.DataFrame, n: int = 8) -> None:
    for _, row in frame.head(n).iterrows():
        st.progress(float(row["probabilite"]), text=f"{row['libelle']} - {row['probabilite']:.3f}")


def update_decisions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    updated = frame.copy()
    updated["decision"] = np.where(updated["probabilite"] >= threshold, "positif", "negatif")
    return updated


def case_markdown(image_name: str, frame: pd.DataFrame, anomaly_result, report: str, fusion_result) -> str:
    top = frame.head(5)
    lines = [
        "# Synthese du cas",
        "",
        f"- Image: `{image_name}`",
        "- Type: radiographie thoracique",
        "",
        "## Predictions image",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['libelle']}: {row['probabilite']:.3f} ({row['decision']})")
    if anomaly_result is not None:
        score, threshold = anomaly_result[:2]
        status = "atypique" if score >= threshold else "sous le seuil"
        lines.extend(["", "## Anomalie", f"- Score AE: {score:.5f}", f"- Seuil: {threshold:.5f}", f"- Statut: {status}"])
    if report.strip():
        lines.extend(["", "## Compte-rendu saisi", report.strip()])
    if fusion_result is not None:
        fusion_frame, _ = fusion_result
        lines.extend(["", "## Top predictions fusion image + texte"])
        for _, row in fusion_frame.head(5).iterrows():
            lines.append(f"- {row['libelle']}: {row['probabilite']:.3f}")
    lines.extend(["", "Note: prototype pedagogique, pas un dispositif medical."])
    return "\n".join(lines)


st.set_page_config(page_title="Tri radiologique", layout="wide")
st.title("Systeme d'aide au tri radiologique")

model, supervised_ckpt = load_supervised()
ae, ae_ckpt = load_ae()
fusion_model, fusion_ckpt = load_fusion()
manifest = load_openi_manifest()

with st.sidebar:
    st.subheader("Entree")
    sample_images = local_openi_images()
    selected_sample = "Aucune"
    if sample_images:
        selected_sample = st.selectbox("Image OpenI locale", ["Aucune", *sample_images])
    uploaded = st.file_uploader("Importer une radiographie", type=["png", "jpg", "jpeg"])
    sample_report = ""
    if manifest is not None and selected_sample != "Aucune":
        row = manifest[manifest["image_path"].eq(selected_sample)]
        if not row.empty:
            sample_report = str(row.iloc[0]["report"])
    use_openi_report = st.checkbox("Utiliser le compte-rendu OpenI", value=bool(sample_report))
    report_default = sample_report if use_openi_report else ""
    report = st.text_area("Compte-rendu", value=report_default, height=150, key=f"report_{selected_sample}_{use_openi_report}")
    st.subheader("Affichage")
    top_n = st.slider("Nombre de pathologies affichees", 3, 14, 8)
    threshold_override = st.slider("Seuil image", 0.05, 0.95, 0.50, 0.05)
    fusion_threshold_override = st.slider("Seuil fusion", 0.05, 0.95, 0.50, 0.05)
    st.caption("Le modele ne remplace pas un avis medical.")

image = None
image_name = None
if uploaded is not None:
    image = Image.open(uploaded).convert("L")
    image_name = uploaded.name
elif selected_sample != "Aucune":
    image_path = ROOT / "data" / "openi" / selected_sample if not selected_sample.startswith("data/") else ROOT / selected_sample
    image = Image.open(image_path).convert("L")
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
supervised_frame = update_decisions(supervised_frame, threshold_override)
supervised_threshold = threshold_override
if fusion_result is not None:
    fusion_frame_raw, _ = fusion_result
    fusion_result = (update_decisions(fusion_frame_raw, fusion_threshold_override), fusion_threshold_override)

case_md = case_markdown(image_name or "image importee", supervised_frame, anomaly_result, report, fusion_result)

tab_simple, tab_pro, tab_anomaly, tab_multi, tab_perf, tab_method = st.tabs(
    ["Vue simple", "Vue professionnelle", "Anomalie", "Image + texte", "Performances", "Methodologie"]
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
        score, threshold = anomaly_result[:2]
        status = "Atypique" if score >= threshold else "Sous le seuil"
        right.metric("Score d'anomalie", f"{score:.5f}", delta=f"{status} / seuil {threshold:.5f}")
    right.caption("Les probabilites sont des aides au tri et doivent etre relues par un professionnel.")
    right.download_button("Exporter la synthese Markdown", case_md, file_name="synthese_cas.md", mime="text/markdown")
    right.download_button(
        "Exporter les predictions CSV",
        supervised_frame.to_csv(index=False),
        file_name="predictions_image.csv",
        mime="text/csv",
    )

with tab_pro:
    st.subheader("Predictions supervisees")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modele image", supervised_ckpt["model"])
    c2.metric("Seuil de decision", f"{supervised_threshold:.2f}")
    if anomaly_result is not None:
        c3.metric("MSE anomalie", f"{anomaly_result[0]:.5f}")
    show_probability_bars(supervised_frame, n=top_n)
    table = supervised_frame.copy()
    table["probabilite"] = table["probabilite"].map(lambda x: f"{x:.4f}")
    st.dataframe(table, hide_index=True, use_container_width=True)
    metrics = load_per_class_metrics(supervised_ckpt["model"])
    if metrics is not None:
        st.subheader("Metriques de reference par classe")
        metric_view = metrics[["label", "support", "average_precision", "auc", "f1"]].copy()
        metric_view.insert(1, "libelle", metric_view["label"].map(LABEL_FR))
        st.dataframe(metric_view, hide_index=True, use_container_width=True)

with tab_anomaly:
    st.subheader("Detection d'anomalies")
    if anomaly_result is None:
        st.warning("Le checkpoint AE `artifacts/best_ae.pt` est introuvable.")
    else:
        score, threshold, input_img, rec_img, err_img = anomaly_result
        c1, c2, c3 = st.columns(3)
        c1.metric("Score MSE", f"{score:.5f}")
        c2.metric("Seuil appris", f"{threshold:.5f}")
        c3.metric("Statut", "Atypique" if score >= threshold else "Sous le seuil")
        img_cols = st.columns(3)
        img_cols[0].image(input_img, caption="Image pretraitee", use_container_width=True)
        img_cols[1].image(rec_img, caption="Reconstruction AE", use_container_width=True)
        img_cols[2].image(err_img, caption="Carte d'erreur", use_container_width=True)
        st.caption("La carte d'erreur montre les zones que l'autoencodeur reconstruit le moins bien.")

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
        show_probability_bars(fusion_frame, n=top_n)
        compare = supervised_frame[["pathologie", "libelle", "probabilite"]].merge(
            fusion_frame[["pathologie", "probabilite"]],
            on="pathologie",
            suffixes=("_image", "_fusion"),
        )
        compare = compare.sort_values("probabilite_fusion", ascending=False)
        compare["probabilite_image"] = compare["probabilite_image"].map(lambda x: f"{x:.4f}")
        compare["probabilite_fusion"] = compare["probabilite_fusion"].map(lambda x: f"{x:.4f}")
        st.dataframe(compare, hide_index=True, use_container_width=True)
        st.download_button(
            "Exporter la comparaison CSV",
            compare.to_csv(index=False),
            file_name="comparaison_image_fusion.csv",
            mime="text/csv",
        )

with tab_perf:
    st.subheader("Runs et artefacts")
    summary = load_mlflow_summary()
    if summary is None:
        st.info("Aucun export MLflow disponible.")
    else:
        cols = [
            "experiment",
            "run_name",
            "status",
            "metric.test_auc_micro",
            "metric.test_ap_micro",
            "metric.test_f1_macro",
        ]
        cols = [c for c in cols if c in summary.columns]
        view = summary[cols].tail(12).copy()
        st.dataframe(view, hide_index=True, use_container_width=True)
    st.subheader("Fichiers utiles")
    for path in [
        ROOT / "artifacts" / "mlflow_runs_summary.csv",
        ROOT / "artifacts" / "eda_label_distribution.png",
        ROOT / "artifacts" / "ae_error_histogram.png",
    ]:
        st.write(f"`{path.relative_to(ROOT)}`", "present" if path.exists() else "absent")

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
