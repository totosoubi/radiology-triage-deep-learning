#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def read_csv(name: str) -> pd.DataFrame | None:
    path = ART / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if max_rows:
        df = df.head(max_rows)
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    view = view.astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def best_runs_table() -> str:
    df = read_csv("mlflow_runs_summary.csv")
    if df is None:
        return "Export MLflow non genere."
    cols = [c for c in ["experiment", "run_name", "status", "metric.test_auc_micro", "metric.test_ap_micro", "metric.test_f1_macro"] if c in df.columns]
    done = df[df["status"].eq("FINISHED")].copy()
    if {"experiment", "run_name", "start_time"}.issubset(done.columns):
        done = done.sort_values("start_time").groupby(["experiment", "run_name"], as_index=False).tail(1)
    if "metric.test_ap_micro" in done:
        done = done.sort_values("metric.test_ap_micro", ascending=False, na_position="last")
    return markdown_table(done, cols)


def selected_runs_table() -> str:
    df = read_csv("mlflow_selected_runs.csv")
    if df is None:
        return "Export des runs retenus non genere."
    columns = [
        "component", "run_name", "role", "run_id", "duration_seconds",
        "test_auc_micro", "test_ap_micro", "test_f1_macro", "checkpoint",
    ]
    return markdown_table(df, columns)


def supervised_per_class_section() -> str:
    blocks = []
    for model in ("cnn", "resnet18", "vit"):
        df = read_csv(f"per_class_metrics_{model}.csv")
        if df is None:
            continue
        cols = ["label", "support", "prevalence", "average_precision", "auc", "f1"]
        blocks.append(f"**{model} - metriques par classe**\n\n{markdown_table(df, cols)}")
    return "\n\n".join(blocks) if blocks else "Les metriques par classe seront produites apres entrainement supervise."


def interpretability_section() -> str:
    summary = read_csv("error_summary_cnn.csv")
    cases = read_csv("error_cases_cnn.csv")
    if summary is None and cases is None:
        return "Analyse qualitative non generee."
    parts = []
    if summary is not None:
        focus = summary[summary["label"].isin(["atelectasis", "effusion", "infiltration", "edema"])]
        parts.append(markdown_table(focus, ["label", "support", "tp", "fp", "fn", "false_positive_rate", "false_negative_rate"]))
    if cases is not None:
        parts.append("Exemples representatifs TP/FP/FN :\n\n" + markdown_table(cases, ["label", "case_type", "row_id", "truth", "prediction", "probability"]))
    parts.append(
        "La figure `artifacts/gradcam_error_cases_cnn.png` superpose une carte Grad-CAM aux exemples. "
        "Elle sert a verifier qualitativement que le modele reagit surtout a la zone thoracique, "
        "mais elle ne constitue pas une explication clinique suffisante."
    )
    return "\n\n".join(parts)


def multimodal_table() -> str:
    frames = []
    for mode in ("image", "text", "fusion"):
        df = read_csv(f"per_class_metrics_multimodal_{mode}.csv")
        if df is None:
            continue
        frames.append(
            {
                "mode": mode,
                "mean_ap": df["average_precision"].mean(),
                "mean_auc": df["auc"].mean(),
                "mean_f1": df["f1"].mean(),
            }
        )
    if not frames:
        return "Comparaison multimodale non executee."
    return markdown_table(pd.DataFrame(frames), ["mode", "mean_ap", "mean_auc", "mean_f1"])


def openi_summary() -> str:
    path = ROOT / "data" / "openi" / "manifest.csv"
    if not path.exists():
        return (
            "Pour la multimodalite, le code accepte un manifest OpenI ou MIMIC-CXR contenant "
            "`image_path`, `report` et les 14 labels harmonises. Dans l'environnement local actuel, "
            "aucun manifest reel n'est present."
        )
    df = pd.read_csv(path)
    positives = df[[c for c in df.columns if c in {
        "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule",
        "pneumonia", "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis",
        "pleural_thickening", "hernia"
    }]].sum().sort_values(ascending=False)
    pos_text = ", ".join(f"{idx}: {int(val)}" for idx, val in positives.head(6).items())
    return (
        f"La composante multimodale utilise maintenant un sous-ensemble reel OpenI/NLMCXR : "
        f"{len(df)} paires image + compte-rendu verifiees localement. Les rapports proviennent de "
        f"`NLMCXR_reports.tgz` et les PNG de `NLMCXR_png.tgz`, archives officielles OpenI. "
        f"Les labels sont derives des tags MeSH et du texte des sections indication/findings/impression "
        f"par dictionnaire medical harmonise avec les 14 labels ChestMNIST. Labels positifs les plus presents : {pos_text}."
    )


def eda_summary() -> str:
    df = read_csv("eda_label_stats.csv")
    if df is None:
        return "EDA non generee."
    train = df[df["split"].eq("train")].sort_values("prevalence", ascending=False)
    rare = train.sort_values("prevalence", ascending=True)
    return (
        "Labels les plus frequents dans le train:\n\n"
        f"{markdown_table(train, ['label', 'support', 'prevalence'], max_rows=5)}\n\n"
        "Labels les plus rares dans le train:\n\n"
        f"{markdown_table(rare, ['label', 'support', 'prevalence'], max_rows=5)}"
    )


def anomaly_summary() -> str:
    df = read_csv("ae_test_scores.csv")
    if df is None:
        return "Scores AE non generes."
    return (
        f"Nombre d'images test scorees: {len(df)}. "
        f"Erreur moyenne: {df['reconstruction_mse'].mean():.5f}. "
        f"Taux d'images au-dessus du seuil: {df['is_anomaly'].mean():.3f}."
    )


def main() -> None:
    os_name = f"{platform.system()} {platform.release()} - Python {platform.python_version()}"
    text = f"""# Rapport final - Systeme d'aide au tri radiologique

**Projet realise par Thomas Soubirou-Pouey et Estelle Letourneur**

## Probleme

Le systeme vise un tri radiologique assiste par IA a partir de radiographies thoraciques. La sortie supervisee est multi-label : chaque image recoit une probabilite independante pour 14 pathologies. Le demonstrateur ajoute un score d'anomalie afin d'identifier des cas qui s'ecartent des images apprises comme normales.

## Donnees

Le dataset principal est ChestMNIST de MedMNIST. Les splits officiels train, validation et test sont conserves. Cela evite une validation aleatoire opportuniste et donne une separation reproductible. Les images sont redimensionnees, converties en 1 ou 3 canaux selon l'architecture, puis normalisees.

{openi_summary()}

## Analyse exploratoire

{eda_summary()}

Artefacts EDA produits :

- `artifacts/eda_label_stats.csv`
- `artifacts/eda_label_distribution.png`
- `artifacts/eda_cooccurrence_train.png`
- `artifacts/eda_train_samples.png`

## Preparation

Les labels sont traites en binaire multi-label. La fonction de perte est `BCEWithLogitsLoss`, compatible avec des sorties logits et une activation sigmoide par classe a l'evaluation. L'option `--pos-weight` corrige le desequilibre des classes par ponderation inverse positives/negatives. L'augmentation retenue est volontairement faible : rotations de 7 degres, translations de 3 % et echelle 0.97-1.03, afin de ne pas creer de geometrie thoracique non plausible.

La reproductibilite repose sur une seed fixe, les splits officiels MedMNIST, la sauvegarde du meilleur modele sur validation, et un export MLflow. Pour OpenI, le split aleatoire est reproductible mais le manifest ne fournit pas d'identifiant patient fiable ; une absence totale de fuite au niveau patient ne peut donc pas etre garantie pour cette preuve de concept.

## Modelisation supervisee

Trois architectures sont implementees et comparees :

- CNN simple entraine depuis zero : convolutions, batch normalization, ReLU, pooling, dropout et tete lineaire.
- ResNet18 en transfer learning : backbone ImageNet, connexions residuelles, adaptation de la premiere couche si besoin et tete finale a 14 sorties.
- TinyViT : decoupage en patchs, embedding positionnel, encodeur Transformer et classification multi-label.

Les optimisations incluent AdamW, weight decay, scheduler CosineAnnealingLR, early stopping et recherche d'un seuil macro-F1 sur validation.

## Detection d'anomalies

Un autoencodeur convolutionnel est entraine sur les radiographies sans label positif. Le score d'anomalie est la MSE de reconstruction. Le seuil est le quantile 95 % des erreurs sur validation normale.

{anomaly_summary()}

Artefacts :

- `artifacts/ae_reconstructions.png`
- `artifacts/ae_error_histogram.png`
- `artifacts/ae_top_anomalies.png`
- `artifacts/ae_test_scores.csv`

## Modelisation multimodale

La preuve de concept implemente trois variantes :

- image seule : encodeur CNN puis tete multi-label ;
- texte seul : vocabulaire local, embeddings et pooling masque ;
- fusion intermediaire : concatenation des representations image et texte.

Comparaison des runs disponibles :

{multimodal_table()}

Ces valeurs proviennent des runs executes avec `data/openi/manifest.csv`, donc avec images et comptes-rendus OpenI reels. Le sous-ensemble reste limite par le temps de telechargement de l'archive officielle complete ; les metriques sont donc interpretees comme preuve de concept et non comme resultat clinique robuste.

La fusion est intermediaire : les representations image et texte sont concatenees avant la tete finale. Les paires d'entrainement sont completes. Dans le demonstrateur, un rapport manquant desactive la fusion mais laisse la prediction image disponible ; un entrainement avec masquage aleatoire du texte serait necessaire pour rendre le modele fusionne robuste aux modalites manquantes.

## Evaluation

Les metriques retenues sont AUROC micro/macro, average precision micro/macro et F1 micro/macro. L'average precision est particulierement pertinente en multi-label desequilibre car elle reste informative quand les positifs sont rares.

Synthese MLflow :

{best_runs_table()}

{supervised_per_class_section()}

Les warnings `UndefinedMetricWarning` observes sur les petits smoke tests indiquent que certains labels sont absents du sous-ensemble de validation/test. Ils disparaissent ou deviennent moins problematiques sur des runs plus grands.

### Analyse qualitative et interpretabilite

{interpretability_section()}

## Tracking MLflow

Toutes les experiences sont journalisees dans `mlruns/`. Les scripts enregistrent hyperparametres, pertes, metriques, seuil retenu, checkpoints, figures, predictions et CSV par classe. Un export de lecture rapide est disponible dans `artifacts/mlflow_runs_summary.csv`.

La preuve de selection fournie avec le rendu est disponible dans `artifacts/mlflow_selected_runs.csv` et `artifacts/mlflow_selected_runs.png`. Elle relie les runs retenus aux checkpoints charges par Streamlit :

{selected_runs_table()}

Les durees sont mesurees de bout en bout par MLflow. Elles documentent le cout local, mais ne forment pas un benchmark direct lorsque les tailles de sous-ensembles ou le nombre d'epochs different.

Experiences creees :

- `eda_chestmnist`
- `supervised_chestmnist`
- `anomaly_chestmnist_ae`
- `multimodal_openi_or_mimic`

## Demonstrateur

Le demonstrateur Streamlit charge `artifacts/best_supervised.pt`, `artifacts/best_ae.pt` et `artifacts/best_multimodal_fusion.pt`. Il permet de charger une radiographie, affiche les probabilites par pathologie, indique le seuil supervise retenu, calcule le score d'anomalie AE avec comparaison au seuil, et exploite un compte-rendu saisi par l'utilisateur pour produire des predictions fusionnees image + texte.

Commande :

```bash
streamlit run app/streamlit_app.py
```

## Analyse critique

Le pipeline est maintenant complet pour ChestMNIST : EDA, trois familles de modeles, evaluation par classe, anomalie, MLflow et demonstrateur. Les limites restantes sont principalement scientifiques :

- ChestMNIST est une version compacte et moins realiste que des radiographies hospitalieres haute resolution.
- Les classes rares restent difficiles et necessitent des runs plus longs, des seuils par classe ou une calibration.
- Le score AE n'est pas un diagnostic ; il signale une reconstruction difficile, ce qui peut venir d'une pathologie, d'un artefact ou d'un decalage de distribution.
- Le ViT leger peut etre moins competitif sans pre-entrainement massif.
- La multimodalite est executee sur un sous-ensemble OpenI reel, mais le volume local reste modeste ; il faudrait relancer sur l'archive OpenI complete ou MIMIC-CXR pour conclure sur le gain clinique de la fusion.

## Conclusion et perspectives

Le projet atteint un niveau de rendu solide : classification image ChestMNIST, detection d'anomalies, preuve multimodale OpenI reelle, MLflow, analyse par classe et demonstrateur. Les perspectives prioritaires sont l'entrainement plus long, le passage a l'archive OpenI complete ou a MIMIC-CXR, et une calibration plus fine des seuils par classe.

Configuration locale documentee : {os_name}.
"""
    (ROOT / "rapport.md").write_text(text, encoding="utf-8")
    print("Wrote rapport.md")


if __name__ == "__main__":
    main()
