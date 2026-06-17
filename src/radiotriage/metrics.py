from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {}

    for average in ("micro", "macro"):
        try:
            metrics[f"auc_{average}"] = float(roc_auc_score(y_true, y_prob, average=average))
        except ValueError:
            metrics[f"auc_{average}"] = float("nan")
        try:
            metrics[f"ap_{average}"] = float(average_precision_score(y_true, y_prob, average=average))
        except ValueError:
            metrics[f"ap_{average}"] = float("nan")

    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["positive_rate"] = float(y_true.mean())
    return metrics


def per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], threshold: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, label in enumerate(labels):
        row = {
            "label": label,
            "support": int(y_true[:, i].sum()),
            "prevalence": float(y_true[:, i].mean()),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "average_precision": float("nan"),
            "auc": float("nan"),
        }
        try:
            row["average_precision"] = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            pass
        try:
            row["auc"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, y_prob >= threshold, average="macro", zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1
