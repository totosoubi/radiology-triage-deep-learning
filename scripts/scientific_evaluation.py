#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from radiotriage import CHEST_LABELS
from radiotriage.metrics import best_threshold_for_f1, multilabel_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["cnn", "resnet18", "vit"])
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_predictions(model: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path("artifacts") / f"predictions_{model}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    prob_cols = [f"prob_{label}" for label in CHEST_LABELS]
    true_cols = [f"true_{label}" for label in CHEST_LABELS]
    if not set(prob_cols + true_cols).issubset(frame.columns):
        return None
    return frame[true_cols].to_numpy(dtype=int), frame[prob_cols].to_numpy(dtype=float)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]
        threshold, _ = best_threshold_for_f1(yt, yp)
        metrics = multilabel_metrics(yt, yp, threshold=threshold)
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for metric in ["auc_micro", "ap_micro", "f1_macro"]:
        values = frame[metric].dropna()
        if len(values) == 0:
            out[f"{metric}_ci_low"] = float("nan")
            out[f"{metric}_ci_high"] = float("nan")
        else:
            out[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            out[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    return out


def class_thresholds(y_true: np.ndarray, y_prob: np.ndarray, model: str) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(CHEST_LABELS):
        best_t, best_f1 = 0.5, -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            score = f1_score(y_true[:, i], y_prob[:, i] >= threshold, zero_division=0)
            if score > best_f1:
                best_t, best_f1 = float(threshold), float(score)
        rows.append(
            {
                "model": model,
                "label": label,
                "support": int(y_true[:, i].sum()),
                "best_threshold": best_t,
                "best_f1": best_f1,
            }
        )
    return pd.DataFrame(rows)


def draw_curve(points: list[tuple[float, float]], out_path: Path, title: str, x_label: str, y_label: str) -> None:
    width, height = 820, 560
    left, top, right, bottom = 80, 50, 30, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="black")
    draw.text((left, 18), title, fill="black")
    draw.text((left + plot_w // 2 - 30, height - 35), x_label, fill="black")
    draw.text((10, top + plot_h // 2), y_label, fill="black")
    clean = [(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(clean) >= 2:
        xy = []
        for x, y in clean:
            px = left + int(np.clip(x, 0, 1) * plot_w)
            py = top + plot_h - int(np.clip(y, 0, 1) * plot_h)
            xy.append((px, py))
        draw.line(xy, fill="#1f5f99", width=3)
    img.save(out_path)


def save_curves(y_true: np.ndarray, y_prob: np.ndarray, model: str) -> None:
    y_flat = y_true.ravel()
    p_flat = y_prob.ravel()
    precision, recall, _ = precision_recall_curve(y_flat, p_flat)
    fpr, tpr, _ = roc_curve(y_flat, p_flat)
    draw_curve(
        list(zip(recall, precision)),
        Path("artifacts") / f"curve_pr_micro_{model}.png",
        f"Courbe precision-rappel micro - {model}",
        "Recall",
        "Precision",
    )
    draw_curve(
        list(zip(fpr, tpr)),
        Path("artifacts") / f"curve_roc_micro_{model}.png",
        f"Courbe ROC micro - {model}",
        "FPR",
        "TPR",
    )


def main() -> None:
    args = parse_args()
    Path("artifacts").mkdir(exist_ok=True)
    summary_rows = []
    threshold_frames = []
    for model in args.models:
        loaded = load_predictions(model)
        if loaded is None:
            print(f"skip {model}: predictions with true/prob columns not found")
            continue
        y_true, y_prob = loaded
        threshold, _ = best_threshold_for_f1(y_true, y_prob)
        metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
        row = {"model": model, "threshold": threshold, **metrics}
        row.update(bootstrap_ci(y_true, y_prob, args.bootstrap, args.seed))
        summary_rows.append(row)
        threshold_frames.append(class_thresholds(y_true, y_prob, model))
        save_curves(y_true, y_prob, model)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv("artifacts/scientific_summary.csv", index=False)
    if threshold_frames:
        pd.concat(threshold_frames, ignore_index=True).to_csv("artifacts/per_class_thresholds.csv", index=False)
    print(f"Evaluated {len(summary_rows)} models")


if __name__ == "__main__":
    main()
