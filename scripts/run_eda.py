#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from radiotriage import CHEST_LABELS
from radiotriage.data import load_chestmnist
from radiotriage.utils import ensure_dir, seed_everything


def collect_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    ds = load_chestmnist(split, image_size=64, in_channels=1, root="data")
    labels, images = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        labels.append(np.asarray(y, dtype="int64"))
        if i < 24:
            arr = x[0].numpy()
            arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
            images.append(arr)
    return np.stack(labels), np.stack(images)


def save_distribution(frame: pd.DataFrame, out_path: Path) -> None:
    width, height = 1200, 560
    margin_l, margin_b, margin_t = 80, 170, 40
    plot_w, plot_h = width - margin_l - 40, height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black")
    colors = {"train": "#4776b4", "val": "#d18b28", "test": "#2f7d59"}
    max_rate = max(frame["prevalence"].max(), 0.01)
    group_w = plot_w / len(CHEST_LABELS)
    for i, label in enumerate(CHEST_LABELS):
        rows = frame[frame["label"] == label]
        for j, (_, row) in enumerate(rows.iterrows()):
            x = int(margin_l + i * group_w + 5 + j * group_w / 4)
            h = int(float(row["prevalence"]) / max_rate * plot_h)
            draw.rectangle((x, margin_t + plot_h - h, x + int(group_w / 5), margin_t + plot_h), fill=colors[row["split"]])
        draw.text((int(margin_l + i * group_w), margin_t + plot_h + 8), label[:14], fill="black")
    draw.text((12, 12), "Prevalence par label et split", fill="black")
    img.save(out_path)


def save_cooccurrence(labels: np.ndarray, out_path: Path) -> None:
    co = labels.T @ labels
    norm = co / np.maximum(np.diag(co)[:, None], 1)
    cell = 42
    left, top = 170, 30
    img = Image.new("RGB", (left + cell * len(CHEST_LABELS) + 20, top + cell * len(CHEST_LABELS) + 180), "white")
    draw = ImageDraw.Draw(img)
    for i, row_label in enumerate(CHEST_LABELS):
        draw.text((8, top + i * cell + 12), row_label[:20], fill="black")
        draw.text((left + i * cell, top + cell * len(CHEST_LABELS) + 8), row_label[:8], fill="black")
        for j in range(len(CHEST_LABELS)):
            value = float(norm[i, j])
            shade = int(255 - 200 * min(value, 1.0))
            draw.rectangle(
                (left + j * cell, top + i * cell, left + (j + 1) * cell - 2, top + (i + 1) * cell - 2),
                fill=(shade, shade, 255),
            )
    draw.text((10, 8), "Cooccurrences normalisees par label source", fill="black")
    img.save(out_path)


def save_sample_grid(images: np.ndarray, out_path: Path) -> None:
    tile, gap = 96, 10
    cols = 6
    rows = int(np.ceil(len(images) / cols))
    img = Image.new("RGB", (cols * (tile + gap) + gap, rows * (tile + gap) + gap), "white")
    for i, arr in enumerate(images):
        tile_img = Image.fromarray(arr, mode="L").resize((tile, tile)).convert("RGB")
        x = gap + (i % cols) * (tile + gap)
        y = gap + (i // cols) * (tile + gap)
        img.paste(tile_img, (x, y))
    img.save(out_path)


def main() -> None:
    seed_everything(42)
    ensure_dir("artifacts")
    rows = []
    split_labels: dict[str, np.ndarray] = {}
    sample_images = None
    for split in ("train", "val", "test"):
        labels, images = collect_split(split)
        split_labels[split] = labels
        if split == "train":
            sample_images = images
        for i, label in enumerate(CHEST_LABELS):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "support": int(labels[:, i].sum()),
                    "prevalence": float(labels[:, i].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    stats_path = Path("artifacts") / "eda_label_stats.csv"
    dist_path = Path("artifacts") / "eda_label_distribution.png"
    co_path = Path("artifacts") / "eda_cooccurrence_train.png"
    sample_path = Path("artifacts") / "eda_train_samples.png"
    frame.to_csv(stats_path, index=False)
    save_distribution(frame, dist_path)
    save_cooccurrence(split_labels["train"], co_path)
    save_sample_grid(sample_images, sample_path)

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("eda_chestmnist")
    with mlflow.start_run(run_name="eda"):
        mlflow.log_metric("train_size", len(split_labels["train"]))
        mlflow.log_metric("val_size", len(split_labels["val"]))
        mlflow.log_metric("test_size", len(split_labels["test"]))
        mlflow.log_artifact(str(stats_path), artifact_path="metrics")
        mlflow.log_artifact(str(dist_path), artifact_path="figures")
        mlflow.log_artifact(str(co_path), artifact_path="figures")
        mlflow.log_artifact(str(sample_path), artifact_path="figures")
    print(f"Wrote {stats_path}, {dist_path}, {co_path}, {sample_path}")


if __name__ == "__main__":
    main()
