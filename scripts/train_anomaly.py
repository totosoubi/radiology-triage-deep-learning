#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from radiotriage import CHEST_LABELS
from radiotriage.data import NormalOnlyDataset, load_chestmnist
from radiotriage.models import ConvAutoencoder
from radiotriage.utils import ensure_dir, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--subset-size", type=int, default=256)
    p.add_argument("--threshold-quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    return p.parse_args()


def reconstruction_errors(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            rec = model(x)
            err = torch.mean((rec - x) ** 2, dim=(1, 2, 3))
            errs.extend(err.cpu().numpy().tolist())
    return np.asarray(errs)


def score_labeled_dataset(model, loader, device):
    model.eval()
    rows = []
    images = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            rec = model(x_dev)
            err = torch.mean((rec - x_dev) ** 2, dim=(1, 2, 3)).cpu().numpy()
            for i, score in enumerate(err):
                row = {"reconstruction_mse": float(score)}
                row.update({label: int(value) for label, value in zip(CHEST_LABELS, y[i].numpy().astype(int))})
                rows.append(row)
                images.append((x[i].clone(), rec[i].cpu(), float(score)))
    return pd.DataFrame(rows), images


def save_reconstruction_grid(model, loader, device, out_path: Path):
    x, _ = next(iter(loader))
    x = x[:6].to(device)
    with torch.no_grad():
        rec = model(x).cpu()
    x = x.cpu()
    tiles = []
    for tensor in list(x) + list(rec):
        arr = tensor[0].numpy()
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
        tiles.append(Image.fromarray(arr, mode="L").resize((96, 96)))
    grid = Image.new("RGB", (6 * 104 + 70, 2 * 112 + 20), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 45), "input", fill="black")
    draw.text((8, 157), "recon", fill="black")
    for row in range(2):
        for col in range(6):
            grid.paste(tiles[row * 6 + col].convert("RGB"), (70 + col * 104, 10 + row * 112))
    grid.save(out_path)


def save_error_histogram(errors: np.ndarray, threshold: float, out_path: Path) -> None:
    width, height = 900, 420
    margin_l, margin_b, margin_t = 70, 60, 30
    plot_w, plot_h = width - margin_l - 30, height - margin_t - margin_b
    hist, edges = np.histogram(errors, bins=24)
    max_count = max(int(hist.max()), 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black")
    for i, count in enumerate(hist):
        x0 = margin_l + int(i * plot_w / len(hist))
        x1 = margin_l + int((i + 0.8) * plot_w / len(hist))
        h = int(count / max_count * plot_h)
        draw.rectangle((x0, margin_t + plot_h - h, x1, margin_t + plot_h), fill="#b45f3a")
    if edges[-1] > edges[0]:
        tx = margin_l + int((threshold - edges[0]) / (edges[-1] - edges[0]) * plot_w)
        draw.line((tx, margin_t, tx, margin_t + plot_h), fill="red", width=2)
    draw.text((10, 10), "Distribution des erreurs de reconstruction AE", fill="black")
    img.save(out_path)


def save_top_anomaly_grid(scored_images: list[tuple[torch.Tensor, torch.Tensor, float]], out_path: Path) -> None:
    top = sorted(scored_images, key=lambda item: item[2], reverse=True)[:6]
    if not top:
        return
    tiles = []
    for x, rec, score in top:
        for tensor in (x, rec):
            arr = tensor[0].numpy()
            arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
            tiles.append(Image.fromarray(arr, mode="L").resize((96, 96)))
    grid = Image.new("RGB", (6 * 112 + 80, 2 * 128 + 32), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 50), "input", fill="black")
    draw.text((8, 178), "recon", fill="black")
    for col, (_, _, score) in enumerate(top):
        draw.text((80 + col * 112, 8), f"{score:.3f}", fill="black")
        grid.paste(tiles[col * 2].convert("RGB"), (80 + col * 112, 28))
        grid.paste(tiles[col * 2 + 1].convert("RGB"), (80 + col * 112, 156))
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    ensure_dir("artifacts")

    train_base = load_chestmnist("train", args.image_size, args.in_channels, args.data_root)
    val_base = load_chestmnist("val", args.image_size, args.in_channels, args.data_root)
    test_base = load_chestmnist("test", args.image_size, args.in_channels, args.data_root)
    train_ds = NormalOnlyDataset(train_base, args.subset_size)
    val_ds = NormalOnlyDataset(val_base, max(32, args.subset_size // 4))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_base, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConvAutoencoder(args.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("anomaly_chestmnist_ae")
    with mlflow.start_run(run_name="conv_ae"):
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), x)
                loss.backward()
                optimizer.step()
                total += loss.item() * x.size(0)
            mlflow.log_metric("train_reconstruction_mse", total / len(train_loader.dataset), step=epoch)
        val_errors = reconstruction_errors(model, val_loader, device)
        threshold = float(np.quantile(val_errors, args.threshold_quantile))
        score_frame, scored_images = score_labeled_dataset(model, test_loader, device)
        score_frame["is_anomaly"] = score_frame["reconstruction_mse"] >= threshold
        mlflow.log_metric("val_error_mean", float(val_errors.mean()))
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_metric("test_anomaly_rate", float(score_frame["is_anomaly"].mean()))
        path = Path("artifacts") / "best_ae.pt"
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "threshold": threshold}, path)
        fig_path = Path("artifacts") / "ae_reconstructions.png"
        hist_path = Path("artifacts") / "ae_error_histogram.png"
        top_path = Path("artifacts") / "ae_top_anomalies.png"
        score_path = Path("artifacts") / "ae_test_scores.csv"
        save_reconstruction_grid(model, val_loader, device, fig_path)
        save_error_histogram(score_frame["reconstruction_mse"].to_numpy(), threshold, hist_path)
        save_top_anomaly_grid(scored_images, top_path)
        score_frame.to_csv(score_path, index=False)
        mlflow.log_artifact(str(path), artifact_path="models")
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
        mlflow.log_artifact(str(hist_path), artifact_path="figures")
        mlflow.log_artifact(str(top_path), artifact_path="figures")
        mlflow.log_artifact(str(score_path), artifact_path="metrics")
        print({"threshold": threshold, "val_error_mean": float(val_errors.mean()), "normal_train_items": len(train_ds)})


if __name__ == "__main__":
    main()
