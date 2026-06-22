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
from radiotriage.data import limit_dataset, load_chestmnist
from radiotriage.metrics import best_threshold_for_f1, multilabel_metrics, per_class_metrics
from radiotriage.models import build_supervised_model
from radiotriage.utils import count_parameters, ensure_dir, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18", "vit"], required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--data-root", default="data")
    return p.parse_args()


def collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            ys.append(y.numpy().astype("float32"))
            ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def labels_from_dataset(dataset) -> np.ndarray:
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(np.asarray(y, dtype="float32"))
    return np.stack(labels)


def save_per_class_bars(frame: pd.DataFrame, metric: str, out_path: Path) -> None:
    values = frame[metric].fillna(0).to_numpy()
    width, height = 1100, 520
    margin_l, margin_b, margin_t = 80, 170, 30
    plot_w, plot_h = width - margin_l - 30, height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black")
    bar_w = max(8, plot_w // (len(values) * 2))
    for i, (label, value) in enumerate(zip(frame["label"], values)):
        x = margin_l + int((i + 0.25) * plot_w / len(values))
        h = int(float(value) * plot_h)
        draw.rectangle((x, margin_t + plot_h - h, x + bar_w, margin_t + plot_h), fill="#2f7d59")
        draw.text((x - 18, margin_t + plot_h + 8), str(label)[:14], fill="black")
    draw.text((10, 20), f"Per-class {metric}", fill="black")
    img.save(out_path)


def save_label_distribution(y_true: np.ndarray, out_path: Path) -> None:
    rates = y_true.mean(axis=0)
    width, height = 1100, 520
    margin_l, margin_b, margin_t = 80, 170, 30
    plot_w, plot_h = width - margin_l - 30, height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black")
    bar_w = max(8, plot_w // (len(rates) * 2))
    max_rate = max(float(rates.max()), 0.01)
    for i, (label, rate) in enumerate(zip(CHEST_LABELS, rates)):
        x = margin_l + int((i + 0.25) * plot_w / len(rates))
        h = int((float(rate) / max_rate) * plot_h)
        draw.rectangle((x, margin_t + plot_h - h, x + bar_w, margin_t + plot_h), fill="#4776b4")
        draw.text((x - 18, margin_t + plot_h + 8), label[:14], fill="black")
    draw.text((10, 20), "Positive rate per ChestMNIST label", fill="black")
    img.save(out_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    ensure_dir("artifacts")

    train_ds = limit_dataset(
        load_chestmnist("train", args.image_size, args.in_channels, args.data_root, augment=args.augment),
        args.subset_size,
        args.seed,
    )
    val_ds = limit_dataset(load_chestmnist("val", args.image_size, args.in_channels, args.data_root), args.subset_size, args.seed + 1)
    test_ds = limit_dataset(load_chestmnist("test", args.image_size, args.in_channels, args.data_root), args.subset_size, args.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_supervised_model(args.model, len(CHEST_LABELS), args.in_channels, args.image_size, not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    train_labels = labels_from_dataset(train_ds)
    pos_weight = None
    if args.pos_weight:
        positives = train_labels.sum(axis=0)
        negatives = train_labels.shape[0] - positives
        pos_weight = torch.tensor(negatives / np.maximum(positives, 1.0), dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("supervised_chestmnist")
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params(vars(args))
        mlflow.log_param("trainable_parameters", count_parameters(model))
        mlflow.log_param("loss", "BCEWithLogitsLoss")
        mlflow.log_param("scheduler", "CosineAnnealingLR")
        best_val = -1.0
        stale_epochs = 0
        best_path = Path("artifacts") / f"best_supervised_{args.model}.pt"
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            y_val, p_val = collect(model, val_loader, device)
            threshold, val_f1_at_threshold = best_threshold_for_f1(y_val, p_val)
            val_metrics = multilabel_metrics(y_val, p_val, threshold=threshold)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)
            mlflow.log_metric("val_best_threshold", threshold, step=epoch)
            mlflow.log_metric("val_f1_macro_threshold_search", val_f1_at_threshold, step=epoch)
            for k, v in val_metrics.items():
                if np.isfinite(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
            score = val_metrics.get("auc_macro", float("nan"))
            if not np.isfinite(score):
                score = val_metrics["f1_macro"]
            if score > best_val:
                best_val = score
                stale_epochs = 0
                torch.save(
                    {
                        "model": args.model,
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "labels": CHEST_LABELS,
                        "threshold": threshold,
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                mlflow.log_param("early_stopped_epoch", epoch)
                break

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        threshold = float(ckpt.get("threshold", 0.5))
        y_test, p_test = collect(model, test_loader, device)
        test_metrics = multilabel_metrics(y_test, p_test, threshold=threshold)
        for k, v in test_metrics.items():
            if np.isfinite(v):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("selected_threshold", threshold)
        per_class = per_class_metrics(y_test, p_test, CHEST_LABELS, threshold=threshold)
        per_class_path = Path("artifacts") / f"per_class_metrics_{args.model}.csv"
        per_class.to_csv(per_class_path, index=False)
        pred_path = Path("artifacts") / f"predictions_{args.model}.csv"
        pred_frame = pd.DataFrame(
            {
                **{f"prob_{label}": p_test[:, i] for i, label in enumerate(CHEST_LABELS)},
                **{f"true_{label}": y_test[:, i].astype(int) for i, label in enumerate(CHEST_LABELS)},
            }
        )
        pred_frame.insert(0, "row_id", np.arange(len(p_test)))
        pred_frame.to_csv(pred_path, index=False)
        dist_path = Path("artifacts") / f"label_distribution_{args.model}.png"
        ap_path = Path("artifacts") / f"per_class_ap_{args.model}.png"
        save_label_distribution(y_test, dist_path)
        save_per_class_bars(per_class, "average_precision", ap_path)
        mlflow.log_artifact(str(best_path), artifact_path="models")
        mlflow.log_artifact(str(per_class_path), artifact_path="metrics")
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")
        mlflow.log_artifact(str(dist_path), artifact_path="figures")
        mlflow.log_artifact(str(ap_path), artifact_path="figures")
        if args.model == "cnn":
            torch.save(torch.load(best_path, map_location="cpu"), Path("artifacts") / "best_supervised.pt")
        print({"model": args.model, "best_val": best_val, **test_metrics})


if __name__ == "__main__":
    main()
