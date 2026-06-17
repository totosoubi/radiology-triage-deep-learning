#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from radiotriage import CHEST_LABELS
from radiotriage.data import OpenIManifestDataset, SyntheticMultimodalDataset, TextVocab, multimodal_collate
from radiotriage.metrics import best_threshold_for_f1, multilabel_metrics, per_class_metrics
from radiotriage.models import MultimodalClassifier
from radiotriage.utils import ensure_dir, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None, help="OpenI/MIMIC-style CSV: image_path, report, 14 label columns")
    p.add_argument("--demo-synthetic", action="store_true")
    p.add_argument("--mode", choices=["image", "text", "fusion"], required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--synthetic-size", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for image, tokens, labels in loader:
            image, tokens = image.to(device), tokens.to(device)
            ps.append(torch.sigmoid(model(image, tokens)).cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    ensure_dir("artifacts")

    if args.demo_synthetic:
        ds = SyntheticMultimodalDataset(args.synthetic_size, args.image_size, args.in_channels, args.seed)
    elif args.manifest:
        ds = OpenIManifestDataset(args.manifest, args.image_size, args.in_channels, args.max_tokens)
    else:
        raise SystemExit("Provide --manifest for OpenI/MIMIC data or --demo-synthetic for a smoke test.")

    texts = [ds[i][1] for i in range(len(ds))]
    vocab = TextVocab(texts)
    n_train = max(1, int(0.7 * len(ds)))
    n_val = max(1, int(0.15 * len(ds)))
    n_test = len(ds) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed))
    collate = multimodal_collate(vocab, args.max_tokens)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = MultimodalClassifier(args.mode, len(vocab.stoi), len(CHEST_LABELS), args.in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("multimodal_openi_or_mimic")
    with mlflow.start_run(run_name=args.mode):
        mlflow.log_params(vars(args))
        mlflow.log_param("vocab_size", len(vocab.stoi))
        mlflow.log_param("dataset_kind", "synthetic_smoke" if args.demo_synthetic else "real_manifest")
        best = -1.0
        best_path = Path("artifacts") / f"best_multimodal_{args.mode}.pt"
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for image, tokens, labels in train_loader:
                image, tokens, labels = image.to(device), tokens.to(device), labels.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(image, tokens), labels)
                loss.backward()
                opt.step()
                total += loss.item() * image.size(0)
            y_val, p_val = collect(model, val_loader, device)
            threshold, val_f1_at_threshold = best_threshold_for_f1(y_val, p_val)
            metrics = multilabel_metrics(y_val, p_val, threshold=threshold)
            mlflow.log_metric("train_loss", total / len(train_loader.dataset), step=epoch)
            mlflow.log_metric("val_best_threshold", threshold, step=epoch)
            mlflow.log_metric("val_f1_macro_threshold_search", val_f1_at_threshold, step=epoch)
            for k, v in metrics.items():
                if np.isfinite(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
            score = metrics["f1_macro"] if not np.isfinite(metrics["auc_macro"]) else metrics["auc_macro"]
            if score > best:
                best = score
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "labels": CHEST_LABELS,
                        "vocab": vocab.stoi,
                        "threshold": threshold,
                    },
                    best_path,
                )
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
        metrics_path = Path("artifacts") / f"per_class_metrics_multimodal_{args.mode}.csv"
        pred_path = Path("artifacts") / f"predictions_multimodal_{args.mode}.csv"
        per_class.to_csv(metrics_path, index=False)
        pd.DataFrame(p_test, columns=CHEST_LABELS).assign(row_id=np.arange(len(p_test))).to_csv(pred_path, index=False)
        mlflow.log_artifact(str(best_path), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")
        print({"mode": args.mode, "best_val": best, **test_metrics})


if __name__ == "__main__":
    main()
