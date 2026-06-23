#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from radiotriage import CHEST_LABELS
from radiotriage.data import limit_dataset, load_chestmnist
from radiotriage.models import build_supervised_model


CASE_ORDER = ["TP", "FP", "FN"]
DEFAULT_LABELS = ["effusion", "infiltration", "atelectasis", "edema"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/best_supervised.pt")
    p.add_argument("--predictions", default="artifacts/predictions_cnn.csv")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--max-cases", type=int, default=12)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def tensor_to_gray(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu()[0].numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heat = np.clip(heatmap, 0.0, 1.0)
    base_rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
    heat_rgb = np.zeros_like(base_rgb)
    heat_rgb[..., 0] = 255.0 * heat
    heat_rgb[..., 1] = 130.0 * heat
    alpha = 0.50 * heat[..., None]
    out = base_rgb * (1.0 - alpha) + heat_rgb * alpha
    return Image.fromarray(out.clip(0, 255).astype("uint8"), mode="RGB")


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = model.features[10]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_inputs, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()[0]
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        cam_img = Image.fromarray((cam.cpu().numpy() * 255).astype("uint8"), mode="L")
        cam_img = cam_img.resize((x.shape[-1], x.shape[-2]), Image.Resampling.BILINEAR)
        return np.asarray(cam_img, dtype=np.float32) / 255.0


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    model = build_supervised_model(
        args["model"],
        len(CHEST_LABELS),
        args["in_channels"],
        args["image_size"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    model.eval()
    return model, ckpt


def select_error_cases(predictions: pd.DataFrame, labels: list[str], threshold: float, max_cases: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in labels:
        if label not in CHEST_LABELS:
            continue
        prob_col = f"prob_{label}"
        true_col = f"true_{label}"
        if prob_col not in predictions or true_col not in predictions:
            continue
        frame = predictions[["row_id", prob_col, true_col]].copy()
        frame = frame.rename(columns={prob_col: "probability", true_col: "truth"})
        frame["prediction"] = (frame["probability"] >= threshold).astype(int)
        candidates = {
            "TP": frame[(frame["truth"] == 1) & (frame["prediction"] == 1)].sort_values("probability", ascending=False),
            "FP": frame[(frame["truth"] == 0) & (frame["prediction"] == 1)].sort_values("probability", ascending=False),
            "FN": frame[(frame["truth"] == 1) & (frame["prediction"] == 0)].sort_values("probability", ascending=False),
        }
        for case_type in CASE_ORDER:
            subset = candidates[case_type]
            if subset.empty:
                continue
            sample = subset.iloc[0]
            rows.append(
                {
                    "label": label,
                    "case_type": case_type,
                    "row_id": int(sample["row_id"]),
                    "truth": int(sample["truth"]),
                    "prediction": int(sample["prediction"]),
                    "probability": float(sample["probability"]),
                    "threshold": threshold,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["case_rank"] = out["case_type"].map({"TP": 0, "FP": 1, "FN": 2}).astype(int)
    out = out.sort_values(["case_rank", "probability"], ascending=[True, False])
    return out.head(max_cases).drop(columns=["case_rank"]).reset_index(drop=True)


def summarize_errors(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for label in CHEST_LABELS:
        y_true = predictions[f"true_{label}"].to_numpy(dtype=int)
        y_pred = (predictions[f"prob_{label}"].to_numpy(dtype=float) >= threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        rows.append(
            {
                "label": label,
                "support": int(y_true.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "false_positive_rate": fp / max(fp + tn, 1),
                "false_negative_rate": fn / max(fn + tp, 1),
            }
        )
    return pd.DataFrame(rows)


def draw_case_grid(case_rows: pd.DataFrame, dataset, model, device: torch.device, out_path: Path) -> None:
    if case_rows.empty:
        return
    cam = GradCAM(model)
    tile_w, tile_h = 230, 190
    pad, header_h = 18, 34
    width = pad * 3 + tile_w * 2
    height = header_h + pad + len(case_rows) * (tile_h + pad)
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 10), "Image originale", fill="black")
    draw.text((pad * 2 + tile_w, 10), "Grad-CAM classe cible", fill="black")

    for i, row in case_rows.iterrows():
        x_tensor, _ = dataset[int(row["row_id"])]
        x = x_tensor.unsqueeze(0).to(device)
        label_idx = CHEST_LABELS.index(str(row["label"]))
        heatmap = cam(x, label_idx)
        base = tensor_to_gray(x_tensor).resize((128, 128), Image.Resampling.BILINEAR).convert("RGB")
        overlay = overlay_heatmap(tensor_to_gray(x_tensor), heatmap).resize((128, 128), Image.Resampling.BILINEAR)
        y0 = header_h + pad + i * (tile_h + pad)
        grid.paste(base, (pad, y0 + 35))
        grid.paste(overlay, (pad * 2 + tile_w, y0 + 35))
        caption = f"{row['case_type']} - {row['label']} - p={row['probability']:.3f} - y={row['truth']}"
        draw.text((pad, y0), caption, fill="black")
        draw.text((pad, y0 + 145), "Reference", fill="#444444")
        draw.text((pad * 2 + tile_w, y0 + 145), "Zones contributives", fill="#444444")
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_model(Path(args.checkpoint), device)
    train_args = ckpt["args"]
    threshold = float(ckpt.get("threshold", 0.5))
    predictions = pd.read_csv(args.predictions)

    test_ds = limit_dataset(
        load_chestmnist(
            "test",
            train_args["image_size"],
            train_args["in_channels"],
            train_args.get("data_root", "data"),
        ),
        train_args.get("subset_size"),
        train_args.get("seed", 42) + 2,
    )

    cases = select_error_cases(predictions, args.labels, threshold, args.max_cases)
    cases.to_csv(artifacts / "error_cases_cnn.csv", index=False)
    summarize_errors(predictions, threshold).to_csv(artifacts / "error_summary_cnn.csv", index=False)
    draw_case_grid(cases, test_ds, model, device, artifacts / "gradcam_error_cases_cnn.png")
    print({"cases": len(cases), "threshold": threshold, "labels": args.labels})


if __name__ == "__main__":
    main()
