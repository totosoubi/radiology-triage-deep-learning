#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def latest_largest_run(frame: pd.DataFrame, size_column: str | None = None) -> pd.Series | None:
    if frame.empty:
        return None
    ranked = frame.copy()
    sort_columns = []
    if size_column and size_column in ranked.columns:
        ranked["_selection_size"] = pd.to_numeric(ranked[size_column], errors="coerce").fillna(-1)
        sort_columns.append("_selection_size")
    sort_columns.append("start_time")
    return ranked.sort_values(sort_columns).iloc[-1]


def selected_runs_export(runs: pd.DataFrame) -> pd.DataFrame:
    finished = runs[runs["status"].eq("FINISHED")].copy()
    selections: list[tuple[pd.Series, str, str, str, str]] = []

    supervised = finished[finished["experiment"].eq("supervised_chestmnist")]
    for model in ("cnn", "resnet18", "vit"):
        row = latest_largest_run(supervised[supervised["run_name"].eq(model)], "param.subset_size")
        if row is not None:
            role = "modele supervise deploye" if model == "cnn" else "comparaison supervisee"
            checkpoint = "artifacts/best_supervised.pt" if model == "cnn" else f"artifacts/best_supervised_{model}.pt"
            selections.append((row, "classification supervisee", role, checkpoint, "oui" if model == "cnn" else "non"))

    anomaly = finished[finished["experiment"].eq("anomaly_chestmnist_ae")]
    row = latest_largest_run(anomaly, "param.subset_size")
    if row is not None:
        selections.append((row, "detection d'anomalies", "autoencodeur deploye", "artifacts/best_ae.pt", "oui"))

    multimodal = finished[
        finished["experiment"].eq("multimodal_openi_or_mimic")
        & finished.get("param.dataset_kind", pd.Series(index=finished.index, dtype=str)).eq("real_manifest")
    ]
    for mode in ("image", "text", "fusion"):
        row = latest_largest_run(multimodal[multimodal["run_name"].eq(mode)])
        if row is not None:
            role = "modele multimodal deploye" if mode == "fusion" else "comparaison multimodale"
            checkpoint = f"artifacts/best_multimodal_{mode}.pt"
            selections.append((row, "multimodalite", role, checkpoint, "oui" if mode == "fusion" else "non"))

    rows = []
    for run, component, role, checkpoint, deployed in selections:
        start = pd.to_numeric(run.get("start_time"), errors="coerce")
        end = pd.to_numeric(run.get("end_time"), errors="coerce")
        duration = (end - start) / 1000 if pd.notna(start) and pd.notna(end) else float("nan")
        rows.append(
            {
                "component": component,
                "run_name": run.get("run_name", ""),
                "role": role,
                "deployed_in_streamlit": deployed,
                "run_id": run.get("run_id", ""),
                "status": run.get("status", ""),
                "duration_seconds": duration,
                "test_auc_micro": run.get("metric.test_auc_micro"),
                "test_ap_micro": run.get("metric.test_ap_micro"),
                "test_f1_macro": run.get("metric.test_f1_macro"),
                "selected_threshold": run.get("metric.selected_threshold"),
                "checkpoint": checkpoint,
            }
        )
    return pd.DataFrame(rows)


def save_selected_runs_figure(selected: pd.DataFrame, path: Path) -> None:
    columns = ["component", "run_name", "role", "test_auc_micro", "test_ap_micro", "test_f1_macro", "run_id"]
    view = selected[columns].copy()
    view["component"] = view["component"].replace(
        {"classification supervisee": "Supervise", "detection d'anomalies": "Anomalie", "multimodalite": "Multimodal"}
    )
    for column in ("test_auc_micro", "test_ap_micro", "test_f1_macro"):
        view[column] = view[column].map(lambda value: "-" if pd.isna(value) else f"{float(value):.4f}")
    view["run_id"] = view["run_id"].astype(str).str.slice(0, 10)
    view.columns = ["Composante", "Run", "Usage", "AUC micro", "AP micro", "F1 macro", "Run ID"]

    fig, ax = plt.subplots(figsize=(14, 1.7 + 0.5 * len(view)))
    ax.axis("off")
    table = ax.table(cellText=view.values, colLabels=view.columns, cellLoc="center", bbox=[0, 0.01, 1, 0.82])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif selected.iloc[row - 1]["deployed_in_streamlit"] == "oui":
            cell.set_facecolor("#dff0e3")
        else:
            cell.set_facecolor("#f3f4f5")
    ax.set_title("Export MLflow - runs de comparaison et modeles retenus", fontsize=14, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri("file:mlruns")
    client = mlflow.tracking.MlflowClient()
    rows = []
    for exp in client.search_experiments():
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
        for run in runs:
            row = {
                "experiment": exp.name,
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
            row.update({f"param.{k}": v for k, v in run.data.params.items()})
            row.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
            rows.append(row)
    out = Path("artifacts") / "mlflow_runs_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out, index=False)

    selected = selected_runs_export(runs)
    selected_out = out.parent / "mlflow_selected_runs.csv"
    selected.to_csv(selected_out, index=False)
    figure_out = out.parent / "mlflow_selected_runs.png"
    save_selected_runs_figure(selected, figure_out)
    print(f"Exported {len(rows)} runs to {out}")
    print(f"Exported {len(selected)} selected runs to {selected_out} and {figure_out}")


if __name__ == "__main__":
    main()
