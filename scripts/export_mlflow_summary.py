#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd


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
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Exported {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
