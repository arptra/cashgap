from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.config import MODELS_DIR, RUNS_DIR
from app.db.base import SessionLocal
from app.db.models import ArtifactRecord
from app.ml.categorization import train_categorizer
from app.ml.forecasting import prepare_flow_series, train_forecaster
from app.storage.database import get_dataset, update_run


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(run_id: str, kind: str, path: Path) -> None:
    with SessionLocal() as session:
        session.add(ArtifactRecord(run_id=run_id, kind=kind, path=str(path)))
        session.commit()


def forecast_run_job(run_id: str, dataset_id: str, model_name: str, params: dict) -> None:
    started = time.perf_counter()
    run_dir = RUNS_DIR / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        update_run(run_id, status="running", started_at=_now(), artifact_path=str(run_dir), error=None)
        dataset = get_dataset(dataset_id)
        if not dataset or dataset["status"] != "completed":
            raise RuntimeError("Dataset is not ready")
        paths = dataset.get("paths") or {}
        if paths.get("forecast_series"):
            source = pd.read_parquet(paths["forecast_series"])
            value = "net_flow" if "net_flow" in source else "inflow"
            series = source.groupby("date", as_index=False)[value].sum().rename(columns={value: "y"})
            series["date"] = pd.to_datetime(series["date"])
            target_name = value
        else:
            canonical = pd.read_parquet(paths["monthly_aggregates"])
            series, target_name = prepare_flow_series(canonical)
        result = train_forecaster(model_name, series, params)
        model_path = MODELS_DIR / f"{run_id}.joblib"
        joblib.dump(result.model, model_path)
        predictions_path = run_dir / "predictions.parquet"
        result.predictions.to_parquet(predictions_path, index=False)
        payload = {"target": target_name, "effective_parameters": result.parameters}
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        (run_dir / "parameters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _artifact(run_id, "model", model_path)
        _artifact(run_id, "predictions", predictions_path)
        _artifact(run_id, "metrics", run_dir / "metrics.json")
        update_run(
            run_id,
            status="completed",
            completed_at=_now(),
            duration_seconds=time.perf_counter() - started,
            metrics_json=result.metrics,
            feature_importance_json=[],
            feature_names_json=[],
            split_json={"target": target_name, "test_rows": len(result.predictions)},
            artifact_path=str(run_dir),
            error=None,
        )
    except Exception:
        update_run(run_id, status="failed", completed_at=_now(), duration_seconds=time.perf_counter() - started, artifact_path=str(run_dir), error=traceback.format_exc())


def categorization_run_job(run_id: str, dataset_id: str, model_name: str, params: dict) -> None:
    started = time.perf_counter()
    run_dir = RUNS_DIR / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        update_run(run_id, status="running", started_at=_now(), artifact_path=str(run_dir), error=None)
        dataset = get_dataset(dataset_id)
        path = (dataset or {}).get("paths", {}).get("categorization")
        if not path:
            raise RuntimeError("Dataset has no categorization table")
        result = train_categorizer(pd.read_parquet(path), params)
        model_path = MODELS_DIR / f"{run_id}.joblib"
        joblib.dump(result.model, model_path)
        predictions_path = run_dir / "predictions.parquet"
        result.predictions.to_parquet(predictions_path, index=False)
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        (run_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        _artifact(run_id, "model", model_path)
        _artifact(run_id, "predictions", predictions_path)
        _artifact(run_id, "metrics", run_dir / "metrics.json")
        update_run(
            run_id,
            status="completed",
            completed_at=_now(),
            duration_seconds=time.perf_counter() - started,
            metrics_json=result.metrics,
            feature_importance_json=[],
            feature_names_json=[],
            split_json={"test_rows": len(result.predictions)},
            artifact_path=str(run_dir),
            error=None,
        )
    except Exception:
        update_run(run_id, status="failed", completed_at=_now(), duration_seconds=time.perf_counter() - started, artifact_path=str(run_dir), error=traceback.format_exc())
