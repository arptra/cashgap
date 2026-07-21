from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app.config import MODELS_DIR, RUNS_DIR
from app.db.base import SessionLocal
from app.db.models import ArtifactRecord
from app.ml.explanations import explain_rows, model_contributions
from app.ml.features import build_feature_frame, feature_columns
from app.ml.training import train_and_evaluate
from app.storage.database import get_dataset, update_run


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_months(
    months: list[str],
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> tuple[list[str], list[str], list[str]]:
    if len(months) < 6:
        raise ValueError("At least six scoring months are required for a temporal split")
    train_end = max(2, int(len(months) * train_ratio))
    validation_end = max(train_end + 2, int(len(months) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(months) - 2)
    return months[:train_end], months[train_end:validation_end], months[validation_end:]


def train_run_job(run_id: str, dataset_id: str, model_name: str, params: dict) -> None:
    started = time.perf_counter()
    run_dir = RUNS_DIR / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        update_run(run_id, status="running", started_at=_utc_now(), artifact_path=str(run_dir), error=None)
        dataset = get_dataset(dataset_id)
        if dataset is None or dataset["status"] != "completed" or not dataset.get("paths"):
            raise RuntimeError("Dataset is not available for training")
        monthly = pd.read_parquet(dataset["paths"]["monthly_aggregates"])
        target = pd.read_parquet(dataset["paths"]["target"])
        frame = build_feature_frame(monthly, target)
        columns = feature_columns(frame)
        months = sorted(frame["month"].unique().tolist())
        train_months, validation_months, test_months = _split_months(
            months,
            float(params.get("train_ratio", 0.60)),
            float(params.get("validation_ratio", 0.20)),
        )
        masks = {
            "train": frame["month"].isin(train_months),
            "validation": frame["month"].isin(validation_months),
            "test": frame["month"].isin(test_months),
        }
        # Missing lags are legitimate history-unavailable values, represented as zero.
        x = frame[columns].astype(float).fillna(0.0)
        y = frame["cash_gap_next_month"].to_numpy(dtype=np.int8)
        seed = int((dataset.get("config") or {}).get("random_seed", 42))
        result = train_and_evaluate(
            model_name,
            params,
            x.loc[masks["train"]],
            y[masks["train"].to_numpy()],
            x.loc[masks["validation"]],
            y[masks["validation"].to_numpy()],
            x.loc[masks["test"]],
            y[masks["test"].to_numpy()],
            seed,
        )

        test_frame = frame.loc[masks["test"]].copy()
        predictions = pd.DataFrame(
            {
                "client_id": test_frame["client_id"].to_numpy(),
                "scoring_month": test_frame["month"].to_numpy(),
                "risk_score": result.predictions,
                "actual_cash_gap_next_month": test_frame["cash_gap_next_month"].to_numpy(dtype=int),
            },
            index=test_frame.index,
        )
        predictions["predicted_cash_gap"] = (predictions["risk_score"] >= result.threshold).astype(int)
        predictions["risk_rank"] = predictions.groupby("scoring_month")["risk_score"].rank(
            method="first", ascending=False
        ).astype(int)
        percentile = predictions.groupby("scoring_month")["risk_score"].rank(
            method="max", pct=True, ascending=True
        )
        predictions["risk_group"] = np.select(
            [percentile >= 0.90, percentile >= 0.60], ["high", "medium"], default="low"
        )
        test_features = x.loc[masks["test"]]
        contributions = model_contributions(model_name, result.model, test_features)
        reasons = explain_rows(test_frame, contributions, columns)
        predictions = predictions.join(reasons)
        predictions = predictions.sort_values(["risk_score", "scoring_month"], ascending=[False, False])

        split = {
            "train_months": train_months,
            "validation_months": validation_months,
            "test_months": test_months,
            "train_rows": int(masks["train"].sum()),
            "validation_rows": int(masks["validation"].sum()),
            "test_rows": int(masks["test"].sum()),
        }
        model_path = MODELS_DIR / f"{run_id}.joblib"
        joblib.dump(result.model, model_path)
        predictions.to_parquet(run_dir / "predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(result.feature_importance).to_csv(run_dir / "feature_importance.csv", index=False)
        artifacts = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "model_name": model_name,
            "requested_parameters": params,
            "effective_parameters": result.effective_params,
            "model_path": str(model_path),
            "created_at": _utc_now(),
        }
        for filename, payload in (
            ("metrics.json", result.metrics),
            ("parameters.json", artifacts),
            ("split.json", split),
            ("features.json", columns),
        ):
            (run_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with SessionLocal() as session:
            session.add_all(
                [
                    ArtifactRecord(run_id=run_id, kind="model", path=str(model_path)),
                    ArtifactRecord(run_id=run_id, kind="predictions", path=str(run_dir / "predictions.parquet")),
                    ArtifactRecord(run_id=run_id, kind="metrics", path=str(run_dir / "metrics.json")),
                ]
            )
            session.commit()
        duration = time.perf_counter() - started
        update_run(
            run_id,
            status="completed",
            completed_at=_utc_now(),
            duration_seconds=duration,
            metrics_json=result.metrics,
            feature_importance_json=result.feature_importance,
            feature_names_json=columns,
            split_json=split,
            artifact_path=str(run_dir),
            error=None,
        )
    except Exception:
        update_run(
            run_id,
            status="failed",
            completed_at=_utc_now(),
            duration_seconds=time.perf_counter() - started,
            artifact_path=str(run_dir),
            error=traceback.format_exc(),
        )
