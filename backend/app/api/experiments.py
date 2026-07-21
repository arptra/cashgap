from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.services.executor import worker_pool
from app.services.experiments import categorization_run_job, forecast_run_job
from app.services.training import train_run_job
from app.storage.database import delete_run_record, get_dataset, get_run, insert_run, list_runs


router = APIRouter(prefix="/experiments", tags=["experiments"])

TaskName = Literal["cash_gap_classification", "flow_forecasting", "transaction_categorization"]
TASK_MODELS = {
    "cash_gap_classification": {"dummy", "logistic_regression", "random_forest", "catboost", "lightgbm"},
    "flow_forecasting": {"seasonal_naive", "auto_ets", "auto_arima", "lightgbm_forecast"},
    "transaction_categorization": {"tfidf_logistic_regression"},
}
COMPATIBILITY_KEYS = {
    "cash_gap_classification": "classification_eligible",
    "flow_forecasting": "forecasting_eligible",
    "transaction_categorization": "categorization_eligible",
}


class ExperimentStartRequest(BaseModel):
    dataset_id: str
    task: TaskName
    models: list[str]
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    train_ratio: float = Field(default=0.60, ge=0.4, le=0.8)
    validation_ratio: float = Field(default=0.20, ge=0.1, le=0.3)

    @field_validator("models")
    @classmethod
    def at_least_one(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Select at least one model")
        return list(dict.fromkeys(value))


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=1)


class CategorizeRequest(BaseModel):
    descriptions: list[str] = Field(min_length=1, max_length=10_000)


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
def start(request: ExperimentStartRequest) -> dict[str, Any]:
    dataset = get_dataset(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset["status"] != "completed":
        raise HTTPException(status_code=409, detail="Dataset is not ready")
    compatibility = (dataset.get("summary") or {}).get("compatibility") or {}
    compatibility_key = COMPATIBILITY_KEYS[request.task]
    if not compatibility.get(compatibility_key, False):
        reasons = compatibility.get("reasons") or ["Dataset is incompatible with the selected task"]
        raise HTTPException(status_code=409, detail={"message": "Incompatible dataset", "reasons": reasons})
    invalid = set(request.models) - TASK_MODELS[request.task]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unsupported models for {request.task}: {sorted(invalid)}")

    run_ids: list[str] = []
    for model_name in request.models:
        run_id = f"run_{uuid4().hex[:12]}"
        params = {
            **request.parameters.get(model_name, {}),
            "train_ratio": request.train_ratio,
            "validation_ratio": request.validation_ratio,
        }
        insert_run(run_id, request.dataset_id, model_name, params, task=request.task)
        if request.task == "cash_gap_classification":
            worker_pool.submit(train_run_job, run_id, request.dataset_id, model_name, params)
        elif request.task == "flow_forecasting":
            worker_pool.submit(forecast_run_job, run_id, request.dataset_id, model_name, params)
        else:
            worker_pool.submit(categorization_run_job, run_id, request.dataset_id, model_name, params)
        run_ids.append(run_id)
    return {"run_ids": run_ids, "status": "queued"}


@router.get("")
def experiments() -> list[dict]:
    return list_runs()


@router.post("/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    runs = []
    for run_id in request.run_ids:
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        runs.append(run)
    tasks = {run["task"] for run in runs}
    return {"runs": runs, "same_task": len(tasks) == 1, "tasks": sorted(tasks)}


@router.get("/{run_id}")
def experiment(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/metrics")
def metrics(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.get("metrics") or {}


def _prediction_frame(run_id: str) -> pd.DataFrame:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "completed" or not run.get("artifact_path"):
        raise HTTPException(status_code=409, detail="Predictions are not ready")
    path = Path(run["artifact_path"]) / "predictions.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Predictions artifact not found")
    return pd.read_parquet(path)


@router.get("/{run_id}/predictions")
def predictions(run_id: str, limit: int = Query(default=1000, ge=1, le=100_000), offset: int = Query(default=0, ge=0)) -> dict:
    frame = _prediction_frame(run_id)
    page = frame.iloc[offset : offset + limit]
    return {"items": page.where(page.notna(), None).to_dict(orient="records"), "total": len(frame), "limit": limit, "offset": offset}


@router.get("/{run_id}/predictions.csv")
def predictions_csv(run_id: str) -> StreamingResponse:
    frame = _prediction_frame(run_id)
    return StreamingResponse(
        iter([frame.to_csv(index=False)]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_predictions.csv"'},
    )


@router.get("/{run_id}/feature-importance")
def feature_importance(run_id: str) -> list[dict]:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.get("feature_importance") or []


@router.post("/{run_id}/categorize")
def categorize(run_id: str, request: CategorizeRequest) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["task"] != "transaction_categorization":
        raise HTTPException(status_code=409, detail="This run is not a transaction categorizer")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail="Categorization model is not ready")
    model_path = Path(run["artifact_path"]).parent.parent / "models" / f"{run_id}.joblib"
    if not model_path.exists():
        from app.config import MODELS_DIR

        model_path = MODELS_DIR / f"{run_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Categorization model artifact not found")
    model = joblib.load(model_path)
    predicted = model.predict(pd.Series(request.descriptions, dtype="string"))
    return {
        "items": [
            {"transaction_description": description, "predicted_category": str(category)}
            for description, category in zip(request.descriptions, predicted, strict=True)
        ]
    }


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete(run_id: str) -> Response:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="A running experiment cannot be deleted")
    if run.get("artifact_path"):
        shutil.rmtree(Path(run["artifact_path"]), ignore_errors=True)
    delete_run_record(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
