from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse

from app.schemas.api import TrainingQueuedResponse, TrainingStartRequest
from app.services.executor import worker_pool
from app.services.training import train_run_job
from app.storage.database import delete_run_record, get_dataset, get_run, insert_run, list_runs


router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start", response_model=TrainingQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training(request: TrainingStartRequest) -> TrainingQueuedResponse:
    dataset = get_dataset(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset["status"] != "completed":
        raise HTTPException(status_code=409, detail="Dataset is not ready")

    run_ids: list[str] = []
    for model_name in request.models:
        run_id = f"run_{uuid4().hex[:12]}"
        params = request.parameters.get(model_name, {})
        insert_run(run_id, request.dataset_id, model_name, params)
        worker_pool.submit(train_run_job, run_id, request.dataset_id, model_name, params)
        run_ids.append(run_id)
    return TrainingQueuedResponse(run_ids=run_ids)


@router.get("/runs")
def runs() -> list[dict]:
    return list_runs()


@router.get("/runs/{run_id}")
def run(run_id: str) -> dict:
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


def _prediction_path(run_record: dict) -> Path:
    if run_record["status"] != "completed" or not run_record.get("artifact_path"):
        raise HTTPException(status_code=409, detail="Predictions are not ready")
    path = Path(run_record["artifact_path"]) / "predictions.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Predictions artifact not found")
    return path


@router.get("/runs/{run_id}/predictions")
def predictions(
    run_id: str,
    limit: int = Query(default=500, ge=1, le=100_000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    frame = pd.read_parquet(_prediction_path(record))
    page = frame.iloc[offset : offset + limit]
    return {"items": page.to_dict(orient="records"), "total": len(frame), "limit": limit, "offset": offset}


@router.get("/runs/{run_id}/predictions.csv")
def predictions_csv(run_id: str) -> StreamingResponse:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    path = _prediction_path(record)

    def stream():
        yield pd.read_parquet(path).to_csv(index=False)

    return StreamingResponse(
        stream(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_predictions.csv"'},
    )


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_run(run_id: str) -> Response:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="A queued or running experiment cannot be deleted")
    artifact_path = record.get("artifact_path")
    if artifact_path:
        shutil.rmtree(Path(artifact_path), ignore_errors=True)
    delete_run_record(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
