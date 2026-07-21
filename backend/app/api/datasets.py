from __future__ import annotations

from uuid import uuid4

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.schemas.api import DatasetGenerateRequest, QueuedResponse
from app.services.datasets import generate_dataset_job
from app.services.executor import worker_pool
from app.services.ingestion import normalize_dataset_job
from app.jobs.manager import create_job
from app.config import DATA_DIR
from app.storage.database import (
    delete_dataset_record, get_dataset, insert_dataset, list_datasets, list_runs,
)


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/generate", response_model=QueuedResponse, status_code=status.HTTP_202_ACCEPTED)
def generate_dataset(request: DatasetGenerateRequest) -> QueuedResponse:
    dataset_id = f"ds_{uuid4().hex[:12]}"
    config = request.model_dump()
    insert_dataset(dataset_id, config)
    worker_pool.submit(generate_dataset_job, dataset_id, config)
    return QueuedResponse(id=dataset_id)


@router.get("")
def datasets() -> list[dict]:
    return list_datasets()


class NormalizeRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


@router.get("/{dataset_id}/preview")
def dataset_preview(
    dataset_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    paths = result.get("paths") or {}
    selected = paths.get("monthly_aggregates") or paths.get("categorization") or paths.get("forecast_series")
    if not selected:
        return {"items": [], "columns": [], "message": "Normalize the dataset to preview canonical rows"}
    frame = pd.read_parquet(selected).head(limit)
    return {"items": frame.where(frame.notna(), None).to_dict(orient="records"), "columns": frame.columns.tolist()}


@router.get("/{dataset_id}/profile")
def dataset_profile(dataset_id: str) -> dict:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return result.get("summary") or {}


@router.get("/{dataset_id}/compatibility")
def dataset_compatibility(dataset_id: str) -> dict:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    summary = result.get("summary") or {}
    return summary.get("compatibility") or {
        "classification_eligible": False,
        "forecasting_eligible": False,
        "proxy_eligible": False,
        "categorization_eligible": False,
        "reasons": ["Dataset has not been normalized"],
    }


@router.post("/{dataset_id}/normalize", status_code=status.HTTP_202_ACCEPTED)
def normalize_dataset(dataset_id: str, request: NormalizeRequest | None = None) -> dict[str, str]:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    options = request.options if request else {}
    job_id = create_job(
        "normalize",
        normalize_dataset_job(dataset_id, options),
        source_id=result["config"].get("source_id"),
        dataset_id=dataset_id,
        options=options,
    )
    return {"job_id": job_id, "dataset_id": dataset_id, "status": "queued"}


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_dataset(dataset_id: str) -> Response:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if any(run["dataset_id"] == dataset_id for run in list_runs()):
        raise HTTPException(status_code=409, detail="Delete dataset experiments first")
    roots: set[Path] = set()
    for raw in (result.get("paths") or {}).values():
        path = Path(raw).resolve()
        if path.is_file():
            path = path.parent
        if path.is_relative_to(DATA_DIR.resolve()):
            roots.add(path)
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)
    delete_dataset_record(dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{dataset_id}")
def dataset(dataset_id: str) -> dict:
    result = get_dataset(dataset_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return result
