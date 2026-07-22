from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.jobs.manager import create_job, list_jobs
from app.model_plugins import create_model_plugin
from app.models_registry.registry import get_model, get_model_spec, list_models
from app.services.benchmarks import benchmark_job, create_benchmark
from app.services.executor import worker_pool
from app.storage.database import get_dataset


router = APIRouter(prefix="/models", tags=["models"])


class CompatibilityRequest(BaseModel):
    dataset_id: str
    target: str
    series_level: str = "client"
    horizon: int = Field(default=1, ge=1, le=3)
    min_history: int = Field(default=6, ge=3, le=120)


class ModelRunRequest(CompatibilityRequest):
    parameters: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def models() -> list[dict[str, Any]]:
    return list_models()


@router.get("/{model_id}")
def model(model_id: str) -> dict[str, Any]:
    result = get_model(model_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.post("/{model_id}/check")
def check(model_id: str) -> dict[str, Any]:
    spec = get_model_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return create_model_plugin(spec).check_environment().model_dump()


@router.post("/{model_id}/install", status_code=status.HTTP_202_ACCEPTED)
def install(model_id: str) -> dict[str, str]:
    spec = get_model_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Model not found")

    active_job = next(
        (
            job
            for job in list_jobs()
            if job["job_type"] == "model_install"
            and job["status"] in {"queued", "installing"}
            and job["options"].get("model_id") == model_id
        ),
        None,
    )
    if active_job:
        return {"job_id": active_job["id"], "model_id": model_id, "status": active_job["status"]}

    def installation(job_id, cancel_event):
        from app.jobs.manager import update_job

        update_job(job_id, status="installing", progress=0.05, message=f"Installing {spec.name}")
        return create_model_plugin(spec).install(cancel_event)

    job_id = create_job("model_install", installation, options={"model_id": model_id})
    return {"job_id": job_id, "model_id": model_id, "status": "queued"}


@router.delete("/{model_id}/install", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def uninstall(model_id: str) -> Response:
    spec = get_model_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if spec.bundled:
        raise HTTPException(status_code=409, detail="Bundled offline model cannot be removed")
    create_model_plugin(spec).uninstall()
    return Response(status_code=204)


@router.post("/{model_id}/compatibility")
def compatibility(model_id: str, request: CompatibilityRequest) -> dict[str, Any]:
    spec = get_model_spec(model_id)
    dataset = get_dataset(request.dataset_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return create_model_plugin(spec).check_compatibility(dataset, target=request.target, series_level=request.series_level, horizon=request.horizon, min_history=request.min_history).model_dump()


@router.post("/{model_id}/run", status_code=status.HTTP_202_ACCEPTED)
def run_model(model_id: str, request: ModelRunRequest) -> dict[str, Any]:
    spec = get_model_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="Model not found")
    config = {**request.model_dump(), "task": spec.task, "model_ids": [model_id], "parameters": {model_id: request.parameters}}
    benchmark = create_benchmark(config)
    job_id = create_job("model_benchmark", benchmark_job(benchmark["id"], config), dataset_id=request.dataset_id, options={"model_id": model_id, "benchmark_id": benchmark["id"]})
    return {"job_id": job_id, "benchmark_id": benchmark["id"], "run_ids": benchmark["run_ids"], "status": "queued"}
