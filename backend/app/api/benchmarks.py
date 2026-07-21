from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.jobs.manager import create_job, get_job, list_jobs
from app.model_plugins import create_model_plugin
from app.models_registry.registry import get_model_spec
from app.services.benchmarks import benchmark_comparison, benchmark_job, create_benchmark, get_benchmark, list_benchmarks
from app.storage.database import get_dataset


router = APIRouter(tags=["model-benchmarks"])


class BenchmarkStartRequest(BaseModel):
    dataset_id: str
    task: Literal["cash_flow_forecasting", "cash_gap_classification"]
    target: str
    model_ids: list[str] = Field(min_length=1)
    series_level: Literal["client", "client_category"] = "client"
    horizon: int = Field(default=1, ge=1, le=3)
    min_history: int = Field(default=6, ge=3, le=120)
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    train_ratio: float = Field(default=0.6, ge=0.4, le=0.8)
    validation_ratio: float = Field(default=0.2, ge=0.1, le=0.3)

    @field_validator("model_ids")
    @classmethod
    def unique_models(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


@router.post("/benchmarks/start", status_code=status.HTTP_202_ACCEPTED)
def start_benchmark(request: BenchmarkStartRequest) -> dict[str, Any]:
    dataset = get_dataset(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    errors = []
    for model_id in request.model_ids:
        spec = get_model_spec(model_id)
        if spec is None:
            errors.append(f"Unknown model: {model_id}")
            continue
        if spec.task != request.task:
            errors.append(f"{model_id}: task is {spec.task}")
            continue
        report = create_model_plugin(spec).check_compatibility(dataset, target=request.target, series_level=request.series_level, horizon=request.horizon, min_history=request.min_history)
        if not report.compatible:
            errors.append(f"{model_id}: {'; '.join(report.reasons)}")
    if errors:
        raise HTTPException(status_code=409, detail={"message": "Incompatible benchmark", "reasons": errors})
    config = request.model_dump()
    benchmark = create_benchmark(config)
    job_id = create_job("model_benchmark", benchmark_job(benchmark["id"], config), dataset_id=request.dataset_id, options={"benchmark_id": benchmark["id"], "model_ids": request.model_ids})
    return {"job_id": job_id, "benchmark_id": benchmark["id"], "run_ids": benchmark["run_ids"], "status": "queued"}


@router.get("/benchmarks")
def benchmarks() -> list[dict[str, Any]]:
    return list_benchmarks()


@router.get("/benchmarks/{benchmark_id}")
def benchmark(benchmark_id: str) -> dict[str, Any]:
    result = get_benchmark(benchmark_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return result


@router.get("/benchmarks/{benchmark_id}/comparison")
def comparison(benchmark_id: str) -> dict[str, Any]:
    try:
        return benchmark_comparison(benchmark_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Benchmark not found")


@router.get("/model-jobs")
def model_jobs() -> list[dict[str, Any]]:
    return [job for job in list_jobs() if job["job_type"].startswith("model_")]


@router.get("/model-jobs/{job_id}")
def model_job(job_id: str) -> dict[str, Any]:
    result = get_job(job_id)
    if result is None or not result["job_type"].startswith("model_"):
        raise HTTPException(status_code=404, detail="Model job not found")
    return result
