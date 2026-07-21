from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.config import RUNS_DIR
from app.db.base import SessionLocal
from app.db.models import ArtifactRecord, BenchmarkRecord
from app.jobs.manager import update_job
from app.model_plugins import create_model_plugin
from app.models_registry.registry import get_model_spec
from app.storage.database import get_dataset, get_run, insert_run, list_runs, update_run


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(record: BenchmarkRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "task": record.task,
        "target": record.target,
        "series_level": record.series_level,
        "horizon": record.horizon,
        "min_history": record.min_history,
        "model_ids": json.loads(record.model_ids_json),
        "run_ids": json.loads(record.run_ids_json),
        "status": record.status,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
    }


def create_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    benchmark_id = f"bench_{uuid4().hex[:12]}"
    task = config["task"]
    run_ids = []
    for model_id in config["model_ids"]:
        run_id = f"run_{uuid4().hex[:12]}"
        insert_run(
            run_id,
            config["dataset_id"],
            model_id,
            {**config.get("parameters", {}).get(model_id, {}), "benchmark_id": benchmark_id, "target": config["target"], "horizon": config["horizon"], "series_level": config["series_level"], "min_history": config["min_history"]},
            task="cash_gap_classification" if task == "cash_gap_classification" else "flow_forecasting",
        )
        run_ids.append(run_id)
    with SessionLocal() as session:
        record = BenchmarkRecord(
            id=benchmark_id,
            dataset_id=config["dataset_id"],
            task=task,
            target=config["target"],
            series_level=config["series_level"],
            horizon=config["horizon"],
            min_history=config["min_history"],
            model_ids_json=json.dumps(config["model_ids"]),
            run_ids_json=json.dumps(run_ids),
            status="queued",
        )
        session.add(record)
        session.commit()
        return _serialize(record)


def update_benchmark(benchmark_id: str, **values: Any) -> None:
    with SessionLocal() as session:
        record = session.get(BenchmarkRecord, benchmark_id)
        if record is None:
            raise KeyError(benchmark_id)
        for key, value in values.items():
            setattr(record, key, value)
        record.updated_at = _now()
        session.commit()


def get_benchmark(benchmark_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.get(BenchmarkRecord, benchmark_id)
        return _serialize(record) if record else None


def list_benchmarks() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        records = session.scalars(select(BenchmarkRecord).order_by(BenchmarkRecord.created_at.desc())).all()
        return [_serialize(record) for record in records]


def _run_plugin(run_id: str, model_id: str, dataset: dict[str, Any], config: dict[str, Any]) -> bool:
    started = time.perf_counter()
    run_dir = RUNS_DIR / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        update_run(run_id, status="running", started_at=_now().isoformat(), artifact_path=str(run_dir), error=None)
        spec = get_model_spec(model_id)
        if spec is None:
            raise KeyError(f"Unknown model: {model_id}")
        if spec.task != config["task"]:
            raise ValueError(f"Model {model_id} has task {spec.task}, benchmark has {config['task']}")
        plugin = create_model_plugin(spec)
        options = {
            "target": config["target"],
            "series_level": config["series_level"],
            "horizon": config["horizon"],
            "min_history": config["min_history"],
            "parameters": config.get("parameters", {}).get(model_id, {}),
            "train_ratio": config.get("train_ratio", 0.6),
            "validation_ratio": config.get("validation_ratio", 0.2),
        }
        result = plugin.run(dataset, options)
        paths = plugin.save_artifacts(run_dir, result)
        importance = result.parameters.get("feature_importance", []) if isinstance(result.parameters, dict) else []
        split = (result.parameters.get("split") or {}) if isinstance(result.parameters, dict) else {}
        split.update({"benchmark_id": config["benchmark_id"], "target": config["target"], "horizon": config["horizon"], "series_level": config["series_level"]})
        with SessionLocal() as session:
            session.add_all([ArtifactRecord(run_id=run_id, kind=kind, path=path) for kind, path in paths.items()])
            session.commit()
        update_run(run_id, status="completed", completed_at=_now().isoformat(), duration_seconds=time.perf_counter() - started, metrics_json=result.metrics, feature_importance_json=importance, feature_names_json=result.parameters.get("feature_columns", []) if isinstance(result.parameters, dict) else [], split_json=split, artifact_path=str(run_dir), error=None)
        return True
    except Exception:
        update_run(run_id, status="failed", completed_at=_now().isoformat(), duration_seconds=time.perf_counter() - started, artifact_path=str(run_dir), error=traceback.format_exc())
        return False


def benchmark_job(benchmark_id: str, config: dict[str, Any]):
    def run(job_id: str, cancel_event) -> dict[str, Any]:
        update_benchmark(benchmark_id, status="running", error=None)
        dataset = get_dataset(config["dataset_id"])
        if dataset is None:
            raise KeyError(f"Unknown dataset: {config['dataset_id']}")
        benchmark = get_benchmark(benchmark_id)
        assert benchmark is not None
        completed = 0
        failures = []
        for index, (model_id, run_id) in enumerate(zip(benchmark["model_ids"], benchmark["run_ids"], strict=True)):
            if cancel_event.is_set():
                update_benchmark(benchmark_id, status="cancelled", completed_at=_now())
                return {"benchmark_id": benchmark_id, "completed_runs": completed}
            update_job(job_id, status="running", progress=index / max(len(benchmark["run_ids"]), 1), message=f"Running {model_id}")
            ok = _run_plugin(run_id, model_id, dataset, {**config, "benchmark_id": benchmark_id})
            completed += int(ok)
            if not ok:
                failures.append(model_id)
        status = "completed" if completed else "failed"
        error = f"Failed models: {', '.join(failures)}" if failures else None
        update_benchmark(benchmark_id, status=status, error=error, completed_at=_now())
        return {"benchmark_id": benchmark_id, "completed_runs": completed, "failed_models": failures}

    return run


def benchmark_comparison(benchmark_id: str) -> dict[str, Any]:
    benchmark = get_benchmark(benchmark_id)
    if benchmark is None:
        raise KeyError(benchmark_id)
    by_id = {run["id"]: run for run in list_runs()}
    runs = [by_id[run_id] for run_id in benchmark["run_ids"] if run_id in by_id]
    return {
        "benchmark": benchmark,
        "runs": runs,
        "comparable": True,
        "comparison_contract": {
            "dataset_id": benchmark["dataset_id"],
            "task": benchmark["task"],
            "target": benchmark["target"],
            "horizon": benchmark["horizon"],
            "series_level": benchmark["series_level"],
        },
    }
