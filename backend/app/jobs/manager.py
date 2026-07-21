from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select

from app.db.base import SessionLocal
from app.db.models import JobRecord
from app.security import sanitize_for_storage
from app.services.executor import worker_pool


JobFunction = Callable[[str, threading.Event], dict[str, Any] | None]
_cancel_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(job: JobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "source_id": job.source_id,
        "dataset_id": job.dataset_id,
        "progress": job.progress,
        "message": job.message,
        "options": json.loads(job.options_json or "{}"),
        "result": json.loads(job.result_json) if job.result_json else None,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def create_job(
    job_type: str,
    function: JobFunction,
    *,
    source_id: str | None = None,
    dataset_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    job_id = f"job_{uuid4().hex[:12]}"
    event = threading.Event()
    with _lock:
        _cancel_events[job_id] = event
    with SessionLocal() as session:
        session.add(
            JobRecord(
                id=job_id,
                job_type=job_type,
                status="queued",
                source_id=source_id,
                dataset_id=dataset_id,
                progress=0.0,
                message="Queued",
                options_json=json.dumps(sanitize_for_storage(options or {}), ensure_ascii=False),
            )
        )
        session.commit()
    worker_pool.submit(_run_job, job_id, function, event)
    return job_id


def _run_job(job_id: str, function: JobFunction, event: threading.Event) -> None:
    try:
        if event.is_set():
            update_job(job_id, status="cancelled", message="Cancelled before start")
            return
        result = function(job_id, event) or {}
        if event.is_set():
            update_job(job_id, status="cancelled", progress=1.0, message="Cancelled")
        else:
            update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Completed",
                result=result,
            )
    except Exception:
        update_job(
            job_id,
            status="cancelled" if event.is_set() else "failed",
            message="Cancelled" if event.is_set() else "Failed",
            error=traceback.format_exc(),
        )
    finally:
        with _lock:
            _cancel_events.pop(job_id, None)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with SessionLocal() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            raise KeyError(job_id)
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = max(0.0, min(float(progress), 1.0))
        if message is not None:
            job.message = message
        if result is not None:
            job.result_json = json.dumps(result, ensure_ascii=False)
        if error is not None:
            job.error = error
        job.updated_at = _now()
        session.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        job = session.get(JobRecord, job_id)
        return _serialize(job) if job else None


def list_jobs() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        jobs = session.scalars(select(JobRecord).order_by(JobRecord.created_at.desc())).all()
        return [_serialize(job) for job in jobs]


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if job is None:
        return False
    if job["status"] in {"completed", "failed", "cancelled"}:
        return True
    with _lock:
        event = _cancel_events.get(job_id)
        if event:
            event.set()
    update_job(job_id, status="cancelled", message="Cancellation requested")
    return True
