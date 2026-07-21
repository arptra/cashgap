from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs.manager import cancel_job, get_job, list_jobs


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def jobs() -> list[dict]:
    return list_jobs()


@router.get("/{job_id}")
def job(job_id: str) -> dict:
    result = get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.post("/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, str]:
    if not cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job_id, "status": "cancelled"}

