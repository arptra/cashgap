from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.jobs.manager import create_job
from app.security import sanitize_for_storage
from app.services.ingestion import check_access_job, download_source_job
from app.services.registry import get_source, list_sources
from app.storage.database import insert_dataset


router = APIRouter(prefix="/sources", tags=["sources"])


class SourceActionRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


class SourceDownloadRequest(SourceActionRequest):
    accepted_terms: bool = False

    @model_validator(mode="after")
    def terms_required(self):
        if not self.accepted_terms:
            raise ValueError("Confirm that you reviewed the source terms and license")
        return self


@router.get("")
def sources() -> list[dict[str, Any]]:
    return list_sources()


@router.get("/{source_id}")
def source(source_id: str) -> dict[str, Any]:
    result = get_source(source_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return result


@router.post("/{source_id}/check-access", status_code=status.HTTP_202_ACCEPTED)
def check_access(source_id: str, request: SourceActionRequest | None = None) -> dict[str, str]:
    if get_source(source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    options = request.options if request else {}
    job_id = create_job("check_access", check_access_job(source_id, options), source_id=source_id, options=options)
    return {"job_id": job_id, "status": "queued"}


@router.post("/{source_id}/download", status_code=status.HTTP_202_ACCEPTED)
def download_source(source_id: str, request: SourceDownloadRequest) -> dict[str, str]:
    source_record = get_source(source_id)
    if source_record is None:
        raise HTTPException(status_code=404, detail="Source not found")
    dataset_id = f"ds_{source_id}_{uuid4().hex[:8]}"
    insert_dataset(
        dataset_id,
        {
            "source_id": source_id,
            "provider": source_record["provider"],
            "stage": "raw",
            "options": sanitize_for_storage(request.options),
        },
    )
    job_id = create_job(
        "download",
        download_source_job(source_id, dataset_id, request.options),
        source_id=source_id,
        dataset_id=dataset_id,
        options=request.options,
    )
    return {"job_id": job_id, "dataset_id": dataset_id, "status": "queued"}
