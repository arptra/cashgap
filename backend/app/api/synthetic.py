from __future__ import annotations

import threading
from uuid import uuid4

from fastapi import APIRouter, status

from app.jobs.manager import create_job, update_job
from app.schemas.api import DatasetGenerateRequest
from app.services.datasets import generate_dataset_job
from app.storage.database import get_dataset, insert_dataset


router = APIRouter(prefix="/synthetic", tags=["synthetic"])


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
def generate(request: DatasetGenerateRequest) -> dict[str, str]:
    dataset_id = f"ds_synthetic_{uuid4().hex[:8]}"
    config = {**request.model_dump(), "source_id": "synthetic", "provider": "synthetic", "stage": "normalized"}
    insert_dataset(dataset_id, config)

    def job(job_id: str, _cancel: threading.Event) -> dict:
        update_job(job_id, status="normalizing", progress=0.1, message="Generating daily trajectories")
        generate_dataset_job(dataset_id, config)
        dataset = get_dataset(dataset_id)
        if not dataset or dataset["status"] != "completed":
            raise RuntimeError(dataset.get("error") if dataset else "Synthetic generation failed")
        return {"dataset_id": dataset_id, "summary": dataset["summary"]}

    job_id = create_job("synthetic_generation", job, dataset_id=dataset_id, options=request.model_dump())
    return {"job_id": job_id, "dataset_id": dataset_id, "status": "queued"}

