from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.adapters import create_adapter
from app.canonical.compatibility import build_compatibility_report
from app.canonical.profiling import profile_canonical, profile_raw_files
from app.config import NORMALIZED_DIR, RAW_DIR
from app.connectors import create_connector
from app.jobs.manager import update_job
from app.services.registry import get_source, update_source_access
from app.storage.database import get_dataset, update_dataset


def check_access_job(source_id: str, options: dict[str, Any]):
    def run(job_id: str, _cancel: threading.Event) -> dict[str, Any]:
        source = get_source(source_id)
        if source is None:
            raise KeyError(f"Unknown source: {source_id}")
        update_job(job_id, status="checking_access", progress=0.2, message="Checking source access")
        connector = create_connector(source, RAW_DIR / source["provider"] / source_id / "_access", options)
        result = connector.check_access()
        revision = result.metadata.get("revision") if result.metadata else None
        update_source_access(source_id, "accessible" if result.accessible else "denied", result.message, revision)
        if not result.accessible:
            raise RuntimeError(result.message)
        return {"accessible": True, "message": result.message, "metadata": result.metadata}

    return run


def download_source_job(source_id: str, dataset_id: str, options: dict[str, Any]):
    def run(job_id: str, cancel_event: threading.Event) -> dict[str, Any]:
        source = get_source(source_id)
        if source is None:
            raise KeyError(f"Unknown source: {source_id}")
        raw_dir = RAW_DIR / source["provider"] / source_id / dataset_id
        connector = create_connector(source, raw_dir, options)
        try:
            update_dataset(dataset_id, status="checking_access", error=None)
            update_job(job_id, status="checking_access", progress=0.05, message="Checking access")
            access = connector.check_access()
            revision = access.metadata.get("revision") if access.metadata else None
            update_source_access(source_id, "accessible" if access.accessible else "denied", access.message, revision)
            if not access.accessible:
                raise RuntimeError(access.message)
            update_dataset(dataset_id, status="downloading")
            update_job(job_id, status="downloading", progress=0.2, message="Downloading source files")
            connector.download(cancel_event)
            update_job(job_id, status="extracting", progress=0.7, message="Inspecting downloaded files")
            if cancel_event.is_set():
                raise RuntimeError("Download cancelled")
            update_dataset(dataset_id, status="profiling")
            update_job(job_id, status="profiling", progress=0.85, message="Profiling raw files")
            summary = {
                **profile_raw_files(raw_dir),
                "source_id": source_id,
                "source_provider": source["provider"],
                "source_revision": revision,
                "compatibility": {
                    "classification_eligible": False,
                    "forecasting_eligible": False,
                    "proxy_eligible": False,
                    "categorization_eligible": False,
                    "reasons": ["Normalize the raw dataset before starting a task"],
                },
            }
            paths = {"raw": str(raw_dir)}
            update_dataset(dataset_id, status="completed", summary_json=summary, paths_json=paths, error=None)
            return {"dataset_id": dataset_id, "files": connector.list_files(), "revision": revision}
        except Exception as exc:
            update_dataset(dataset_id, status="failed", error=str(exc))
            raise

    return run


def normalize_dataset_job(dataset_id: str, options: dict[str, Any]):
    def run(job_id: str, cancel_event: threading.Event) -> dict[str, Any]:
        dataset = get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        source_id = dataset["config"].get("source_id")
        source = get_source(source_id)
        if source is None:
            raise KeyError(f"Unknown source for dataset: {source_id}")
        raw_path = Path(dataset["paths"]["raw"])
        files = [path for path in raw_path.rglob("*") if path.is_file() and ".cache" not in path.parts]
        normalized_dir = NORMALIZED_DIR / dataset_id
        try:
            update_dataset(dataset_id, status="normalizing", error=None)
            update_job(job_id, status="normalizing", progress=0.1, message="Mapping source columns")
            if cancel_event.is_set():
                raise RuntimeError("Normalization cancelled")
            adapter_options = {**(dataset["config"].get("options") or {}), **options}
            adapter = create_adapter(source, adapter_options)
            result = adapter.normalize(files, normalized_dir)
            update_job(job_id, status="profiling", progress=0.75, message="Profiling canonical data")
            paths: dict[str, str] = {"raw": str(raw_path)}
            for key, path in (
                ("monthly_aggregates", result.canonical_path),
                ("liquidity", result.liquidity_path),
                ("target", result.target_path),
                ("categorization", result.categorization_path),
                ("forecast_series", result.forecast_path),
            ):
                if path:
                    paths[key] = str(path)
            if result.canonical_path:
                profile = profile_canonical(
                    result.canonical_path,
                    liquidity_path=result.liquidity_path,
                    target_path=result.target_path,
                    extra_paths=[path for path in (result.categorization_path, result.forecast_path) if path],
                )
            else:
                category_rows = 0
                if result.categorization_path:
                    import pandas as pd

                    category_rows = len(pd.read_parquet(result.categorization_path, columns=["category"]))
                profile = {
                    "rows": category_rows, "clients": 0, "months": 0, "labels": 0,
                    "has_debit": False, "has_credit": False, "has_balance": False,
                    "has_cash_gap_target": False, "multiple_currencies": False,
                    "missing_percent": 0.0,
                    "size_bytes": result.categorization_path.stat().st_size if result.categorization_path else 0,
                }
            compatibility = build_compatibility_report(profile, source)
            summary = {
                **profile,
                "stage": "normalized",
                "source_id": source_id,
                "source_provider": source["provider"],
                "mapping": result.mapping,
                "quality_flags": result.quality_flags,
                "compatibility": compatibility,
            }
            update_dataset(dataset_id, status="completed", summary_json=summary, paths_json=paths, error=None)
            return {"dataset_id": dataset_id, "profile": profile, "compatibility": compatibility}
        except Exception as exc:
            update_dataset(dataset_id, status="failed", error=str(exc))
            raise

    return run
