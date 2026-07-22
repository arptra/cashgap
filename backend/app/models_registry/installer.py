from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.config import COMPETITION_SOURCES_DIR, MODEL_CACHE_DIR
from app.models_registry.schemas import ModelSpec


SAFE_WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.spm", "*.py"]
BLOCKED_WEIGHT_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.joblib"]


def _retry_hub_call(action: Callable[[], Any], cancel_event: threading.Event | None = None) -> Any:
    try:
        attempts = max(1, int(os.getenv("CASHGAP_HF_DOWNLOAD_ATTEMPTS", "4")))
    except ValueError:
        attempts = 4
    for attempt in range(attempts):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Model installation cancelled")
        try:
            return action()
        except Exception:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(2**attempt, 15))


def model_install_dir(spec: ModelSpec) -> Path:
    return COMPETITION_SOURCES_DIR / spec.id if spec.type == "competition_recipe" else MODEL_CACHE_DIR / spec.id


def cache_metadata(spec: ModelSpec) -> dict[str, Any]:
    root = model_install_dir(spec)
    metadata_path = root / ".cashgap-model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    return {
        **metadata,
        # Only a completed, revision-pinned install writes this marker.  A
        # cancelled/failed snapshot may leave files behind, but must never be
        # presented as an installed model.
        "installed": metadata_path.exists(),
        "size_bytes": sum(path.stat().st_size for path in files),
        "path": str(root),
        "files": len(files),
    }


def install_pretrained(spec: ModelSpec, cancel_event: threading.Event | None = None) -> dict[str, Any]:
    if not spec.model_id:
        raise ValueError("pretrained_model requires model_id")
    from huggingface_hub import HfApi, snapshot_download

    token = os.getenv("HF_TOKEN") or None
    info = _retry_hub_call(lambda: HfApi(token=token).model_info(spec.model_id), cancel_event)
    revision = info.sha
    root = model_install_dir(spec)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{spec.id}-{revision}.partial"
    for stale_staging in root.parent.glob(f".{spec.id}-*.partial"):
        if stale_staging != staging:
            shutil.rmtree(stale_staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    if cancel_event and cancel_event.is_set():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("Model installation cancelled")
    try:
        _retry_hub_call(
            lambda: snapshot_download(
                repo_id=spec.model_id,
                revision=revision,
                token=token,
                local_dir=staging,
                allow_patterns=SAFE_WEIGHT_PATTERNS,
                ignore_patterns=BLOCKED_WEIGHT_PATTERNS,
            ),
            cancel_event,
        )
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Model installation cancelled")
        metadata = {
            "model_id": spec.model_id,
            "revision": revision,
            "source_url": spec.source_url,
            "license": spec.license,
            "safe_formats_only": True,
        }
        (staging / ".cashgap-model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if root.exists():
            shutil.rmtree(root)
        staging.replace(root)
    except Exception:
        # Keep partial downloads so a user retry resumes instead of losing
        # hundreds of megabytes after a transient proxy/DNS interruption.
        if cancel_event and cancel_event.is_set():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return cache_metadata(spec)


def _has_kaggle_credentials() -> bool:
    return bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".kaggle" / "access_token").exists()
    )


def install_competition_source(spec: ModelSpec, cancel_event: threading.Event | None = None) -> dict[str, Any]:
    if not spec.kernel_ref:
        raise ValueError("competition_recipe requires kernel_ref")
    if not _has_kaggle_credentials():
        raise PermissionError("Kaggle authentication required. Configure KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json")
    root = model_install_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Recipe connection cancelled")
    executable = shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Kaggle CLI is not installed. Run make setup")
    process = subprocess.run(
        [executable, "kernels", "pull", spec.kernel_ref, "--path", str(root), "--metadata"],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if process.returncode:
        message = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"Kaggle notebook is unavailable: {message}")
    metadata = {
        "kernel_ref": spec.kernel_ref,
        "source_url": spec.source_url,
        "attribution_only": True,
        "executed": False,
        "license": spec.license,
    }
    (root / "ATTRIBUTION.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / ".cashgap-model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_metadata(spec)


def install_model(spec: ModelSpec, cancel_event: threading.Event | None = None) -> dict[str, Any]:
    if spec.type == "pretrained_model":
        return install_pretrained(spec, cancel_event)
    if spec.type == "competition_recipe":
        return install_competition_source(spec, cancel_event)
    return {"installed": True, "size_bytes": 0, "path": None, "revision": None}


def uninstall_model(spec: ModelSpec) -> None:
    if spec.bundled:
        raise RuntimeError("Bundled offline model cannot be removed")
    root = model_install_dir(spec).resolve()
    allowed = {MODEL_CACHE_DIR.resolve(), COMPETITION_SOURCES_DIR.resolve()}
    if not any(root.is_relative_to(parent) for parent in allowed):
        raise RuntimeError("Refusing to remove a model outside managed directories")
    if root.exists():
        shutil.rmtree(root)
