from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from app.config import MODEL_REGISTRY_PATH
from app.jobs.manager import list_jobs
from app.model_plugins import create_model_plugin
from app.models_registry.schemas import ModelSpec, ModelStatus


@lru_cache(maxsize=1)
def load_model_specs() -> tuple[ModelSpec, ...]:
    payload = yaml.safe_load(MODEL_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("config/models.yaml must contain a models list")
    models = tuple(ModelSpec.model_validate(item) for item in raw_models)
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("Model ids must be unique")
    return models


def get_model_spec(model_id: str) -> ModelSpec | None:
    return next((model for model in load_model_specs() if model.id == model_id), None)


def _installing_model_ids() -> set[str]:
    terminal = {"completed", "failed", "cancelled"}
    return {
        str((job.get("options") or {}).get("model_id"))
        for job in list_jobs()
        if job.get("job_type") == "model_install" and job.get("status") not in terminal
    }


def _describe(spec: ModelSpec, installing: set[str] | None = None) -> dict[str, Any]:
    plugin = create_model_plugin(spec)
    environment = plugin.check_environment()
    if spec.id in (installing or set()):
        environment = environment.model_copy(
            update={"status": ModelStatus.INSTALLING, "message": "Модель устанавливается в фоновом режиме"}
        )
    return {**spec.model_dump(), "environment": environment.model_dump()}


def list_models() -> list[dict[str, Any]]:
    installing = _installing_model_ids()
    return [_describe(spec, installing) for spec in load_model_specs()]


def get_model(model_id: str) -> dict[str, Any] | None:
    spec = get_model_spec(model_id)
    return _describe(spec, _installing_model_ids()) if spec else None
