from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_INSTALLED = "NOT_INSTALLED"
    INSTALLING = "INSTALLING"
    INSTALLED = "INSTALLED"
    INCOMPATIBLE = "INCOMPATIBLE"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    FAILED = "FAILED"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    type: Literal["competition_recipe", "pretrained_model", "local_trainable_model"]
    provider: str
    task: Literal["cash_flow_forecasting", "cash_gap_classification"]
    plugin: str
    requires_training: bool
    cpu_supported: bool
    compatible_targets: list[str] = Field(default_factory=list)
    description: str = ""
    model_id: str | None = None
    kernel_ref: str | None = None
    source_url: str | None = None
    license: str | None = None
    dependency: str | None = None
    install_command: str | None = None
    supports_zero_shot: bool = False
    supports_multivariate: bool = False
    supports_covariates: bool = False
    limitations: list[str] = Field(default_factory=list)


class EnvironmentReport(BaseModel):
    status: ModelStatus
    message: str
    installed: bool = False
    dependency_installed: bool = True
    weights_cached: bool = False
    size_bytes: int = 0
    revision: str | None = None
    install_command: str | None = None


class ModelCompatibility(BaseModel):
    compatible: bool
    reasons: list[str] = Field(default_factory=list)
    target: str | None = None
    task: str | None = None
    estimated_series: int = 0
    estimated_memory_mb: float = 0.0
    device: str = "cpu"
    requires_training: bool = False
    requires_target: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
