from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DatasetGenerateRequest(BaseModel):
    n_clients: int = Field(default=3000, ge=10, le=100_000)
    n_months: int = Field(default=24, ge=12, le=120)
    random_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    target_gap_rate: float = Field(default=0.10, ge=0.01, le=0.50)
    noise_level: float = Field(default=0.15, ge=0.0, le=1.0)
    overdraft_share: float = Field(default=0.55, ge=0.0, le=1.0)


ModelName = Literal["dummy", "logistic_regression", "random_forest", "catboost", "lightgbm"]


class TrainingStartRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    models: list[ModelName] = Field(default_factory=lambda: ["dummy", "logistic_regression"])
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def unique_models(cls, value: list[ModelName]) -> list[ModelName]:
        if not value:
            raise ValueError("Select at least one model")
        return list(dict.fromkeys(value))


class QueuedResponse(BaseModel):
    id: str
    status: str = "queued"


class TrainingQueuedResponse(BaseModel):
    run_ids: list[str]
    status: str = "queued"
