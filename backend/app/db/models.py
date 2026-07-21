from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SourceRecord(Base):
    __tablename__ = "source_registry"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    remote_id: Mapped[str | None] = mapped_column(String)
    adapter: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    cash_gap_target: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    access_status: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    access_message: Mapped[str | None] = mapped_column(Text)
    source_revision: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class JobRecord(Base):
    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    source_id: Mapped[str | None] = mapped_column(String, ForeignKey("source_registry.id"))
    dataset_id: Mapped[str | None] = mapped_column(String)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    options_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class ArtifactRecord(Base):
    __tablename__ = "experiment_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class BenchmarkRecord(Base):
    __tablename__ = "model_benchmarks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    task: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    series_level: Mapped[str] = mapped_column(String, nullable=False, default="client")
    horizon: Mapped[int] = mapped_column(nullable=False, default=1)
    min_history: Mapped[int] = mapped_column(nullable=False, default=6)
    model_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    run_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
