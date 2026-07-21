from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import yaml
from sqlalchemy import select

from app.config import DATASET_REGISTRY_PATH
from app.db.base import SessionLocal
from app.db.models import SourceRecord


def _load_yaml() -> list[dict[str, Any]]:
    if not DATASET_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Dataset registry not found: {DATASET_REGISTRY_PATH}")
    payload = yaml.safe_load(DATASET_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("config/datasets.yaml must contain a 'datasets' list")
    ids = [item.get("id") for item in datasets]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Every source must have a unique non-empty id")
    return datasets


def seed_registry() -> None:
    configured = _load_yaml()
    with SessionLocal() as session:
        existing = {row.id: row for row in session.scalars(select(SourceRecord)).all()}
        for item in configured:
            source = existing.get(item["id"])
            values = {
                "provider": item["provider"],
                "remote_id": item.get("remote_id"),
                "adapter": item["adapter"],
                "title": item.get("title", item["id"]),
                "config_json": json.dumps(item, ensure_ascii=False),
                "cash_gap_target": bool(item.get("cash_gap_target", False)),
                "updated_at": datetime.now(timezone.utc),
            }
            if source is None:
                session.add(SourceRecord(id=item["id"], **values))
            else:
                for key, value in values.items():
                    setattr(source, key, value)
        session.commit()


def _as_dict(source: SourceRecord) -> dict[str, Any]:
    config = json.loads(source.config_json)
    return {
        **config,
        "id": source.id,
        "title": source.title,
        "provider": source.provider,
        "remote_id": source.remote_id,
        "adapter": source.adapter,
        "cash_gap_target": source.cash_gap_target,
        "access_status": source.access_status,
        "access_message": source.access_message,
        "source_revision": source.source_revision,
        "updated_at": source.updated_at.isoformat() if source.updated_at else None,
    }


def list_sources() -> list[dict[str, Any]]:
    with SessionLocal() as session:
        sources = session.scalars(select(SourceRecord).order_by(SourceRecord.id)).all()
        return [_as_dict(source) for source in sources]


def get_source(source_id: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        source = session.get(SourceRecord, source_id)
        return _as_dict(source) if source else None


def update_source_access(
    source_id: str,
    status: str,
    message: str | None,
    revision: str | None = None,
) -> None:
    with SessionLocal() as session:
        source = session.get(SourceRecord, source_id)
        if source is None:
            raise KeyError(source_id)
        source.access_status = status
        source.access_message = message
        if revision:
            source.source_revision = revision
        source.updated_at = datetime.now(timezone.utc)
        session.commit()

