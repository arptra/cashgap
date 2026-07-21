from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import DB_PATH, ensure_directories


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT,
                paths_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL,
                params_json TEXT NOT NULL,
                metrics_json TEXT,
                feature_importance_json TEXT,
                feature_names_json TEXT,
                split_json TEXT,
                artifact_path TEXT,
                error TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id);
            """
        )
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "task" not in run_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN task TEXT NOT NULL DEFAULT 'cash_gap_classification'")


def insert_dataset(dataset_id: str, config: dict[str, Any]) -> None:
    now = utc_now()
    with connection() as conn:
        conn.execute(
            "INSERT INTO datasets(id, created_at, updated_at, status, config_json) VALUES (?, ?, ?, ?, ?)",
            (dataset_id, now, now, "queued", json.dumps(config, ensure_ascii=False)),
        )


def update_dataset(dataset_id: str, **fields: Any) -> None:
    allowed = {"status", "summary_json", "paths_json", "error"}
    values: dict[str, Any] = {"updated_at": utc_now()}
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported dataset field: {key}")
        values[key] = json.dumps(value, ensure_ascii=False) if key.endswith("_json") and value is not None else value
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connection() as conn:
        conn.execute(
            f"UPDATE datasets SET {assignments} WHERE id = ?",
            (*values.values(), dataset_id),
        )


def insert_run(
    run_id: str,
    dataset_id: str,
    model_name: str,
    params: dict[str, Any],
    task: str = "cash_gap_classification",
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO runs(id, dataset_id, model_name, status, created_at, params_json, task)
            VALUES (?, ?, ?, 'queued', ?, ?, ?)
            """,
            (run_id, dataset_id, model_name, utc_now(), json.dumps(params, ensure_ascii=False), task),
        )


def update_run(run_id: str, **fields: Any) -> None:
    allowed = {
        "status", "started_at", "completed_at", "duration_seconds", "metrics_json",
        "feature_importance_json", "feature_names_json", "split_json", "artifact_path", "error",
    }
    values: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported run field: {key}")
        values[key] = json.dumps(value, ensure_ascii=False) if key.endswith("_json") and value is not None else value
    if not values:
        return
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connection() as conn:
        conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*values.values(), run_id))


def _decode(row: sqlite3.Row | None, json_fields: tuple[str, ...]) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for field in json_fields:
        raw = result.pop(field, None)
        result[field.removesuffix("_json")] = json.loads(raw) if raw else None
    return result


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _decode(row, ("config_json", "summary_json", "paths_json"))


def list_datasets() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    return [_decode(row, ("config_json", "summary_json", "paths_json")) for row in rows]  # type: ignore[misc]


def get_run(run_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _decode(
        row,
        ("params_json", "metrics_json", "feature_importance_json", "feature_names_json", "split_json"),
    )


def list_runs() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return [
        _decode(
            row,
            ("params_json", "metrics_json", "feature_importance_json", "feature_names_json", "split_json"),
        )
        for row in rows
    ]  # type: ignore[misc]


def delete_run_record(run_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def delete_dataset_record(dataset_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))

