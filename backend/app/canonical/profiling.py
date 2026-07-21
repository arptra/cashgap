from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def profile_canonical(
    canonical_path: Path,
    *,
    liquidity_path: Path | None = None,
    target_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    escaped = str(canonical_path).replace("'", "''")
    with duckdb.connect() as conn:
        row = conn.execute(
            f"""
            SELECT
              count(*) AS rows,
              count(DISTINCT client_id) AS clients,
              min(month) AS first_month,
              max(month) AS last_month,
              count(DISTINCT month) AS months,
              count(DISTINCT transaction_label) AS labels,
              max(CASE WHEN debit_sum > 0 THEN 1 ELSE 0 END) AS has_debit,
              max(CASE WHEN credit_sum > 0 THEN 1 ELSE 0 END) AS has_credit,
              count(DISTINCT currency) FILTER (WHERE currency IS NOT NULL) AS currencies,
              avg(
                (client_id IS NULL)::INT + (month IS NULL)::INT +
                (debit_sum IS NULL)::INT + (credit_sum IS NULL)::INT
              ) / 4.0 AS missing_fraction
            FROM read_parquet('{escaped}')
            """
        ).fetchone()
    size_paths = [canonical_path, *(extra_paths or [])]
    if liquidity_path:
        size_paths.append(liquidity_path)
    if target_path:
        size_paths.append(target_path)
    return {
        "rows": int(row[0] or 0),
        "clients": int(row[1] or 0),
        "first_month": row[2],
        "last_month": row[3],
        "months": int(row[4] or 0),
        "labels": int(row[5] or 0),
        "has_debit": bool(row[6]),
        "has_credit": bool(row[7]),
        "has_balance": bool(liquidity_path and liquidity_path.exists()),
        "multiple_currencies": int(row[8] or 0) > 1,
        "currency_count": int(row[8] or 0),
        "has_cash_gap_target": bool(target_path and target_path.exists()),
        "missing_percent": float(row[9] or 0.0) * 100,
        "size_bytes": sum(path.stat().st_size for path in size_paths if path.exists()),
    }


def profile_raw_files(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts]
    return {
        "stage": "raw",
        "files": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "file_list": [{"name": str(path.relative_to(root)), "size_bytes": path.stat().st_size} for path in files],
    }

