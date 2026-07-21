from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from app.canonical.schemas import CANONICAL_COLUMNS, validate_canonical_frame


GROUP_COLUMNS = [
    "client_id",
    "month",
    "transaction_label",
    "source_dataset",
    "source_provider",
    "currency",
]


def _join_flags(values: pd.Series) -> str | None:
    flags: set[str] = set()
    for raw in values.dropna().astype(str):
        flags.update(item for item in raw.split("|") if item)
    return "|".join(sorted(flags)) if flags else None


def aggregate_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    validated = validate_canonical_frame(frame)
    # dropna=False is essential: rows without a known currency must not disappear.
    result = (
        validated.groupby(GROUP_COLUMNS, dropna=False, observed=True)
        .agg(
            debit_sum=("debit_sum", "sum"),
            credit_sum=("credit_sum", "sum"),
            debit_nonzero_count=("debit_nonzero_count", "sum"),
            credit_nonzero_count=("credit_nonzero_count", "sum"),
            data_quality_flags=("data_quality_flags", _join_flags),
        )
        .reset_index()
    )
    return validate_canonical_frame(result)


def combine_canonical_parts(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(parts)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return aggregate_canonical(pd.concat(frames, ignore_index=True))


def save_canonical(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_canonical(frame).to_parquet(path, index=False, compression="zstd")
    return path


def require_fx_for_mixed_currency(frame: pd.DataFrame, combine_currencies: bool, fx_rates: pd.DataFrame | None) -> None:
    currencies = frame["currency"].dropna().astype(str).unique()
    if combine_currencies and len(currencies) > 1 and fx_rates is None:
        raise ValueError("Multiple currencies cannot be combined without an explicit FX rates table")

