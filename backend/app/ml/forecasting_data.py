from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np
import pandas as pd


FORECAST_TARGETS = {"total_credit_sum", "total_debit_sum", "net_flow"}


@dataclass
class PreparedForecastData:
    context: pd.DataFrame
    test: pd.DataFrame
    target: str
    horizon: int
    series_level: str
    min_history: int

    @property
    def series_ids(self) -> list[str]:
        return self.context["series_id"].drop_duplicates().astype(str).tolist()

    def batches(self, batch_size: int) -> Iterator[list[str]]:
        ids = self.series_ids
        for start in range(0, len(ids), max(1, batch_size)):
            yield ids[start : start + max(1, batch_size)]


def _monthly_frame(path: Path, series_level: str) -> pd.DataFrame:
    escaped = str(path).replace("'", "''")
    if series_level == "client_category":
        series = "client_id || '::' || transaction_label"
        extra = ", transaction_label"
        grouping = "client_id, transaction_label, month"
    elif series_level == "client":
        series = "client_id"
        extra = ""
        grouping = "client_id, month"
    else:
        raise ValueError("series_level must be client or client_category")
    with duckdb.connect() as conn:
        return conn.execute(
            f"""
            SELECT {series} AS series_id, client_id {extra}, month,
                   sum(credit_sum)::DOUBLE AS total_credit_sum,
                   sum(debit_sum)::DOUBLE AS total_debit_sum,
                   (sum(credit_sum) - sum(debit_sum))::DOUBLE AS net_flow
            FROM read_parquet('{escaped}')
            GROUP BY {grouping}
            ORDER BY series_id, month
            """
        ).fetchdf()


def prepare_forecast_data(
    canonical_path: Path,
    *,
    target: str,
    series_level: str = "client",
    horizon: int = 1,
    min_history: int = 6,
) -> PreparedForecastData:
    if target not in FORECAST_TARGETS:
        raise ValueError(f"Unsupported forecasting target: {target}")
    if horizon not in {1, 2, 3}:
        raise ValueError("Prediction horizon must be 1, 2 or 3 months")
    frame = _monthly_frame(canonical_path, series_level)
    numeric = ["total_credit_sum", "total_debit_sum", "net_flow"]
    values = frame[numeric].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Forecasting data contains NaN or infinity")
    counts = frame.groupby("series_id")["month"].transform("size")
    frame = frame[counts >= min_history + horizon].copy()
    if frame.empty:
        raise ValueError(f"No series has at least {min_history + horizon} monthly observations")
    rank_from_end = frame.groupby("series_id").cumcount(ascending=False)
    test = frame[rank_from_end < horizon].copy()
    context = frame[rank_from_end >= horizon].copy()
    return PreparedForecastData(
        context=context.reset_index(drop=True),
        test=test.reset_index(drop=True),
        target=target,
        horizon=horizon,
        series_level=series_level,
        min_history=min_history,
    )


def evaluate_forecast_predictions(predictions: pd.DataFrame, context: pd.DataFrame, seconds: float) -> dict:
    if predictions.empty:
        raise ValueError("No forecasting predictions were produced")
    actual = predictions["actual"].to_numpy(float)
    forecast = predictions["forecast"].to_numpy(float)
    error = actual - forecast
    denominator = float(np.abs(actual).sum())
    scales = []
    for series_id, group in context.groupby("series_id", sort=False):
        values = group.sort_values("month")["target_value"].to_numpy(float)
        scale = float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0
        if scale > 0:
            series_error = predictions.loc[predictions["series_id"] == series_id, "actual"] - predictions.loc[predictions["series_id"] == series_id, "forecast"]
            if len(series_error):
                scales.append(float(np.abs(series_error).mean() / scale))
    successful = int(predictions["series_id"].nunique())
    attempted = int(predictions.attrs.get("attempted_series", successful))
    failed = max(attempted - successful, 0)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else None,
        "mase": float(np.mean(scales)) if scales else None,
        "training_seconds": float(seconds),
        "processed_series": successful,
        "failed_series": failed,
        "error_rate": float(failed / attempted) if attempted else 0.0,
        "test_rows": int(len(predictions)),
    }
