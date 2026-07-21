from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.synthetic import CATEGORIES


BASE_VALUE_COLUMNS = ["debit_sum", "credit_sum", "debit_nonzero_count", "credit_nonzero_count"]


def _negative_streak(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=bool)
    result = np.zeros(len(values), dtype=np.int16)
    running = 0
    for index, value in enumerate(values):
        running = running + 1 if value else 0
        result[index] = running
    return pd.Series(result, index=series.index)


def build_feature_frame(monthly: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features using current and historical months only."""

    required = {"client_id", "month", "transaction_label", *BASE_VALUE_COLUMNS}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"Monthly data is missing columns: {sorted(missing)}")

    monthly = monthly.copy()
    monthly["month"] = monthly["month"].astype(str)
    keys = ["client_id", "month"]
    totals = monthly.groupby(keys, observed=True)[BASE_VALUE_COLUMNS].sum().reset_index()
    totals = totals.rename(
        columns={
            "debit_sum": "total_debit",
            "credit_sum": "total_credit",
            "debit_nonzero_count": "debit_count",
            "credit_nonzero_count": "credit_count",
        }
    )
    totals["net_flow"] = totals["total_credit"] - totals["total_debit"]
    totals["avg_credit_operation"] = totals["total_credit"] / totals["credit_count"].clip(lower=1)
    totals["avg_debit_operation"] = totals["total_debit"] / totals["debit_count"].clip(lower=1)
    totals["credit_to_debit_ratio"] = totals["total_credit"] / totals["total_debit"].clip(lower=1.0)
    totals["credit_to_debit_count_ratio"] = totals["credit_count"] / totals["debit_count"].clip(lower=1)

    active = monthly.assign(
        active=(monthly["debit_sum"] + monthly["credit_sum"] > 0).astype(np.int8)
    ).groupby(keys, observed=True)["active"].sum().rename("active_categories").reset_index()
    frame = totals.merge(active, on=keys, how="left")

    pivot = monthly.pivot_table(
        index=keys,
        columns="transaction_label",
        values=BASE_VALUE_COLUMNS,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot.columns = [f"{value}__{category}" for value, category in pivot.columns]
    pivot = pivot.reset_index()
    # Keep the schema stable even if uploaded data does not contain every category.
    for value in BASE_VALUE_COLUMNS:
        for category in CATEGORIES:
            column = f"{value}__{category}"
            if column not in pivot:
                pivot[column] = 0.0
    frame = frame.merge(pivot, on=keys, how="left")
    frame = frame.sort_values(keys).reset_index(drop=True)

    group = frame.groupby("client_id", sort=False)
    core = [
        "total_debit", "total_credit", "net_flow", "debit_count", "credit_count",
        "avg_credit_operation", "avg_debit_operation", "credit_to_debit_ratio",
    ]
    category_amounts = [
        f"{direction}_sum__{category}"
        for direction in ("debit", "credit")
        for category in CATEGORIES
    ]
    lag_sources = core + category_amounts
    for lag in (1, 2, 3, 6):
        shifted = group[lag_sources].shift(lag)
        shifted.columns = [f"{column}_lag_{lag}" for column in lag_sources]
        frame = pd.concat([frame, shifted], axis=1)

    group = frame.groupby("client_id", sort=False)
    for window in (3, 6):
        for column in core:
            frame[f"{column}_mean_{window}"] = group[column].transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            frame[f"{column}_std_{window}"] = group[column].transform(
                lambda values: values.rolling(window, min_periods=2).std()
            )

    for column in ("total_credit", "total_debit", "credit_count", "debit_count", "avg_debit_operation"):
        previous = frame[f"{column}_lag_1"]
        frame[f"{column}_change"] = frame[column] - previous
        frame[f"{column}_change_pct"] = (frame[column] - previous) / previous.abs().clip(lower=1.0)

    frame["credit_trend_3"] = (frame["total_credit"] - frame["total_credit_lag_2"]) / 2.0
    frame["debit_trend_3"] = (frame["total_debit"] - frame["total_debit_lag_2"]) / 2.0
    negative = frame["net_flow"] < 0
    frame["negative_flow_months_6"] = negative.groupby(frame["client_id"]).transform(
        lambda values: values.rolling(6, min_periods=1).sum()
    )
    frame["negative_flow_streak"] = negative.groupby(frame["client_id"], group_keys=False).apply(_negative_streak)
    frame["credit_count_decline"] = (-frame["credit_count_change_pct"]).clip(lower=0)
    frame["debit_count_growth"] = frame["debit_count_change_pct"].clip(lower=0)
    frame["credit_volatility_6"] = frame["total_credit_std_6"] / frame["total_credit_mean_6"].abs().clip(lower=1.0)

    target_columns = target[["client_id", "month", "cash_gap_next_month"]].copy()
    target_columns["month"] = target_columns["month"].astype(str)
    frame = frame.merge(target_columns, on=keys, how="inner")
    frame = frame[frame["cash_gap_next_month"].notna()].copy()
    frame["cash_gap_next_month"] = frame["cash_gap_next_month"].astype(np.int8)
    return frame.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"client_id", "month", "cash_gap_next_month"}
    return [column for column in frame.columns if column not in excluded]

