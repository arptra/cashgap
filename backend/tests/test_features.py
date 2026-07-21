from __future__ import annotations

import pandas as pd

from app.ml.features import build_feature_frame, feature_columns
from app.ml.synthetic import SyntheticConfig, generate_synthetic_dataset


def test_features_are_point_in_time_and_include_required_signals(tmp_path) -> None:
    generate_synthetic_dataset(
        SyntheticConfig(n_clients=20, n_months=12, random_seed=19, target_gap_rate=0.1), tmp_path
    )
    monthly = pd.read_parquet(tmp_path / "monthly_aggregates.parquet")
    target = pd.read_parquet(tmp_path / "target.parquet")
    original = build_feature_frame(monthly, target)

    future_month = sorted(monthly["month"].astype(str).unique())[-2]
    changed = monthly.copy()
    changed.loc[changed["month"].astype(str) == future_month, "credit_sum"] *= 100
    rebuilt = build_feature_frame(changed, target)
    earlier_month = sorted(original["month"].unique())[-3]
    left = original[original["month"] == earlier_month].sort_values("client_id").reset_index(drop=True)
    right = rebuilt[rebuilt["month"] == earlier_month].sort_values("client_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

    columns = set(feature_columns(original))
    assert {"net_flow", "total_credit_lag_6", "total_debit_mean_3", "credit_trend_3"} <= columns
    assert {"negative_flow_months_6", "negative_flow_streak", "credit_volatility_6"} <= columns

