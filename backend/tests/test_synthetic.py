from __future__ import annotations

import pandas as pd

from app.ml.synthetic import CATEGORIES, SyntheticConfig, generate_synthetic_dataset


def test_synthetic_dataset_has_expected_layers(tmp_path) -> None:
    config = SyntheticConfig(n_clients=24, n_months=12, random_seed=7, target_gap_rate=0.12, noise_level=0.1)
    summary = generate_synthetic_dataset(config, tmp_path)
    daily = pd.read_parquet(tmp_path / "daily_hidden.parquet")
    monthly = pd.read_parquet(tmp_path / "monthly_aggregates.parquet")
    target = pd.read_parquet(tmp_path / "target.parquet")

    assert summary["n_clients"] == 24
    assert summary["monthly_rows"] == 24 * 12 * len(CATEGORIES)
    assert set(CATEGORIES) == set(monthly["transaction_label"].unique())
    assert {"opening_balance", "closing_balance", "available_overdraft", "cash_gap"} <= set(daily.columns)
    assert target["cash_gap_next_month"].notna().sum() == 24 * 11
    assert 0 <= summary["target_rate"] <= 1

