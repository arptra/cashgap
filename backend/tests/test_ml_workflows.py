from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.forecasting import temporal_forecast_split, train_forecaster
from app.ml.training import train_and_evaluate
from app.services.training import _split_months


def test_temporal_classification_split_is_ordered_and_disjoint() -> None:
    months = [f"2023-{month:02d}" for month in range(1, 13)]
    train, validation, test = _split_months(months)
    assert train[-1] < validation[0] < test[0]
    assert set(train).isdisjoint(validation) and set(validation).isdisjoint(test)


def test_classification_pipeline_trains_and_returns_required_metrics() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(240, 4)), columns=list("abcd"))
    target = ((frame["a"] + frame["b"] * 0.4) > 0).astype(int).to_numpy()
    result = train_and_evaluate(
        "logistic_regression", {"max_iter": 200},
        frame.iloc[:140], target[:140], frame.iloc[140:190], target[140:190],
        frame.iloc[190:], target[190:], 42,
    )
    assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score", "precision_at_top_10", "recall_at_top_10"} <= set(result.metrics)
    assert len(result.predictions) == 50


def test_statsforecast_seasonal_naive_uses_last_period_as_test() -> None:
    dates = pd.date_range("2021-01-01", periods=24, freq="MS")
    series = pd.DataFrame({"date": dates, "y": np.tile([100.0, 120.0, 90.0, 110.0, 130.0, 95.0], 4)})
    train, test = temporal_forecast_split(series, horizon=4)
    assert train["date"].max() < test["date"].min()
    result = train_forecaster("seasonal_naive", series, {"horizon": 4, "season_length": 6})
    assert len(result.predictions) == 4
    assert {"mae", "rmse", "wape", "mase", "training_seconds"} <= set(result.metrics)
    assert np.isfinite(result.predictions[["forecast", "lower_bound", "upper_bound"]].to_numpy()).all()

