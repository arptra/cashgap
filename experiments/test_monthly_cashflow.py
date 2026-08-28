from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from experiments.benchmark_monthly_cashflow import build_monthly_dataset, predict_linear
    from experiments.monthly_model_runtime import MonthlyCashflowRuntime
    from experiments.monthly_objective import (
        MONTHLY_OBJECTIVE_VERSION,
        baseline_from_feature_matrix,
        fit_feature_normalizer,
        fit_residual_scale,
        fit_money_scale,
        monthly_regression_metrics,
        normalize_features,
        restore_residual_predictions,
        restore_money_predictions,
        scale_residual_targets,
        scale_money_targets,
    )
except ImportError:
    from benchmark_monthly_cashflow import build_monthly_dataset, predict_linear
    from monthly_model_runtime import MonthlyCashflowRuntime
    from monthly_objective import (
        MONTHLY_OBJECTIVE_VERSION,
        baseline_from_feature_matrix,
        fit_feature_normalizer,
        fit_residual_scale,
        fit_money_scale,
        monthly_regression_metrics,
        normalize_features,
        restore_residual_predictions,
        restore_money_predictions,
        scale_residual_targets,
        scale_money_targets,
    )


def test_raw_money_transform_round_trip_preserves_rubles() -> None:
    targets = np.asarray([[0.0, 10.0], [100.0, 1_000.0], [10_000.0, 50.0]])
    scale = fit_money_scale(targets)
    restored = restore_money_predictions(scale_money_targets(targets, scale), scale)
    np.testing.assert_allclose(restored, targets, rtol=1e-6, atol=1e-5)
    assert MONTHLY_OBJECTIVE_VERSION >= 2


def test_monthly_total_metric_is_a_plain_percentage_error() -> None:
    actual = np.asarray([[40.0, 60.0], [60.0, 40.0]])
    predicted = np.asarray([[44.0, 54.0], [66.0, 36.0]])
    rows = monthly_regression_metrics(actual, predicted, zero_floor=1.0)
    assert rows[0]["aggregate_mape_percent"] == 10.0
    assert rows[1]["aggregate_mape_percent"] == 10.0
    assert rows[0]["bias_percent"] == 10.0
    assert rows[1]["bias_percent"] == -10.0


def test_residual_objective_starts_from_the_trailing_mean_baseline() -> None:
    feature_names = ["other", "target_inflow_mean_3", "target_outflow_mean_3"]
    features = np.asarray([[99.0, 100.0, 80.0], [98.0, 200.0, 120.0]])
    baseline = baseline_from_feature_matrix(features, feature_names)
    actual = np.asarray([[110.0, 70.0], [180.0, 150.0]])
    residual = actual - baseline
    scale = fit_residual_scale(residual)
    restored = restore_residual_predictions(
        baseline, scale_residual_targets(residual, scale), scale
    )
    np.testing.assert_allclose(restored, actual, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(
        restore_residual_predictions(baseline, np.zeros_like(actual), scale), baseline
    )


def test_unseen_value_in_constant_training_feature_cannot_explode() -> None:
    train = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mean, scale, active = fit_feature_normalizer(train)
    future = normalize_features(np.asarray([[4.0, 9_000_000.0]]), mean, scale, active)
    assert active.tolist() == [True, False]
    assert future[0, 1] == 0.0
    assert abs(float(future[0, 0])) <= 10.0


def test_linear_model_accepts_an_all_constant_early_fold() -> None:
    columns = ["target_inflow_mean_3", "target_outflow_mean_3", "other"]
    features = pd.DataFrame(np.zeros((4, 3)), columns=columns)
    train = pd.DataFrame({
        "target_inflow": [0.0, 0.0, 0.0, 0.0],
        "target_outflow": [0.0, 0.0, 0.0, 0.0],
    })
    prediction = predict_linear(features, train, features.iloc[:2], jobs=1)
    np.testing.assert_allclose(prediction, np.zeros((2, 2)))


def test_standalone_runtime_restores_network_and_ruble_baseline(tmp_path) -> None:
    features = ["target_inflow_mean_3", "target_outflow_mean_3", "history_months"]
    model = nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(), nn.Dropout(0.1), nn.Linear(4, 2)
    )
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    checkpoint = {
        "format_version": 2,
        "model_id": "torch_mlp_2_layers",
        "objective_version": 2,
        "objective_name": "test",
        "features": features,
        "layers": [4],
        "activation": "relu",
        "dropout": 0.1,
        "feature_mean": [0.0, 0.0, 0.0],
        "feature_scale": [1.0, 1.0, 1.0],
        "active_features": [True, True, False],
        "residual_scale": [100.0, 200.0],
        "state_dict": model.state_dict(),
    }
    model_path = tmp_path / "model.pt"
    torch.save(checkpoint, model_path)
    runtime = MonthlyCashflowRuntime(str(model_path), device="cpu", cpu_threads=1)
    raw = np.asarray([[120.0, 80.0, 9_000_000.0], [75.0, 95.0, 1.0]])
    prediction = runtime.predict_matrix(raw, batch_size=1)
    np.testing.assert_allclose(prediction, raw[:, :2])
    assert runtime.predict_one(dict(zip(features, raw[0]))) == {
        "predicted_inflow": 120.0,
        "predicted_outflow": 80.0,
    }
    history = np.zeros((1, 12, 6), dtype=np.float64)
    history[0, -3:, 0] = [100.0, 120.0, 140.0]
    history[0, -3:, 1] = [50.0, 60.0, 70.0]
    history[0, :, 2] = history[0, :, 0] - history[0, :, 1]
    history_prediction = runtime.predict_history(history, [12], [7])
    np.testing.assert_allclose(history_prediction, [[120.0, 60.0]])
    self_test_path = tmp_path / "runtime_self_test.npz"
    np.savez_compressed(
        self_test_path, raw_features=raw, expected_predictions=prediction
    )
    assert runtime.run_self_test(str(self_test_path))["status"] == "OK"


def test_monthly_grid_keeps_dormant_company_without_future_survivorship() -> None:
    daily = pd.DataFrame({
        "inn": ["A", "B", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-03-31"]),
        "inflow": [100.0, 50.0, 70.0],
        "outflow": [20.0, 10.0, 30.0],
    })
    daily["net_flow"] = daily["inflow"] - daily["outflow"]
    monthly, features = build_monthly_dataset(daily)

    company_a = monthly[monthly["inn"].eq("A")].sort_values("month")
    assert company_a["month"].dt.strftime("%Y-%m").tolist() == [
        "2024-01", "2024-02", "2024-03"
    ]
    assert company_a["target_inflow"].tolist() == [100.0, 0.0, 0.0]
    assert company_a["target_inflow_lag_1"].iloc[-1] == 0.0
    assert not {
        "target_inflow", "target_outflow", "target_net_flow"
    }.intersection(features)
