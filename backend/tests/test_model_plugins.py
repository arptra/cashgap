from __future__ import annotations

import time
from uuid import uuid4

import numpy as np

from app.db.base import init_orm
from app.model_plugins import create_model_plugin
from app.models_registry.registry import get_model_spec
from app.services.datasets import generate_dataset_job
from app.storage.database import get_dataset, init_db, insert_dataset


def _dataset() -> dict:
    init_db()
    init_orm()
    dataset_id = f"plugin_{uuid4().hex[:10]}"
    config = {
        "n_clients": 20,
        "n_months": 12,
        "random_seed": 73,
        "target_gap_rate": 0.18,
        "noise_level": 0.1,
        "overdraft_share": 0.5,
    }
    insert_dataset(dataset_id, config)
    generate_dataset_job(dataset_id, config)
    dataset = get_dataset(dataset_id)
    assert dataset and dataset["status"] == "completed"
    return dataset


def test_forecast_baselines_use_real_series_and_keep_shell_flow_identity() -> None:
    dataset = _dataset()
    options = {
        "target": "net_flow",
        "series_level": "client",
        "horizon": 1,
        "min_history": 8,
        "parameters": {"iterations": 12, "depth": 3},
    }
    seasonal_spec = get_model_spec("seasonal_naive_local")
    shell_spec = get_model_spec("shell_style_catboost_lag")
    assert seasonal_spec and shell_spec
    seasonal = create_model_plugin(seasonal_spec).run(dataset, options)
    shell = create_model_plugin(shell_spec).run(dataset, options)
    assert len(seasonal.predictions) == dataset["summary"]["n_clients"]
    assert {"mae", "rmse", "wape", "mase", "processed_series", "failed_series"} <= set(seasonal.metrics)
    assert np.isfinite(shell.predictions[["actual", "forecast"]].to_numpy()).all()
    assert np.allclose(
        shell.predictions["predicted_net_flow"],
        shell.predictions["predicted_credit"] - shell.predictions["predicted_debit"],
    )


def test_chronos_adapter_calls_quantile_api_without_a_random_fallback() -> None:
    dataset = _dataset()
    spec = get_model_spec("chronos_bolt_tiny")
    assert spec
    plugin = create_model_plugin(spec)
    prepared = plugin.prepare_data(dataset, target="net_flow", series_level="client", horizon=1, min_history=8)

    class FakeChronos:
        def predict_quantiles(self, contexts, prediction_length, quantile_levels):
            batch = len(contexts)
            assert prediction_length == 1
            assert quantile_levels == [0.1, 0.5, 0.9]
            point = np.asarray([[float(context[-1])] for context in contexts])
            quantiles = np.stack((point - 1.0, point, point + 1.0), axis=-1)
            return quantiles, point

    plugin._model = FakeChronos()
    plugin._started = time.perf_counter()
    predictions = plugin.predict(prepared, {"batch_size": 7})
    metrics = plugin.evaluate(predictions, prepared)
    assert len(predictions) == dataset["summary"]["n_clients"]
    assert predictions["forecast"].notna().all()
    assert metrics["failed_series"] == 0


def test_timesfm_adapter_uses_point_and_quantile_outputs() -> None:
    dataset = _dataset()
    spec = get_model_spec("timesfm_2_5")
    assert spec
    plugin = create_model_plugin(spec)
    prepared = plugin.prepare_data(dataset, target="total_credit_sum", series_level="client", horizon=1, min_history=8)

    class FakeTimesFm:
        def forecast(self, *, horizon, inputs):
            assert horizon == 1
            batch = len(inputs)
            point = np.asarray([[float(values[-1])] for values in inputs])
            quantiles = np.zeros((batch, horizon, 10), dtype=float)
            quantiles[:, :, 1] = point - 2.0
            quantiles[:, :, 5] = point
            quantiles[:, :, 9] = point + 2.0
            return point, quantiles

    plugin._model = FakeTimesFm()
    plugin._started = time.perf_counter()
    predictions = plugin.predict(prepared, {"batch_size": 6})
    assert len(predictions) == dataset["summary"]["n_clients"]
    assert np.all(predictions["lower_bound"] <= predictions["median"])
    assert np.all(predictions["median"] <= predictions["upper_bound"])
    assert plugin.evaluate(predictions, prepared)["failed_series"] == 0
