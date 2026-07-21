from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app


def _wait(client: TestClient, path: str, *, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(path).json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


def test_source_list_is_seeded_without_network_or_credentials() -> None:
    with TestClient(app) as client:
        response = client.get("/api/sources")
    assert response.status_code == 200
    assert len(response.json()) >= 8
    assert {item["id"] for item in response.json()} >= {"kaggle_paysim", "hf_paysim_banks"}


def test_synthetic_generation_and_experiment_start() -> None:
    with TestClient(app) as client:
        generated = client.post(
            "/api/synthetic/generate",
            json={"n_clients": 40, "n_months": 12, "target_gap_rate": 0.15, "random_seed": 11, "noise_level": 0.1, "overdraft_share": 0.5},
        )
        assert generated.status_code == 202
        request = generated.json()
        job = _wait(client, f"/api/jobs/{request['job_id']}")
        assert job["status"] == "completed", job.get("error")
        dataset = client.get(f"/api/datasets/{request['dataset_id']}").json()
        assert dataset["status"] == "completed"
        assert dataset["summary"]["compatibility"]["classification_eligible"] is True

        started = client.post(
            "/api/experiments/start",
            json={"dataset_id": request["dataset_id"], "task": "cash_gap_classification", "models": ["dummy"]},
        )
        assert started.status_code == 202, started.text
        run_id = started.json()["run_ids"][0]
        run = _wait(client, f"/api/experiments/{run_id}")
        assert run["status"] == "completed", run.get("error")
        assert "pr_auc" in client.get(f"/api/experiments/{run_id}/metrics").json()
        predictions = client.get(f"/api/experiments/{run_id}/predictions?limit=5")
        assert predictions.status_code == 200
        assert predictions.json()["items"]


def test_model_first_forecast_and_classification_benchmarks() -> None:
    with TestClient(app) as client:
        generated = client.post(
            "/api/synthetic/generate",
            json={"n_clients": 40, "n_months": 12, "target_gap_rate": 0.18, "random_seed": 29, "noise_level": 0.1, "overdraft_share": 0.5},
        )
        assert generated.status_code == 202
        dataset_id = generated.json()["dataset_id"]
        generated_job = _wait(client, f"/api/jobs/{generated.json()['job_id']}")
        assert generated_job["status"] == "completed", generated_job.get("error")

        catalog = client.get("/api/models")
        assert catalog.status_code == 200
        assert {model["id"] for model in catalog.json()} >= {
            "seasonal_naive_local",
            "shell_style_catboost_lag",
            "chronos_bolt_tiny",
            "logistic_cashgap",
            "catboost_cashgap",
        }
        compatibility = client.post(
            "/api/models/chronos_bolt_tiny/compatibility",
            json={"dataset_id": dataset_id, "target": "net_flow", "series_level": "client", "horizon": 1, "min_history": 8},
        )
        assert compatibility.status_code == 200
        assert compatibility.json()["compatible"] is True
        assert compatibility.json()["estimated_series"] == 40

        forecast = client.post(
            "/api/benchmarks/start",
            json={
                "dataset_id": dataset_id,
                "task": "cash_flow_forecasting",
                "target": "net_flow",
                "model_ids": ["seasonal_naive_local", "shell_style_catboost_lag"],
                "series_level": "client",
                "horizon": 1,
                "min_history": 8,
                "parameters": {"shell_style_catboost_lag": {"iterations": 12, "depth": 3}},
            },
        )
        assert forecast.status_code == 202, forecast.text
        forecast_job = _wait(client, f"/api/model-jobs/{forecast.json()['job_id']}", timeout=60)
        assert forecast_job["status"] == "completed", forecast_job.get("error")
        forecast_comparison = client.get(f"/api/benchmarks/{forecast.json()['benchmark_id']}/comparison").json()
        assert forecast_comparison["comparison_contract"]["task"] == "cash_flow_forecasting"
        assert {run["model_name"] for run in forecast_comparison["runs"]} == {
            "seasonal_naive_local",
            "shell_style_catboost_lag",
        }
        assert all(run["status"] == "completed" for run in forecast_comparison["runs"])
        assert all("mae" in run["metrics"] for run in forecast_comparison["runs"])

        classification = client.post(
            "/api/benchmarks/start",
            json={
                "dataset_id": dataset_id,
                "task": "cash_gap_classification",
                "target": "cash_gap_next_month",
                "model_ids": ["logistic_cashgap", "catboost_cashgap"],
                "series_level": "client",
                "horizon": 1,
                "min_history": 6,
                "parameters": {
                    "logistic_cashgap": {"max_iter": 100},
                    "catboost_cashgap": {"iterations": 20, "depth": 3},
                },
            },
        )
        assert classification.status_code == 202, classification.text
        classification_job = _wait(client, f"/api/model-jobs/{classification.json()['job_id']}", timeout=60)
        assert classification_job["status"] == "completed", classification_job.get("error")
        class_comparison = client.get(f"/api/benchmarks/{classification.json()['benchmark_id']}/comparison").json()
        assert class_comparison["comparison_contract"]["task"] == "cash_gap_classification"
        assert all(run["status"] == "completed" for run in class_comparison["runs"])
        assert all("pr_auc" in run["metrics"] for run in class_comparison["runs"])

        mixed = client.post(
            "/api/benchmarks/start",
            json={
                "dataset_id": dataset_id,
                "task": "cash_flow_forecasting",
                "target": "net_flow",
                "model_ids": ["seasonal_naive_local", "logistic_cashgap"],
            },
        )
        assert mixed.status_code == 409
