from __future__ import annotations

import argparse
import time
from uuid import uuid4

from app.db.base import init_orm
from app.jobs.manager import create_job, get_job
from app.model_plugins import create_model_plugin
from app.models_registry.registry import get_model_spec
from app.models_registry.schemas import ModelStatus
from app.services.benchmarks import benchmark_comparison, benchmark_job, create_benchmark
from app.services.registry import seed_registry
from app.services.datasets import generate_dataset_job
from app.storage.database import get_dataset, init_db, insert_dataset


def _initialize() -> None:
    init_db()
    init_orm()
    seed_registry()


def _generate(config: dict) -> str:
    dataset_id = f"demo_{uuid4().hex[:10]}"
    insert_dataset(dataset_id, config)
    generate_dataset_job(dataset_id, config)
    result = get_dataset(dataset_id)
    if result is None or result["status"] != "completed":
        raise SystemExit(result["error"] if result else "Dataset record disappeared")
    return dataset_id


def _run_benchmark(config: dict) -> dict:
    benchmark = create_benchmark(config)
    job_id = create_job(
        "model_benchmark",
        benchmark_job(benchmark["id"], config),
        dataset_id=config["dataset_id"],
        options={"benchmark_id": benchmark["id"], "model_ids": config["model_ids"]},
    )
    while True:
        job = get_job(job_id)
        if job is None:
            raise RuntimeError(f"Background job {job_id} disappeared")
        if job["status"] in {"completed", "failed", "cancelled"}:
            if job["status"] != "completed":
                raise RuntimeError(job.get("error") or f"Benchmark job ended as {job['status']}")
            break
        time.sleep(0.15)
    return benchmark_comparison(benchmark["id"])


def _chronos_status() -> ModelStatus:
    spec = get_model_spec("chronos_bolt_tiny")
    if spec is None:
        print("  Chronos-Bolt Tiny is missing from config/models.yaml")
        return ModelStatus.FAILED
    plugin = create_model_plugin(spec)
    environment = plugin.check_environment()
    if environment.status == ModelStatus.AVAILABLE:
        print("  Downloading revision-pinned Chronos-Bolt Tiny safetensors...")
        try:
            plugin.install()
        except Exception as exc:
            print(f"  Chronos install failed: {exc}")
        environment = plugin.check_environment()
    print(f"  Chronos-Bolt Tiny: {environment.status} — {environment.message}")
    return environment.status


def _print_forecast(comparison: dict) -> None:
    for run in comparison["runs"]:
        if run["status"] != "completed":
            error = (run.get("error") or "unknown error").strip().splitlines()[-1]
            print(f"  {run['model_name']:28s} FAILED: {error}")
            continue
        metrics = run["metrics"]
        print(
            f"  {run['model_name']:28s} MAE={metrics['mae']:.2f} "
            f"WAPE={metrics['wape']:.4f} series={metrics['processed_series']}"
        )


def _print_classification(comparison: dict) -> None:
    for run in comparison["runs"]:
        if run["status"] != "completed":
            error = (run.get("error") or "unknown error").strip().splitlines()[-1]
            print(f"  {run['model_name']:28s} FAILED: {error}")
            continue
        metrics = run["metrics"]
        print(
            f"  {run['model_name']:28s} PR-AUC={metrics['pr_auc']:.4f} "
            f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CashGap Lab command line utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate a local synthetic dataset")
    generate.add_argument("--clients", type=int, default=500)
    generate.add_argument("--months", type=int, default=18)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--target-rate", type=float, default=0.10)
    generate.add_argument("--noise", type=float, default=0.15)
    demo = subparsers.add_parser("demo", help="Run reproducible forecasting and classification benchmarks")
    demo.add_argument("--clients", type=int, default=160)
    demo.add_argument("--months", type=int, default=12)
    demo.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    if arguments.command == "generate":
        config = {
            "n_clients": arguments.clients,
            "n_months": arguments.months,
            "random_seed": arguments.seed,
            "target_gap_rate": arguments.target_rate,
            "noise_level": arguments.noise,
        }
        _initialize()
        dataset_id = _generate(config)
        result = get_dataset(dataset_id)
        assert result is not None
        summary = result["summary"]
        print(
            f"Created {dataset_id}: {summary['n_clients']} clients, "
            f"{summary['n_months']} months, target={summary['target_rate']:.2%}"
        )

    if arguments.command == "demo":
        _initialize()
        dataset_id = _generate(
            {
                "n_clients": arguments.clients,
                "n_months": arguments.months,
                "random_seed": arguments.seed,
                "target_gap_rate": 0.12,
                "noise_level": 0.15,
                "overdraft_share": 0.55,
            }
        )
        print(f"Dataset {dataset_id} is ready. Checking pretrained models...")
        _chronos_status()

        print("Forecast benchmark: same dataset, target, horizon and holdout")
        forecast = _run_benchmark(
            {
                "dataset_id": dataset_id,
                "task": "cash_flow_forecasting",
                "target": "net_flow",
                "model_ids": ["seasonal_naive_local", "shell_style_catboost_lag", "chronos_bolt_tiny"],
                "series_level": "client",
                "horizon": 1,
                "min_history": 8,
                "parameters": {
                    "shell_style_catboost_lag": {"iterations": 60, "depth": 4},
                    "chronos_bolt_tiny": {"batch_size": 16},
                },
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
            }
        )
        _print_forecast(forecast)

        print("Classification benchmark: same temporal train/validation/test split")
        classification = _run_benchmark(
            {
                "dataset_id": dataset_id,
                "task": "cash_gap_classification",
                "target": "cash_gap_next_month",
                "model_ids": ["logistic_cashgap", "catboost_cashgap"],
                "series_level": "client",
                "horizon": 1,
                "min_history": 6,
                "parameters": {
                    "logistic_cashgap": {"max_iter": 250},
                    "catboost_cashgap": {"iterations": 60, "depth": 4},
                },
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
            }
        )
        _print_classification(classification)
        print("Demo completed. Open http://127.0.0.1:5173/models or /comparison after make dev.")


if __name__ == "__main__":
    main()
