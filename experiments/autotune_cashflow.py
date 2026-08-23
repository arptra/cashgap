#!/usr/bin/env python3
"""Optuna hyperparameter tuning without touching the final 10 test months.

Python 3.8 installation:
  python -m pip install "optuna==3.6.2"

Tune an MLP (two trials may use two GPUs):
  python experiments/autotune_cashflow.py --model mlp --trials 30 --jobs 2 \
    --devices cuda:0,cuda:1 --outflow /data/out.parquet --inflow /data/in.parquet

Tune gradient boosting:
  python experiments/autotune_cashflow.py --model gradient_boosting --trials 40 \
    --jobs 2 --outflow /data/out.parquet --inflow /data/in.parquet

The objective is the mean aggregate monthly MAPE for credit and debit over
several rolling validation months. The final holdout months are excluded.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import shlex
import sys
from pathlib import Path
from typing import Dict, List


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "threadpoolctl": "threadpoolctl",
        "optuna": "optuna==3.6.2" if sys.version_info[:2] == (3, 8) else "optuna",
    }
    optional = {
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing_required = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    missing_optional = [package for module, package in optional.items() if importlib.util.find_spec(module) is None]
    executable = shlex.quote(sys.executable)
    print("Dependency check | Python {} | {}".format(sys.version.split()[0], sys.executable))
    if missing_required:
        print("ERROR: missing autotuning libraries: {}".format(", ".join(missing_required)))
        print("Install: {} -m pip install {}".format(executable, " ".join(missing_required)))
        print("Jupyter: %pip install {}".format(" ".join(missing_required)))
        print("After installation restart the Jupyter kernel.")
        raise SystemExit(2)
    if missing_optional:
        print("WARNING: MLP tuning requires: {}".format(", ".join(missing_optional)))
        print("CUDA 11.8: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118".format(executable))
        print("CUDA 12.1: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121".format(executable))
        print("After installation restart the Jupyter kernel.")
    else:
        print("Dependency check: all autotuning libraries are available.")


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import pandas as pd

try:
    from experiments.train_cashflow_proxy import build_observed_daily
    from experiments.benchmark_monthly_cashflow import (
        build_monthly_dataset,
        materialize_fold,
        predict_boosting,
        predict_torch_mlp,
        regression_metrics,
        rolling_month_folds,
        target_matrix,
    )
except ImportError:
    from train_cashflow_proxy import build_observed_daily
    from benchmark_monthly_cashflow import (
        build_monthly_dataset,
        materialize_fold,
        predict_boosting,
        predict_torch_mlp,
        regression_metrics,
        rolling_month_folds,
        target_matrix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("mlp", "gradient_boosting"))
    parser.add_argument("--outflow", required=True)
    parser.add_argument("--inflow", required=True)
    parser.add_argument("--output-dir", default="artifacts/cashflow_tuning")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent Optuna trials")
    parser.add_argument("--devices", default="auto", help="Comma-separated: cuda:0,cuda:1 or cpu")
    parser.add_argument("--tuning-periods", type=int, default=3)
    parser.add_argument("--holdout-test-periods", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    return parser.parse_args()


def resolve_devices(value: str) -> List[str]:
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if value == "auto":
        return ["cuda:{}".format(index) for index in range(count)] or ["cpu"]
    devices = [item.strip() for item in value.split(",") if item.strip()]
    for item in devices:
        device = torch.device(item)
        if device.type == "cuda":
            index = 0 if device.index is None else int(device.index)
            if index >= count:
                raise ValueError("Requested {}, but PyTorch sees {} GPU(s).".format(item, count))
    return devices


def tuning_frame_without_holdout(
    monthly: pd.DataFrame, holdout_periods: int,
) -> pd.DataFrame:
    months = [pd.Timestamp(value) for value in sorted(monthly["month"].unique())]
    if holdout_periods < 1 or len(months) <= holdout_periods:
        raise ValueError("Not enough months to reserve {} final holdouts.".format(holdout_periods))
    holdout_start = months[-holdout_periods]
    print("Final holdout is protected from {} through {}.".format(
        holdout_start.strftime("%Y-%m"), months[-1].strftime("%Y-%m")
    ))
    return monthly[monthly["month"] < holdout_start].copy()


def score_prediction(actual: np.ndarray, predicted: np.ndarray, zero_floor: float) -> float:
    metrics = regression_metrics(actual, predicted, zero_floor)
    return float(np.mean([row["aggregate_mape"] for row in metrics]))


def mlp_layers(trial) -> List[int]:
    count = trial.suggest_int("n_layers", 1, 6)
    first = trial.suggest_categorical("first_width", [64, 128, 256, 512, 1024, 2048])
    shrink = trial.suggest_categorical("width_shrink", [0.5, 0.75, 1.0])
    return [max(32, int(first * (shrink ** index))) for index in range(count)]


def main() -> None:
    args = parse_args()
    if args.model == "mlp" and importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "MLP tuning requires torch. Install the CUDA command printed above, restart the kernel, and retry."
        )
    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            'Optuna is required. Python 3.8 command: python -m pip install "optuna==3.6.2"'
        ) from error

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_root = Path(args.output_dir)
    output = output_root if output_root.name == args.model else output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    observed_daily = build_observed_daily(args)
    monthly, features = build_monthly_dataset(observed_daily)
    del observed_daily
    tuning_frame = tuning_frame_without_holdout(monthly, args.holdout_test_periods)
    fold_specs = rolling_month_folds(tuning_frame, args.tuning_periods, args.min_train_months)
    folds = [materialize_fold(tuning_frame, fold, features) for fold in fold_specs]
    del monthly, tuning_frame
    gc.collect()
    devices = resolve_devices(args.devices) if args.model == "mlp" else ["cpu"]
    total_cpus = max(1, os.cpu_count() or 1)
    per_trial_jobs = args.cpu_threads or max(1, total_cpus // max(1, args.jobs))
    print("Tuning model={} | trials={} | parallel jobs={} | devices={}".format(
        args.model, args.trials, args.jobs, devices
    ))
    print("Tuning months: {}".format(
        ", ".join(pd.Timestamp(fold["test_month"]).strftime("%Y-%m") for fold in folds)
    ))

    def objective(trial):
        if args.model == "mlp":
            layers = mlp_layers(trial)
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
                "patience": trial.suggest_int("patience", 8, 18),
            }
            batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384, 32768, 65536])
            device = devices[trial.number % len(devices)]
            trial.set_user_attr("layers", layers)
        else:
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63, 127]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, step=10),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
                "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
            }
            layers, batch_size, device = [], 0, "cpu"

        fold_scores = []
        for step, fold in enumerate(folds):
            train = fold["train"]
            valid = fold["valid"]
            test = fold["test"]
            X_train = fold["X_train"]
            X_valid = fold["X_valid"]
            X_test = fold["X_test"]
            try:
                if args.model == "mlp":
                    prediction = predict_torch_mlp(
                        layers, X_train, train, X_valid, valid, X_test,
                        device, args.epochs, batch_size, args.seed + trial.number, params,
                    )
                else:
                    prediction = predict_boosting(
                        X_train, train, X_test, int(params["max_iter"]),
                        per_trial_jobs, args.seed + trial.number, params,
                    )
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    if args.model == "mlp":
                        import torch
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned("Out of memory")
                raise
            score = score_prediction(target_matrix(test), prediction, args.mape_zero_floor)
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    storage = "sqlite:///{}".format((output / "study.sqlite3").resolve())
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.jobs * 2), n_warmup_steps=1)
    study = optuna.create_study(
        study_name="cashflow_{}".format(args.model), storage=storage,
        direction="minimize", sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, gc_after_trial=True)
    best_params = dict(study.best_trial.params)
    if args.model == "mlp":
        best_params["layers"] = list(study.best_trial.user_attrs["layers"])
    payload = {
        "model": args.model,
        "objective": "mean aggregate monthly MAPE for credit and debit",
        "best_value": float(study.best_value),
        "best_value_percent": float(study.best_value * 100),
        "best_params": best_params,
        "tuning_periods": [pd.Timestamp(fold["test_month"]).strftime("%Y-%m") for fold in folds],
        "protected_holdout_periods": args.holdout_test_periods,
        "trials_total": len(study.trials),
    }
    (output / "best_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    print("\nBest objective: {:.3f}%".format(study.best_value * 100))
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    print("Saved: {}".format((output / "best_params.json").resolve()))


if __name__ == "__main__":
    main()
