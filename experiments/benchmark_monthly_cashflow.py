#!/usr/bin/env python3
"""Rolling monthly cash-flow benchmark for Python 3.8+.

The benchmark predicts two non-negative monthly totals for every INN:
credit/inflow and debit/outflow. It evaluates the last N calendar months with
an expanding training window and a separate preceding validation month.

Models:
  * trailing_mean: transparent average baseline;
  * linear_regression: scaled ordinary least squares;
  * gradient_boosting: sklearn histogram gradient boosting;
  * torch_mlp_2_layers / torch_mlp_3_layers: fully connected networks.

Example:
  python experiments/benchmark_monthly_cashflow.py \
    --outflow /data/outflow.parquet --inflow /data/inflow.parquet \
    --output-dir artifacts/monthly_benchmark --test-periods 10 --parallel
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "threadpoolctl": "threadpoolctl",
    }
    optional = {
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing_required = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    missing_optional = [package for module, package in optional.items() if importlib.util.find_spec(module) is None]
    executable = shlex.quote(sys.executable)
    print("Dependency check | Python {} | {}".format(sys.version.split()[0], sys.executable))
    if missing_required:
        print("ERROR: missing required libraries: {}".format(", ".join(missing_required)))
        print("Install: {} -m pip install {}".format(executable, " ".join(missing_required)))
        print("Jupyter: %pip install {}".format(" ".join(missing_required)))
        print("After installation restart the Jupyter kernel.")
        raise SystemExit(2)
    if missing_optional:
        print("WARNING: Torch MLP models are unavailable: {}".format(", ".join(missing_optional)))
        print("CPU install: {} -m pip install {}".format(executable, " ".join(missing_optional)))
        print("CUDA 11.8: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118".format(executable))
        print("CUDA 12.1: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121".format(executable))
        print("After installation restart the Jupyter kernel.")
    else:
        print("Dependency check: all benchmark libraries are available.")


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from experiments.train_cashflow_proxy import build_daily
except ImportError:
    from train_cashflow_proxy import build_daily


TARGET_COLUMNS = ["target_inflow", "target_outflow"]
MODEL_NAMES = (
    "trailing_mean",
    "linear_regression",
    "gradient_boosting",
    "torch_mlp_2_layers",
    "torch_mlp_3_layers",
    "torch_mlp_tuned",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outflow", required=True, help="Parquet with tr_date, dt_inn, tr_sum")
    parser.add_argument("--inflow", required=True, help="Parquet with tr_date, kt_inn, tr_sum")
    parser.add_argument("--output-dir", default="artifacts/monthly_benchmark")
    parser.add_argument("--test-periods", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--models", default=",".join(MODEL_NAMES[:-1]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--mlp2-device", default="auto")
    parser.add_argument("--mlp3-device", default="auto")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--boosting-iterations", type=int, default=250)
    parser.add_argument("--mlp-params", default=None, help="best_params.json produced by autotune_cashflow.py")
    parser.add_argument("--boosting-params", default=None, help="best_params.json produced by autotune_cashflow.py")
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    return parser.parse_args()


def build_monthly_dataset(daily: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create point-in-time monthly features; current month is target only."""
    values = daily.copy()
    global_first_date = pd.Timestamp(values["date"].min()).normalize()
    global_last_date = pd.Timestamp(values["date"].max()).normalize()
    values["month"] = values["date"].dt.to_period("M").dt.to_timestamp()
    values["active_inflow"] = values["inflow"].gt(0).astype(np.int16)
    values["active_outflow"] = values["outflow"].gt(0).astype(np.int16)
    values["negative_day"] = values["net_flow"].lt(0).astype(np.int16)
    monthly = values.groupby(["inn", "month"], as_index=False).agg(
        target_inflow=("inflow", "sum"),
        target_outflow=("outflow", "sum"),
        active_inflow_days=("active_inflow", "sum"),
        active_outflow_days=("active_outflow", "sum"),
        negative_days=("negative_day", "sum"),
    )
    monthly["target_net_flow"] = monthly["target_inflow"] - monthly["target_outflow"]
    first_month = global_first_date.to_period("M").to_timestamp()
    last_month = global_last_date.to_period("M").to_timestamp()
    if global_first_date.day != 1:
        monthly = monthly[monthly["month"].ne(first_month)]
    if global_last_date != global_last_date + pd.offsets.MonthEnd(0):
        monthly = monthly[monthly["month"].ne(last_month)]
    monthly = monthly.sort_values(["inn", "month"]).reset_index(drop=True)
    group = monthly.groupby("inn", sort=False)

    sources = [
        "target_inflow", "target_outflow", "target_net_flow",
        "active_inflow_days", "active_outflow_days", "negative_days",
    ]
    for column in sources:
        for lag in (1, 2, 3, 6, 12):
            monthly["{}_lag_{}".format(column, lag)] = group[column].shift(lag)
        for window in (3, 6, 12):
            monthly["{}_mean_{}".format(column, window)] = group[column].transform(
                lambda series, size=window: series.shift(1).rolling(size, min_periods=1).mean()
            )
            monthly["{}_std_{}".format(column, window)] = group[column].transform(
                lambda series, size=window: series.shift(1).rolling(size, min_periods=2).std()
            )

    monthly["inflow_change_1"] = monthly["target_inflow_lag_1"] - monthly["target_inflow_lag_2"]
    monthly["outflow_change_1"] = monthly["target_outflow_lag_1"] - monthly["target_outflow_lag_2"]
    monthly["inflow_ratio_to_mean_6"] = (
        monthly["target_inflow_lag_1"] / monthly["target_inflow_mean_6"].clip(lower=1.0)
    )
    monthly["outflow_ratio_to_mean_6"] = (
        monthly["target_outflow_lag_1"] / monthly["target_outflow_mean_6"].clip(lower=1.0)
    )
    month_number = monthly["month"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * month_number / 12.0)
    monthly["month_cos"] = np.cos(2 * np.pi * month_number / 12.0)
    monthly["history_months"] = group.cumcount()
    # Current-month activity is part of the target month and must never be a
    # feature. Only its lagged/rolling versions are point-in-time safe.
    excluded = {
        "inn", "month", "target_inflow", "target_outflow", "target_net_flow",
        "active_inflow_days", "active_outflow_days", "negative_days",
    }
    features = [column for column in monthly.columns if column not in excluded]
    return monthly.replace([np.inf, -np.inf], np.nan), features


def rolling_month_folds(
    frame: pd.DataFrame, test_periods: int, min_train_months: int,
) -> List[Dict[str, object]]:
    months = [pd.Timestamp(value) for value in sorted(frame["month"].unique())]
    if len(months) < min_train_months + test_periods + 1:
        raise ValueError(
            "Need at least {} distinct months for {} test periods; found {}.".format(
                min_train_months + test_periods + 1, test_periods, len(months)
            )
        )
    folds = []
    for fold_index, test_month in enumerate(months[-test_periods:], start=1):
        position = months.index(test_month)
        validation_month = months[position - 1]
        train_months = months[:position - 1]
        if len(train_months) < min_train_months:
            continue
        train = frame[frame["month"].isin(train_months)].copy()
        valid = frame[frame["month"].eq(validation_month)].copy()
        test = frame[frame["month"].eq(test_month)].copy()
        if min(len(train), len(valid), len(test)) == 0:
            continue
        folds.append({
            "fold": fold_index,
            "train_start": train_months[0],
            "train_end": train_months[-1],
            "validation_month": validation_month,
            "test_month": test_month,
            "train": train,
            "valid": valid,
            "test": test,
        })
    if len(folds) != test_periods:
        raise ValueError("Could build only {} of {} requested folds.".format(len(folds), test_periods))
    return folds


def target_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COLUMNS].to_numpy(dtype=np.float64)


def inverse_log_predictions(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(values), 0.0)


def predict_trailing_mean(test: pd.DataFrame) -> np.ndarray:
    return np.maximum(np.column_stack([
        test["target_inflow_mean_3"].fillna(test["target_inflow_lag_1"]).fillna(0.0),
        test["target_outflow_mean_3"].fillna(test["target_outflow_lag_1"]).fillna(0.0),
    ]), 0.0)


def predict_linear(
    X_train: pd.DataFrame, train: pd.DataFrame, X_test: pd.DataFrame, jobs: int,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LinearRegression(n_jobs=1))
    with threadpool_limits(limits=jobs):
        model.fit(X_train, np.log1p(target_matrix(train)))
    return inverse_log_predictions(model.predict(X_test))


def predict_boosting(
    X_train: pd.DataFrame, train: pd.DataFrame, X_test: pd.DataFrame,
    iterations: int, jobs: int, seed: int, params: Optional[Dict] = None,
) -> np.ndarray:
    options = {
        "max_iter": iterations,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1,
        "early_stopping": True,
        "random_state": seed,
    }
    if params:
        options.update(params)
    # Two targets do not justify spawning separate Python processes. The
    # underlying OpenMP math is limited explicitly to avoid CPU oversubscription.
    model = MultiOutputRegressor(HistGradientBoostingRegressor(**options), n_jobs=1)
    with threadpool_limits(limits=jobs):
        model.fit(X_train, np.log1p(target_matrix(train)))
    return inverse_log_predictions(model.predict(X_test))


def _activation(name: str):
    from torch import nn

    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]


def predict_torch_mlp(
    layers: Sequence[int], X_train: pd.DataFrame, train: pd.DataFrame,
    X_valid: pd.DataFrame, valid: pd.DataFrame, X_test: pd.DataFrame,
    device_name: str, epochs: int, batch_size: int, seed: int,
    params: Optional[Dict] = None,
) -> np.ndarray:
    import torch
    from torch import nn

    options = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "activation": "relu",
        "patience": 12,
    }
    if params:
        options.update(params)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)
    x_train_np = scaler.transform(X_train).astype(np.float32, copy=False)
    x_valid_np = scaler.transform(X_valid).astype(np.float32, copy=False)
    x_test_np = scaler.transform(X_test).astype(np.float32, copy=False)
    y_train_np = np.log1p(target_matrix(train)).astype(np.float32, copy=False)
    y_valid_np = np.log1p(target_matrix(valid)).astype(np.float32, copy=False)
    y_mean = y_train_np.mean(axis=0, keepdims=True)
    y_scale = np.maximum(y_train_np.std(axis=0, keepdims=True), 1e-3)
    y_train_np = (y_train_np - y_mean) / y_scale
    y_valid_np = (y_valid_np - y_mean) / y_scale

    # Everything numeric is copied once to VRAM. No batch performs a CPU->GPU copy.
    x_train = torch.as_tensor(x_train_np, device=device)
    x_valid = torch.as_tensor(x_valid_np, device=device)
    x_test = torch.as_tensor(x_test_np, device=device)
    y_train = torch.as_tensor(y_train_np, device=device)
    y_valid = torch.as_tensor(y_valid_np, device=device)
    activation_class = _activation(str(options["activation"]))
    modules = []
    width = X_train.shape[1]
    for hidden in layers:
        modules.extend([nn.Linear(width, int(hidden)), activation_class(), nn.Dropout(float(options["dropout"]))])
        width = int(hidden)
    modules.append(nn.Linear(width, 2))
    model = nn.Sequential(*modules).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    loss_function = nn.SmoothL1Loss()
    best_loss, best_state = float("inf"), None
    patience = int(options["patience"])
    remaining = patience
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator, device=device)
        for start in range(0, len(x_train), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_valid), y_valid))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(x_test).cpu().numpy()
    prediction_log = prediction_scaled * y_scale + y_mean
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print("    {}: epoch={}, {:.2f}s, VRAM {:.2f} GiB".format(
            device, epoch, time.perf_counter() - started, allocated
        ))
    return inverse_log_predictions(prediction_log)


def resolve_devices(mlp2: str, mlp3: str, parallel: bool) -> Tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "cpu", "cpu"
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def resolve(requested: str, preferred: int) -> str:
        if requested == "auto":
            if count == 0:
                return "cpu"
            if parallel and preferred < count:
                return "cuda:{}".format(preferred)
            return "cuda:0"
        device = torch.device(requested)
        if device.type == "cuda":
            index = 0 if device.index is None else int(device.index)
            if index >= count:
                raise ValueError("Requested {}, but PyTorch sees {} GPU(s).".format(requested, count))
            return "cuda:{}".format(index)
        return "cpu"

    result = resolve(mlp2, 0), resolve(mlp3, 1)
    print("PyTorch GPU count: {} | MLP-2: {} | MLP-3: {}".format(count, result[0], result[1]))
    return result


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, zero_floor: float,
) -> List[Dict[str, float]]:
    rows = []
    for index, flow in enumerate(("inflow_credit", "outflow_debit")):
        truth = actual[:, index].astype(float)
        forecast = predicted[:, index].astype(float)
        nonzero = np.abs(truth) > zero_floor
        total_truth = float(truth.sum())
        total_forecast = float(forecast.sum())
        aggregate_mape = abs(total_forecast - total_truth) / max(abs(total_truth), zero_floor)
        wape = np.abs(forecast - truth).sum() / max(np.abs(truth).sum(), zero_floor)
        company_mape = (
            float(np.mean(np.abs(forecast[nonzero] - truth[nonzero]) / np.abs(truth[nonzero])))
            if nonzero.any() else float("nan")
        )
        rows.append({
            "flow": flow,
            "aggregate_mape": float(aggregate_mape),
            "aggregate_mape_percent": float(aggregate_mape * 100),
            "company_mape_nonzero": company_mape,
            "company_mape_nonzero_percent": float(company_mape * 100),
            "wape": float(wape),
            "wape_percent": float(wape * 100),
            "mae": float(np.mean(np.abs(forecast - truth))),
            "bias_percent": float((total_forecast - total_truth) / max(abs(total_truth), zero_floor) * 100),
            "actual_total": total_truth,
            "predicted_total": total_forecast,
            "nonzero_companies": int(nonzero.sum()),
            "companies": int(len(truth)),
        })
    return rows


def run_model(
    name: str, fold: Dict[str, object], features: List[str], args: argparse.Namespace,
    devices: Tuple[str, str], jobs: int,
) -> Tuple[np.ndarray, float]:
    train = fold["train"]
    valid = fold["valid"]
    test = fold["test"]
    X_train = train[features].fillna(0.0)
    X_valid = valid[features].fillna(0.0)
    X_test = test[features].fillna(0.0)
    started = time.perf_counter()
    if name == "trailing_mean":
        prediction = predict_trailing_mean(test)
    elif name == "linear_regression":
        prediction = predict_linear(X_train, train, X_test, jobs)
    elif name == "gradient_boosting":
        tuned = getattr(args, "boosting_tuned_params", None)
        prediction = predict_boosting(
            X_train, train, X_test, args.boosting_iterations, jobs, args.seed, tuned
        )
    elif name == "torch_mlp_2_layers":
        prediction = predict_torch_mlp(
            (128, 64), X_train, train, X_valid, valid, X_test,
            devices[0], args.epochs, args.batch_size, args.seed,
        )
    elif name == "torch_mlp_3_layers":
        prediction = predict_torch_mlp(
            (256, 128, 64), X_train, train, X_valid, valid, X_test,
            devices[1], args.epochs, args.batch_size, args.seed,
        )
    elif name == "torch_mlp_tuned":
        tuned = getattr(args, "mlp_tuned_params", None)
        if not tuned:
            raise ValueError("torch_mlp_tuned requires --mlp-params from autotune_cashflow.py")
        layers = tuple(int(value) for value in tuned.get("layers", [256, 128, 64]))
        tuned_batch = int(tuned.get("batch_size", args.batch_size))
        prediction = predict_torch_mlp(
            layers, X_train, train, X_valid, valid, X_test,
            devices[0], args.epochs, tuned_batch, args.seed, tuned,
        )
    else:
        raise ValueError("Unknown model: {}".format(name))
    return prediction, time.perf_counter() - started


def validate_models(value: str) -> List[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError("Unknown models: {}. Available: {}".format(unknown, MODEL_NAMES))
    return models


def load_tuned_params(path: Optional[str], expected_model: str) -> Optional[Dict]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model") != expected_model:
        raise ValueError("{} contains model={!r}, expected {!r}".format(
            path, payload.get("model"), expected_model
        ))
    return dict(payload["best_params"])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    models = validate_models(args.models)
    if any(name.startswith("torch_") for name in models) and importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "Selected Torch MLP models, but torch is missing. Install the CUDA command printed above, restart the kernel, and retry."
        )
    args.mlp_tuned_params = load_tuned_params(args.mlp_params, "mlp")
    args.boosting_tuned_params = load_tuned_params(args.boosting_params, "gradient_boosting")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("\n=== MONTHLY CASH-FLOW BENCHMARK ===")
    daily = build_daily(args)
    monthly, features = build_monthly_dataset(daily)
    folds = rolling_month_folds(monthly, args.test_periods, args.min_train_months)
    devices = resolve_devices(args.mlp2_device, args.mlp3_device, args.parallel)
    total_cpus = max(1, os.cpu_count() or 1)
    jobs = args.cpu_threads or (max(1, total_cpus // 2) if args.parallel else total_cpus)
    print("INNs: {:,} | months: {} | features: {} | folds: {}".format(
        monthly["inn"].nunique(), monthly["month"].nunique(), len(features), len(folds)
    ))

    metric_rows, prediction_rows, window_rows = [], [], []
    for fold in folds:
        fold_number = int(fold["fold"])
        test_month = pd.Timestamp(fold["test_month"])
        print("\n--- Fold {}/{} | test {} ---".format(fold_number, len(folds), test_month.strftime("%Y-%m")))
        window_rows.append({
            "fold": fold_number,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "validation_month": fold["validation_month"],
            "test_month": fold["test_month"],
            "train_rows": len(fold["train"]),
            "validation_rows": len(fold["valid"]),
            "test_rows": len(fold["test"]),
        })

        results = {}
        if args.parallel and len(models) > 1:
            # DataFrames are read-only and shared in RAM; GPU models receive one
            # device each while CPU alternatives run alongside them.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
                future_names = {
                    executor.submit(run_model, name, fold, features, args, devices, jobs): name
                    for name in models
                }
                for future in concurrent.futures.as_completed(future_names):
                    results[future_names[future]] = future.result()
        else:
            for name in models:
                results[name] = run_model(name, fold, features, args, devices, jobs)

        actual = target_matrix(fold["test"])
        for name in models:
            prediction, seconds = results[name]
            for metric in regression_metrics(actual, prediction, args.mape_zero_floor):
                metric_rows.append({
                    "fold": fold_number,
                    "test_month": test_month,
                    "model": name,
                    "training_seconds": round(seconds, 3),
                    **metric,
                })
            prediction_rows.append(pd.DataFrame({
                "fold": fold_number,
                "test_month": test_month,
                "inn": fold["test"]["inn"].astype(str).to_numpy(),
                "model": name,
                "actual_inflow": actual[:, 0],
                "predicted_inflow": prediction[:, 0],
                "actual_outflow": actual[:, 1],
                "predicted_outflow": prediction[:, 1],
            }))
            primary = [row for row in metric_rows if row["fold"] == fold_number and row["model"] == name]
            print("  {:24s} | credit MAPE {:6.2f}% | debit MAPE {:6.2f}% | {:.1f}s".format(
                name, primary[0]["aggregate_mape_percent"], primary[1]["aggregate_mape_percent"], seconds
            ))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    windows = pd.DataFrame(window_rows)
    summary = metrics.groupby(["model", "flow"], as_index=False).agg(
        aggregate_mape_mean_percent=("aggregate_mape_percent", "mean"),
        aggregate_mape_std_percent=("aggregate_mape_percent", "std"),
        aggregate_mape_worst_percent=("aggregate_mape_percent", "max"),
        wape_mean_percent=("wape_percent", "mean"),
        company_mape_mean_percent=("company_mape_nonzero_percent", "mean"),
        bias_mean_percent=("bias_percent", "mean"),
        folds=("fold", "nunique"),
    ).sort_values(["flow", "aggregate_mape_mean_percent"])
    metrics.to_csv(output / "monthly_fold_metrics.csv", index=False)
    summary.to_csv(output / "monthly_stability_summary.csv", index=False)
    windows.to_csv(output / "monthly_fold_windows.csv", index=False)
    predictions.to_parquet(output / "monthly_predictions.parquet", index=False)
    config = vars(args).copy()
    config["features"] = features
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== STABILITY OVER {} TEST MONTHS ===".format(len(folds)))
    print(summary.to_string(index=False))
    print("\nSaved to {}".format(output.resolve()))


if __name__ == "__main__":
    main()
