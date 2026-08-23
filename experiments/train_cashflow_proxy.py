#!/usr/bin/env python3
"""Daily cash-flow forecast and a business-friendly historical demo.

For every INN and forecast date the models predict each of the next N days:
  * probability that net cash flow will be negative;
  * expected inflow, outflow and net flow.

This is a liquidity-stress proxy, not a true cash-gap forecast: balances are
not present in the source data.

Train and immediately show a random demo:
  python experiments/train_cashflow_proxy.py train-demo \
    --outflow /data/outflow.parquet --inflow /data/inflow.parquet

Show another random historical example without retraining:
  python experiments/train_cashflow_proxy.py demo \
    --output-dir artifacts/cashflow_daily
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import secrets
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
    }
    optional = {
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
        "xgboost": "xgboost==2.1.4" if sys.version_info[:2] == (3, 8) else "xgboost",
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
        print("WARNING: unavailable optional models: {}".format(", ".join(missing_optional)))
        print("Install: {} -m pip install {}".format(executable, " ".join(missing_optional)))
        if any(package.startswith("torch") for package in missing_optional):
            print("CUDA 11.8: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118".format(executable))
            print("CUDA 12.1: {} -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121".format(executable))
        print("After installation restart the Jupyter kernel.")
    else:
        print("Dependency check: required and optional libraries are available.")


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FLOW_COLUMNS = ("inflow", "outflow", "net_flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("train", "demo", "train-demo"), default="train-demo")
    parser.add_argument("--outflow", help="Parquet: tr_date, dt_inn, tr_sum (required for training)")
    parser.add_argument("--inflow", help="Parquet: tr_date, kt_inn, tr_sum (required for training)")
    parser.add_argument("--output-dir", default="artifacts/cashflow_daily")
    parser.add_argument("--horizon", type=int, default=14, help="Number of future calendar days")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible training seed")
    parser.add_argument("--demo-seed", type=int, default=None, help="Optional reproducible demo selection")
    parser.add_argument("--demo-model", default="best", help="Model name or 'best'")
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--max-inns", type=int, default=None, help="Quick-run limit to most active INNs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--xgb-estimators", type=int, default=250)
    parser.add_argument("--parallel", action="store_true", help="Train XGBoost and both MLPs concurrently")
    parser.add_argument("--mlp2-device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--mlp3-device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--xgb-threads", type=int, default=None, help="CPU threads for XGBoost")
    parser.add_argument("--torch-cpu-threads", type=int, default=None, help="CPU threads shared by PyTorch")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def parse_transaction_dates(values: pd.Series) -> pd.Series:
    """Parse strings, YYYYMMDD integers, Unix timestamps or epoch-day integers."""
    if not pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, errors="coerce").dt.normalize()

    numeric = pd.to_numeric(values, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return pd.to_datetime(values, errors="coerce").dt.normalize()

    typical = float(non_null.abs().median())
    if 10_000_000 <= typical <= 99_999_999:
        # Common integer representation: 20240813 means 13 August 2024.
        return pd.to_datetime(numeric.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce").dt.normalize()
    if 1_000_000_000 <= typical < 100_000_000_000:
        return pd.to_datetime(numeric, unit="s", origin="unix", errors="coerce").dt.normalize()
    if 100_000_000_000 <= typical < 100_000_000_000_000:
        return pd.to_datetime(numeric, unit="ms", origin="unix", errors="coerce").dt.normalize()
    if 10_000 <= typical < 1_000_000:
        # Arrow/Parquet dates can be stored as a number of days since 1970-01-01.
        return pd.to_datetime(numeric, unit="D", origin="unix", errors="coerce").dt.normalize()
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _column_key(name: str) -> str:
    """Compare schema names without case, underscores, spaces or punctuation."""
    return "".join(character for character in str(name).casefold() if character.isalnum())


def resolve_parquet_columns(path: str, expected: Sequence[str]) -> Dict[str, str]:
    """Resolve common schema variants such as dt_inn/dtinn and kt_inn/ktinn."""
    import pyarrow.dataset as arrow_dataset

    available = list(arrow_dataset.dataset(path, format="parquet").schema.names)
    normalized: Dict[str, List[str]] = {}
    for actual in available:
        normalized.setdefault(_column_key(actual), []).append(actual)

    resolved: Dict[str, str] = {}
    for requested in expected:
        if requested in available:
            resolved[requested] = requested
            continue
        matches = normalized.get(_column_key(requested), [])
        if len(matches) == 1:
            resolved[requested] = matches[0]
            continue
        if len(matches) > 1:
            raise ValueError(
                "Ambiguous parquet column {!r}: matches {}. Available columns: {}".format(
                    requested, matches, available
                )
            )
        raise ValueError(
            "Required parquet column {!r} was not found in {}. Available columns: {}".format(
                requested, path, available
            )
        )
    return resolved


def normalize(path: str, inn_column: str, value_name: str) -> pd.DataFrame:
    columns = resolve_parquet_columns(path, ["tr_date", inn_column, "tr_sum"])
    selected = [columns["tr_date"], columns[inn_column], columns["tr_sum"]]
    aliases = {
        columns["tr_date"]: "date",
        columns[inn_column]: "inn",
        columns["tr_sum"]: value_name,
    }
    changed = ["{} -> {}".format(actual, expected) for expected, actual in columns.items() if actual != expected]
    if changed:
        print("Parquet column aliases | {} | {}".format(path, ", ".join(changed)))
    frame = pd.read_parquet(path, columns=selected).rename(columns=aliases)
    frame["date"] = parse_transaction_dates(frame["date"])
    frame["inn"] = frame["inn"].astype("string").str.strip()
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["date", "inn"])
    frame = frame[frame["inn"].ne("")]
    return frame.groupby(["inn", "date"], as_index=False)[value_name].sum()


def build_observed_daily(args: argparse.Namespace) -> pd.DataFrame:
    """Load one row per INN/transaction day without filling inactive dates."""
    if not args.outflow or not args.inflow:
        raise ValueError("Training requires both --outflow and --inflow parquet paths.")
    outflow = normalize(args.outflow, "dt_inn", "outflow")
    inflow = normalize(args.inflow, "kt_inn", "inflow")
    daily = outflow.merge(inflow, on=["inn", "date"], how="outer").fillna(0.0)
    if args.max_inns:
        selected = daily["inn"].value_counts().head(args.max_inns).index
        daily = daily[daily["inn"].isin(selected)]
    if daily.empty:
        raise ValueError("No valid transactions were found in the parquet files.")
    daily["net_flow"] = daily["inflow"] - daily["outflow"]
    return daily.sort_values(["inn", "date"]).reset_index(drop=True)


def build_daily(args: argparse.Namespace) -> pd.DataFrame:
    daily = build_observed_daily(args)

    parts = []
    for inn, group in daily.groupby("inn", sort=False):
        dates = pd.date_range(group.date.min(), group.date.max(), freq="D")
        part = group.set_index("date").reindex(dates).rename_axis("date").reset_index()
        part["inn"] = inn
        parts.append(part)
    if not parts:
        raise ValueError("No valid transactions were found in the parquet files.")
    result = pd.concat(parts, ignore_index=True)
    result[["inflow", "outflow"]] = result[["inflow", "outflow"]].fillna(0.0)
    result["net_flow"] = result["inflow"] - result["outflow"]
    return result.sort_values(["inn", "date"]).reset_index(drop=True)


def build_dataset(daily: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    frame = daily.copy()
    group = frame.groupby("inn", sort=False)
    for days in (1, 7, 14, 30, 60):
        for column in FLOW_COLUMNS:
            frame["{}_{}d".format(column, days)] = group[column].transform(
                lambda values, window=days: values.rolling(window, min_periods=1).sum()
            )
    for column in FLOW_COLUMNS:
        frame["{}_mean_30d".format(column)] = group[column].transform(
            lambda values: values.rolling(30, min_periods=2).mean()
        )
        frame["{}_std_30d".format(column)] = group[column].transform(
            lambda values: values.rolling(30, min_periods=2).std()
        )
        for lag in (1, 7, 14, 28):
            frame["{}_lag_{}".format(column, lag)] = group[column].shift(lag)
    frame["inflow_change_7d"] = frame["inflow"] - frame["inflow_lag_7"]
    frame["outflow_change_7d"] = frame["outflow"] - frame["outflow_lag_7"]
    frame["day_of_week"] = frame.date.dt.dayofweek
    frame["day_of_month"] = frame.date.dt.day
    frame["month"] = frame.date.dt.month

    target_columns = []
    for day in range(1, horizon + 1):
        for column in ("inflow", "outflow"):
            target = "actual_{}_d{}".format(column, day)
            frame[target] = group[column].shift(-day)
            target_columns.append(target)
        frame["actual_net_flow_d{}".format(day)] = (
            frame["actual_inflow_d{}".format(day)] - frame["actual_outflow_d{}".format(day)]
        )
        frame["target_negative_d{}".format(day)] = (
            frame["actual_net_flow_d{}".format(day)] < 0
        ).astype(float)
        target_columns.extend(["actual_net_flow_d{}".format(day), "target_negative_d{}".format(day)])

    # Keep only origins for which all future days are known. This makes the
    # historical demo fair and prevents future leakage.
    frame = frame.dropna(subset=target_columns).copy()
    features = [column for column in frame.columns if column not in {"inn", "date"} and column not in target_columns]
    return frame.replace([np.inf, -np.inf], np.nan), features


def time_split(frame: pd.DataFrame, validation_days: int, test_days: int):
    dates = np.sort(frame.date.unique())
    minimum = validation_days + test_days + 60
    if len(dates) <= minimum:
        raise ValueError("Need more than {} distinct dates; found {}.".format(minimum, len(dates)))
    test_start = dates[-test_days]
    validation_start = dates[-(test_days + validation_days)]
    train = frame[frame.date < validation_start]
    valid = frame[(frame.date >= validation_start) & (frame.date < test_start)]
    test = frame[frame.date >= test_start]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Time split produced an empty train, validation or test set.")
    return train, valid, test


def target_arrays(frame: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    negative = np.column_stack([
        frame["target_negative_d{}".format(day)].to_numpy(dtype=np.float32)
        for day in range(1, horizon + 1)
    ])
    flows = np.column_stack([
        frame["actual_{}_d{}".format(kind, day)].to_numpy(dtype=np.float32)
        for day in range(1, horizon + 1)
        for kind in ("inflow", "outflow")
    ])
    return negative, flows


def safe_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    labels = (probabilities >= threshold).astype(int)
    unique = np.unique(y)
    return {
        "pr_auc": float(average_precision_score(y, probabilities)) if len(unique) == 2 else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(unique) == 2 else None,
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "positive_rate": float(y.mean()),
    }


def best_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    best, selected = -1.0, 0.5
    for threshold in candidates:
        precision = precision_score(y, probabilities >= threshold, zero_division=0)
        recall = recall_score(y, probabilities >= threshold, zero_division=0)
        score = 2 * precision * recall / max(precision + recall, 1e-9)
        if score > best:
            best, selected = score, float(threshold)
    return selected


def train_torch_multitask(
    layers: Sequence[int], X_train: pd.DataFrame, train: pd.DataFrame,
    X_valid: pd.DataFrame, valid: pd.DataFrame, X_test: pd.DataFrame,
    horizon: int, seed: int, epochs: int, batch_size: int, device_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One multi-task MLP predicts H risks and H pairs of monetary flows."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device(device_name)
    print("  PyTorch device: {}".format(device))

    scaler = StandardScaler().fit(X_train)
    train_negative, train_flows = target_arrays(train, horizon)
    valid_negative, valid_flows = target_arrays(valid, horizon)
    flow_scale = np.maximum(np.nanpercentile(np.log1p(train_flows), 90, axis=0), 1.0).astype(np.float32)

    # The prepared numeric dataset is small compared with modern GPU memory.
    # Copy it once and form batches by GPU indices instead of transferring every
    # batch from CPU. Raw parquet parsing and feature engineering remain on CPU.
    x_train = torch.as_tensor(
        scaler.transform(X_train).astype(np.float32, copy=False), device=device
    )
    y_negative = torch.as_tensor(train_negative, dtype=torch.float32, device=device)
    y_flows = torch.as_tensor(
        (np.log1p(train_flows) / flow_scale).astype(np.float32, copy=False), device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print("  Dataset preloaded to {}: {:.2f} GiB".format(device, allocated_gb))

    modules = []
    width = X_train.shape[1]
    for hidden in layers:
        modules.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.15)])
        width = hidden
    trunk = nn.Sequential(*modules)

    class MultiTaskMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.risk = nn.Linear(width, horizon)
            self.flow = nn.Linear(width, horizon * 2)

        def forward(self, values):
            hidden = self.trunk(values)
            return self.risk(hidden), self.flow(hidden)

    model = MultiTaskMLP().to(device)
    positive = train_negative.sum(axis=0)
    positive_weight = (len(train_negative) - positive) / np.maximum(positive, 1)
    classification_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    regression_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_valid = torch.tensor(scaler.transform(X_valid), dtype=torch.float32, device=device)
    valid_negative_t = torch.tensor(valid_negative, dtype=torch.float32, device=device)
    valid_flows_t = torch.tensor(np.log1p(valid_flows) / flow_scale, dtype=torch.float32, device=device)
    best_loss, best_state, patience = float("inf"), None, 12
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)
        permutation = torch.randperm(len(x_train), generator=generator, device=device)
        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            batch_x = x_train[indices]
            batch_negative = y_negative[indices]
            batch_flows = y_flows[indices]
            optimizer.zero_grad()
            logits, flow_prediction = model(batch_x)
            loss = classification_loss(logits, batch_negative) + 0.35 * regression_loss(flow_prediction, batch_flows)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_logits, valid_flow_prediction = model(x_valid)
            validation_loss = float(
                classification_loss(valid_logits, valid_negative_t)
                + 0.35 * regression_loss(valid_flow_prediction, valid_flows_t)
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 12
        else:
            patience -= 1
            if patience == 0:
                break
        if epoch == 1 or epoch % 10 == 0:
            print("  epoch {:3d} | validation loss {:.4f}".format(epoch, validation_loss))

    model.load_state_dict(best_state)

    def predict(values: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probabilities, all_flows = [], []
        transformed = torch.as_tensor(
            scaler.transform(values).astype(np.float32, copy=False), device=device
        )
        with torch.no_grad():
            for start in range(0, len(transformed), batch_size):
                batch = transformed[start:start + batch_size]
                logits, flow_prediction = model(batch)
                all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
                all_flows.append(flow_prediction.cpu().numpy())
        probabilities = np.concatenate(all_probabilities)
        scaled_flows = np.concatenate(all_flows)
        flows = np.maximum(np.expm1(scaled_flows * flow_scale), 0.0)
        return probabilities, flows

    valid_probability, valid_flow = predict(X_valid)
    test_probability, test_flow = predict(X_test)
    return valid_probability, valid_flow, test_probability, test_flow


def train_xgboost_daily(
    X_train: pd.DataFrame, train: pd.DataFrame, X_valid: pd.DataFrame, valid: pd.DataFrame,
    X_test: pd.DataFrame, horizon: int, seed: int, estimators: int, jobs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier, XGBRegressor

    valid_probability = np.zeros((len(valid), horizon), dtype=np.float32)
    test_probability = np.zeros((len(X_test), horizon), dtype=np.float32)
    valid_flows = np.zeros((len(valid), horizon * 2), dtype=np.float32)
    test_flows = np.zeros((len(X_test), horizon * 2), dtype=np.float32)
    for index, day in enumerate(range(1, horizon + 1)):
        print("  XGBoost horizon day {}/{}".format(day, horizon))
        y_train = train["target_negative_d{}".format(day)].astype(int)
        classifier = XGBClassifier(
            n_estimators=estimators, max_depth=7, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=jobs, tree_method="hist", eval_metric="logloss",
            random_state=seed,
        )
        classifier.fit(X_train, y_train)
        valid_probability[:, index] = classifier.predict_proba(X_valid)[:, 1]
        test_probability[:, index] = classifier.predict_proba(X_test)[:, 1]
        for offset, kind in enumerate(("inflow", "outflow")):
            target = np.log1p(train["actual_{}_d{}".format(kind, day)].to_numpy())
            regressor = XGBRegressor(
                n_estimators=max(100, estimators // 2), max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=jobs, tree_method="hist",
                objective="reg:squarederror", random_state=seed,
            )
            regressor.fit(X_train, target)
            column = index * 2 + offset
            valid_flows[:, column] = np.maximum(np.expm1(regressor.predict(X_valid)), 0.0)
            test_flows[:, column] = np.maximum(np.expm1(regressor.predict(X_test)), 0.0)
    return valid_probability, valid_flows, test_probability, test_flows


def collect_results(
    name: str, valid: pd.DataFrame, test: pd.DataFrame, valid_probability: np.ndarray,
    test_probability: np.ndarray, test_flows: np.ndarray, horizon: int, seconds: float,
) -> Tuple[List[Dict], pd.DataFrame]:
    metric_rows, prediction_parts, thresholds = [], [], []
    for index, day in enumerate(range(1, horizon + 1)):
        valid_y = valid["target_negative_d{}".format(day)].to_numpy(dtype=int)
        test_y = test["target_negative_d{}".format(day)].to_numpy(dtype=int)
        threshold = best_threshold(valid_y, valid_probability[:, index])
        thresholds.append(threshold)
        row = {"model": name, "horizon_day": day, "threshold": threshold, **safe_metrics(test_y, test_probability[:, index], threshold)}
        metric_rows.append(row)
        predicted_inflow = test_flows[:, index * 2]
        predicted_outflow = test_flows[:, index * 2 + 1]
        prediction_parts.append(pd.DataFrame({
            "origin_date": test.date.to_numpy(),
            "forecast_date": test.date.to_numpy() + pd.to_timedelta(day, unit="D"),
            "horizon_day": day,
            "inn": test.inn.astype(str).to_numpy(),
            "model": name,
            "probability_negative": test_probability[:, index],
            "threshold": threshold,
            "predicted_inflow": predicted_inflow,
            "predicted_outflow": predicted_outflow,
            "predicted_net_flow": predicted_inflow - predicted_outflow,
            "actual_inflow": test["actual_inflow_d{}".format(day)].to_numpy(),
            "actual_outflow": test["actual_outflow_d{}".format(day)].to_numpy(),
            "actual_net_flow": test["actual_net_flow_d{}".format(day)].to_numpy(),
            "actual_negative": test_y,
        }))
    aggregate_y = np.column_stack([
        test["target_negative_d{}".format(day)].to_numpy(dtype=int)
        for day in range(1, horizon + 1)
    ]).reshape(-1)
    aggregate_probability = test_probability.reshape(-1)
    aggregate = safe_metrics(aggregate_y, aggregate_probability, 0.5)
    aggregate_labels = (test_probability >= np.asarray(thresholds)[None, :]).astype(int).reshape(-1)
    aggregate["precision"] = float(precision_score(aggregate_y, aggregate_labels, zero_division=0))
    aggregate["recall"] = float(recall_score(aggregate_y, aggregate_labels, zero_division=0))
    metric_rows.append({"model": name, "horizon_day": 0, "threshold": float(np.mean(thresholds)),
                        "training_seconds": round(seconds, 2), **aggregate})
    return metric_rows, pd.concat(prediction_parts, ignore_index=True)


def resolve_torch_devices(args: argparse.Namespace) -> Tuple[str, str]:
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def resolve(requested: str, preferred_index: int) -> str:
        if requested == "auto":
            if gpu_count == 0:
                return "cpu"
            if args.parallel and preferred_index < gpu_count:
                return "cuda:{}".format(preferred_index)
            return "cuda:0"
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError("Invalid PyTorch device {!r}: {}".format(requested, error)) from error
        if device.type == "cuda":
            index = 0 if device.index is None else device.index
            if index >= gpu_count:
                raise ValueError("Requested {}, but PyTorch sees only {} CUDA GPU(s).".format(requested, gpu_count))
            return "cuda:{}".format(index)
        if device.type != "cpu":
            raise ValueError("Only CPU and CUDA devices are supported, got {!r}.".format(requested))
        return "cpu"

    devices = resolve(args.mlp2_device, 0), resolve(args.mlp3_device, 1)
    print("PyTorch видит GPU: {} | MLP-2: {} | MLP-3: {}".format(gpu_count, devices[0], devices[1]))
    if args.parallel and gpu_count == 1 and devices == ("cuda:0", "cuda:0"):
        print("ВНИМАНИЕ: обе MLP используют одну GPU; параллельный режим может быть медленнее или вызвать OOM.")
    if args.parallel and gpu_count == 0:
        print("ВНИМАНИЕ: CUDA не найдена; обе MLP и XGBoost будут конкурировать за CPU.")
    return devices


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("\n=== ПОДГОТОВКА ДАННЫХ ===")
    daily = build_daily(args)
    frame, features = build_dataset(daily, args.horizon)
    train_frame, valid, test = time_split(frame, args.validation_days, args.test_days)
    X_train = train_frame[features].fillna(0.0)
    X_valid = valid[features].fillna(0.0)
    X_test = test[features].fillna(0.0)
    print("ИНН: {:,} | дневных строк: {:,} | признаков: {}".format(daily.inn.nunique(), len(daily), len(features)))
    print("Train: {:,} | Validation: {:,} | Test: {:,}".format(len(train_frame), len(valid), len(test)))
    print("Горизонт: следующие {} календарных дней".format(args.horizon))

    reports, predictions = [], []
    total_cpus = max(1, os.cpu_count() or 1)
    xgb_jobs = args.xgb_threads or (max(1, total_cpus // 2) if args.parallel else total_cpus)
    tasks = []

    try:
        import xgboost  # noqa: F401

        def run_xgboost():
            print("\n=== ОБУЧЕНИЕ XGBOOST (CPU: {} ПОТОКОВ) ===".format(xgb_jobs))
            started = time.perf_counter()
            values = train_xgboost_daily(
                X_train, train_frame, X_valid, valid, X_test, args.horizon,
                args.seed, args.xgb_estimators, xgb_jobs,
            )
            return collect_results(
                "xgboost", valid, test, values[0], values[2], values[3],
                args.horizon, time.perf_counter() - started,
            )

        tasks.append(("xgboost", run_xgboost))
    except ImportError:
        print("\nXGBoost не установлен — пропускаю. Команда: pip install xgboost")

    try:
        import torch

        torch_threads = args.torch_cpu_threads or (max(1, total_cpus // 4) if args.parallel else total_cpus)
        torch.set_num_threads(torch_threads)
        devices = resolve_torch_devices(args)
        for index, (name, layers) in enumerate((
            ("torch_mlp_2_layers", (128, 64)),
            ("torch_mlp_3_layers", (256, 128, 64)),
        )):
            device_name = devices[index]

            def run_mlp(model_name=name, model_layers=layers, model_device=device_name):
                print("\n=== ОБУЧЕНИЕ {} НА {} ===".format(model_name.upper(), model_device))
                started = time.perf_counter()
                values = train_torch_multitask(
                    model_layers, X_train, train_frame, X_valid, valid, X_test,
                    args.horizon, args.seed, args.epochs, args.batch_size, model_device,
                )
                return collect_results(
                    model_name, valid, test, values[0], values[2], values[3],
                    args.horizon, time.perf_counter() - started,
                )

            tasks.append((name, run_mlp))
    except ImportError:
        print("\nPyTorch не установлен — пропускаю нейросети. Команда: pip install torch")

    if args.parallel and len(tasks) > 1:
        print("\n=== ПАРАЛЛЕЛЬНЫЙ ЗАПУСК {} МОДЕЛЕЙ ===".format(len(tasks)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="cashflow") as executor:
            future_names = {executor.submit(task): name for name, task in tasks}
            for future in concurrent.futures.as_completed(future_names):
                name = future_names[future]
                rows, prediction = future.result()
                reports.extend(rows)
                predictions.append(prediction)
                print("\n>>> {} завершена".format(name))
    else:
        print("\n=== ПОСЛЕДОВАТЕЛЬНЫЙ ЗАПУСК {} МОДЕЛЕЙ ===".format(len(tasks)))
        for name, task in tasks:
            rows, prediction = task()
            reports.extend(rows)
            predictions.append(prediction)
            print("\n>>> {} завершена".format(name))

    if not predictions:
        raise RuntimeError("No model was trained. Install xgboost and/or torch.")
    metrics_frame = pd.DataFrame(reports)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(output / "daily_metrics.csv", index=False)
    predictions_frame.to_parquet(output / "daily_test_predictions.parquet", index=False)
    config = vars(args).copy()
    config["feature_columns"] = features
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = metrics_frame[metrics_frame.horizon_day.eq(0)].sort_values("pr_auc", ascending=False)
    print("\n=== ИТОГОВОЕ КАЧЕСТВО НА TEST ===")
    print(summary[["model", "pr_auc", "roc_auc", "precision", "recall", "training_seconds"]].to_string(index=False))
    print("\nРезультаты сохранены: {}".format(output.resolve()))


def money(value: float) -> str:
    sign = "−" if value < 0 else "+"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return "{}{:.2f} млрд ₽".format(sign, absolute / 1_000_000_000)
    if absolute >= 1_000_000:
        return "{}{:.2f} млн ₽".format(sign, absolute / 1_000_000)
    if absolute >= 1_000:
        return "{}{:.0f} тыс ₽".format(sign, absolute / 1_000)
    return "{}{:.0f} ₽".format(sign, absolute)


def color(text: str, code: str, enabled: bool) -> str:
    return "\033[{}m{}\033[0m".format(code, text) if enabled else text


def risk_label(probability: float) -> Tuple[str, str]:
    if probability >= 0.70:
        return "ВЫСОКИЙ", "31"
    if probability >= 0.50:
        return "ПОВЫШЕН", "33"
    if probability >= 0.30:
        return "СРЕДНИЙ", "93"
    return "НИЗКИЙ", "32"


def demo(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    predictions_path = output / "daily_test_predictions.parquet"
    metrics_path = output / "daily_metrics.csv"
    if not predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run training first; demo artifacts were not found in {}".format(output.resolve()))
    predictions = pd.read_parquet(predictions_path)
    metrics_frame = pd.read_csv(metrics_path)
    aggregate = metrics_frame[metrics_frame.horizon_day.eq(0)].sort_values("pr_auc", ascending=False)
    model = str(aggregate.iloc[0].model) if args.demo_model == "best" else args.demo_model
    available = set(predictions.model.unique())
    if model not in available:
        raise ValueError("Model {!r} not found. Available: {}".format(model, sorted(available)))
    model_predictions = predictions[predictions.model.eq(model)].copy()
    grouped = model_predictions.groupby(["inn", "origin_date"])
    complete = grouped.horizon_day.nunique()
    future_volume = grouped[["actual_inflow", "actual_outflow"]].sum().sum(axis=1)
    candidates = list(complete[complete.eq(model_predictions.horizon_day.max()) & future_volume.gt(0)].index)
    if not candidates:
        raise ValueError("No complete forecast windows found in test predictions.")
    chooser = random.Random(args.demo_seed) if args.demo_seed is not None else secrets.SystemRandom()
    selected_inn, selected_date = chooser.choice(candidates)
    sample = model_predictions[
        model_predictions.inn.eq(selected_inn) & model_predictions.origin_date.eq(selected_date)
    ].sort_values("horizon_day")

    use_color = not args.no_color
    print("\n" + "═" * 105)
    print(color("  ДЕМО: ЕЖЕДНЕВНЫЙ ПРОГНОЗ ДЕНЕЖНОГО ПОТОКА", "1;36", use_color))
    print("  Компания (ИНН): {}".format(selected_inn))
    print("  Прогноз сформирован на конец: {}".format(pd.Timestamp(selected_date).strftime("%d.%m.%Y")))
    print("  Модель: {} | Горизонт: {} дней".format(model, len(sample)))
    print("  Сценарий случайно выбран из скрытой test-выборки и сменится при следующем запуске demo.")
    print("═" * 105)
    header = " День | Дата       | Риск минуса | Уровень   | Прогноз прихода | Прогноз расхода | Чистый поток | Факт"
    print(header)
    print("─" * 105)
    for row in sample.itertuples(index=False):
        label, code = risk_label(float(row.probability_negative))
        risk = color("{:5.1f}%".format(row.probability_negative * 100), code, use_color)
        level = color("{:9s}".format(label), code, use_color)
        fact = "минус" if row.actual_net_flow < 0 else "плюс"
        fact = color(fact, "31" if row.actual_net_flow < 0 else "32", use_color)
        print(" +{:>3} | {} | {:>14} | {} | {:>15} | {:>15} | {:>12} | {} ({})".format(
            row.horizon_day, pd.Timestamp(row.forecast_date).strftime("%d.%m.%Y"), risk, level,
            money(row.predicted_inflow), money(row.predicted_outflow), money(row.predicted_net_flow),
            fact, money(row.actual_net_flow),
        ))
    print("─" * 105)
    high = sample[sample.probability_negative.ge(0.70)]
    worst = sample.loc[sample.probability_negative.idxmax()]
    predicted_total = float(sample.predicted_net_flow.sum())
    actual_total = float(sample.actual_net_flow.sum())
    print(color("  БИЗНЕС-ВЫВОД", "1;36", use_color))
    if len(high):
        dates = ", ".join(pd.to_datetime(high.forecast_date).dt.strftime("%d.%m").tolist())
        print(color("  • Высокий риск отрицательного потока: {} дн. ({})".format(len(high), dates), "31", use_color))
    else:
        print(color("  • Дней с высоким риском не обнаружено.", "32", use_color))
    print("  • Максимальный риск: {:.1f}% на {}.".format(
        worst.probability_negative * 100, pd.Timestamp(worst.forecast_date).strftime("%d.%m.%Y")))
    print("  • Прогноз чистого потока за период: {}.".format(money(predicted_total)))
    print("  • Исторический факт за период: {}.".format(money(actual_total)))
    print("  • Это проверка на скрытой исторической test-выборке: факт показан только для оценки модели.")
    print("  • Отрицательный поток не равен кассовому разрыву: для него нужны остатки на счетах.")
    print("═" * 105 + "\n")


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1")
    if args.mode in ("train", "train-demo"):
        train(args)
    if args.mode in ("demo", "train-demo"):
        demo(args)


if __name__ == "__main__":
    main()
