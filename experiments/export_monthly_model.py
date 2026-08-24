#!/usr/bin/env python3
"""Обучить финальную месячную модель и сохранить пакет для API.

Пакет содержит checkpoint модели, метаданные, последние 12 месяцев истории и
предрассчитанные прогнозы. Первый будущий месяц является прямым прогнозом,
последующие месяцы — рекурсивными.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "threadpoolctl": "threadpoolctl",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: не установлены библиотеки: {}".format(", ".join(missing)))
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
        print("Jupyter: %pip install {}".format(" ".join(missing)))
        print("После установки перезапустите kernel Jupyter.")
        raise SystemExit(2)
    print("Проверка зависимостей: библиотеки экспорта модели установлены.")


if __name__ == "__main__":
    _check_dependencies()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from experiments.benchmark_monthly_cashflow import (
        MODEL_NAMES,
        _activation,
        build_monthly_dataset,
        load_tuned_params,
        parse_layers,
        target_matrix,
    )
    from experiments.monthly_objective import (
        MONTHLY_OBJECTIVE_NAME,
        MONTHLY_OBJECTIVE_VERSION,
        baseline_from_feature_matrix,
        fit_feature_normalizer,
        fit_residual_scale,
        normalize_features,
        restore_residual_predictions,
        scale_residual_targets,
    )
    from experiments.monthly_reports_ru import model_name_ru
    from experiments.train_cashflow_proxy import build_observed_daily
except ImportError:
    from benchmark_monthly_cashflow import (
        MODEL_NAMES,
        _activation,
        build_monthly_dataset,
        load_tuned_params,
        parse_layers,
        target_matrix,
    )
    from monthly_objective import (
        MONTHLY_OBJECTIVE_NAME,
        MONTHLY_OBJECTIVE_VERSION,
        baseline_from_feature_matrix,
        fit_feature_normalizer,
        fit_residual_scale,
        normalize_features,
        restore_residual_predictions,
        scale_residual_targets,
    )
    from monthly_reports_ru import model_name_ru
    from train_cashflow_proxy import build_observed_daily


SOURCE_COLUMNS = [
    "target_inflow",
    "target_outflow",
    "target_net_flow",
    "active_inflow_days",
    "active_outflow_days",
    "negative_days",
]
SUPPORTED_MODELS = tuple(MODEL_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outflow", required=True)
    parser.add_argument("--inflow", required=True)
    parser.add_argument("--output-dir", required=True, help="Каталог пакета модели")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--forecast-months", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlp2-layers", default="512,256")
    parser.add_argument("--mlp3-layers", default="768,512,256")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--boosting-iterations", type=int, default=250)
    parser.add_argument("--mlp-params", default=None)
    parser.add_argument("--boosting-params", default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        index = 0 if device.index is None else int(device.index)
        if index >= count:
            raise ValueError("Запрошена {}, но PyTorch видит GPU: {}.".format(requested, count))
        return "cuda:{}".format(index)
    return "cpu"


def _fit_sklearn(
    model_id: str,
    frame: pd.DataFrame,
    features: Sequence[str],
    args: argparse.Namespace,
    output: Path,
) -> Tuple[Callable[[pd.DataFrame], np.ndarray], str]:
    X = frame[list(features)].fillna(0.0).astype(np.float32, copy=False)
    baseline = baseline_from_feature_matrix(X.to_numpy(), list(X.columns))
    y = target_matrix(frame) - baseline
    if model_id == "linear_regression":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    else:
        options = {
            "max_iter": args.boosting_iterations,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.1,
            "early_stopping": True,
            "random_state": args.seed,
        }
        tuned = load_tuned_params(args.boosting_params, "gradient_boosting")
        if tuned:
            options.update(tuned)
        estimator = MultiOutputRegressor(HistGradientBoostingRegressor(**options), n_jobs=1)
    print("Обучение {} на {:,} строках...".format(model_name_ru(model_id), len(X)))
    with threadpool_limits(limits=max(1, args.cpu_threads)):
        estimator.fit(X, y)
    model_file = output / "model.joblib"
    joblib.dump(estimator, model_file, compress=3)

    def predict(X_future: pd.DataFrame) -> np.ndarray:
        future_baseline = baseline_from_feature_matrix(
            X_future.to_numpy(), list(X_future.columns)
        )
        return np.maximum(future_baseline + estimator.predict(X_future), 0.0)

    return predict, model_file.name


def _fit_torch(
    model_id: str,
    frame: pd.DataFrame,
    features: Sequence[str],
    args: argparse.Namespace,
    output: Path,
) -> Tuple[Callable[[pd.DataFrame], np.ndarray], str, str, List[int]]:
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "Для сохранения MLP нужен torch. Python 3.8: %pip install torch==2.3.1 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    import torch
    from torch import nn

    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    tuned = load_tuned_params(args.mlp_params, "mlp") if model_id == "torch_mlp_tuned" else None
    if model_id == "torch_mlp_2_layers":
        layers = list(parse_layers(args.mlp2_layers, "--mlp2-layers"))
    elif model_id == "torch_mlp_3_layers":
        layers = list(parse_layers(args.mlp3_layers, "--mlp3-layers"))
    else:
        if not tuned:
            raise ValueError("torch_mlp_tuned требует --mlp-params от autotюнинга.")
        layers = [int(value) for value in tuned.get("layers", [256, 128, 64])]

    options = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "activation": "relu",
        "patience": 12,
    }
    if tuned:
        for key in options:
            if key in tuned:
                options[key] = tuned[key]
    batch_size = int(tuned.get("batch_size", args.batch_size)) if tuned else args.batch_size
    validation_month = pd.Timestamp(frame["month"].max())
    train = frame[frame["month"].lt(validation_month)]
    valid = frame[frame["month"].eq(validation_month)]
    if train.empty or valid.empty:
        raise ValueError("Недостаточно месяцев для train/validation финальной MLP.")
    X_train = train[list(features)].fillna(0.0).astype(np.float32, copy=False)
    X_valid = valid[list(features)].fillna(0.0).astype(np.float32, copy=False)
    feature_mean, feature_scale, active_features = fit_feature_normalizer(X_train.to_numpy())
    x_train_np = normalize_features(
        X_train.to_numpy(), feature_mean, feature_scale, active_features
    )
    x_valid_np = normalize_features(
        X_valid.to_numpy(), feature_mean, feature_scale, active_features
    )
    baseline_train = baseline_from_feature_matrix(X_train.to_numpy(), list(X_train.columns))
    baseline_valid = baseline_from_feature_matrix(X_valid.to_numpy(), list(X_valid.columns))
    train_residual = target_matrix(train) - baseline_train
    valid_residual = target_matrix(valid) - baseline_valid
    residual_scale = fit_residual_scale(train_residual)
    y_train_np = scale_residual_targets(train_residual, residual_scale)
    y_valid_np = scale_residual_targets(valid_residual, residual_scale)

    torch.manual_seed(args.seed)
    x_train = torch.as_tensor(x_train_np, device=device)
    x_valid = torch.as_tensor(x_valid_np, device=device)
    y_train = torch.as_tensor(y_train_np, device=device)
    y_valid = torch.as_tensor(y_valid_np, device=device)
    activation_class = _activation(str(options["activation"]))
    modules = []
    width = len(features)
    for hidden in layers:
        modules.extend([
            nn.Linear(width, int(hidden)),
            activation_class(),
            nn.Dropout(float(options["dropout"])),
        ])
        width = int(hidden)
    output_layer = nn.Linear(width, 2)
    nn.init.zeros_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)
    modules.append(output_layer)
    model = nn.Sequential(*modules).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        best_loss = float(loss_function(model(x_valid), y_valid))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    patience = int(options["patience"])
    remaining = patience
    print(
        "Обучение {} | устройство {} | слои {} | строк {:,}".format(
            model_name_ru(model_id), device_name, layers, len(train)
        )
    )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + epoch)
        order = torch.randperm(len(x_train), generator=generator, device=device)
        for start in range(0, len(x_train), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[indices]), y_train[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_valid), y_valid))
        if epoch == 1 or epoch % 10 == 0:
            print("  эпоха {:3d} | ошибка валидации {:.5f}".format(epoch, validation_loss))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            remaining = patience
        else:
            remaining -= 1
            if remaining == 0:
                print("  ранняя остановка на эпохе {}".format(epoch))
                break
    if best_state is None:
        raise RuntimeError("Не удалось получить checkpoint MLP.")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        "format_version": 2,
        "model_id": model_id,
        "objective_version": MONTHLY_OBJECTIVE_VERSION,
        "objective_name": MONTHLY_OBJECTIVE_NAME,
        "features": list(features),
        "layers": layers,
        "activation": str(options["activation"]),
        "dropout": float(options["dropout"]),
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_scale": feature_scale.astype(float).tolist(),
        "active_features": active_features.astype(bool).tolist(),
        "residual_scale": residual_scale.astype(float).tolist(),
        "state_dict": best_state,
    }
    model_file = output / "model.pt"
    torch.save(checkpoint, model_file)
    print("Checkpoint MLP сохранён за {:.1f} сек: {}".format(
        time.perf_counter() - started, model_file.resolve()
    ))

    def predict(X_future: pd.DataFrame) -> np.ndarray:
        future_baseline = baseline_from_feature_matrix(
            X_future.to_numpy(), list(X_future.columns)
        )
        values = normalize_features(
            X_future.to_numpy(), feature_mean, feature_scale, active_features
        )
        parts = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                tensor = torch.as_tensor(values[start:start + batch_size], device=device)
                parts.append(model(tensor).cpu().numpy())
        scaled = np.concatenate(parts, axis=0)
        return restore_residual_predictions(future_baseline, scaled, residual_scale)

    return predict, model_file.name, device_name, layers


def _nan_mean(values: np.ndarray) -> np.ndarray:
    count = np.sum(~np.isnan(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def _nan_std(values: np.ndarray) -> np.ndarray:
    count = np.sum(~np.isnan(values), axis=1)
    mean = _nan_mean(values)
    centered = np.where(np.isnan(values), 0.0, values - mean[:, None])
    variance = np.divide(
        np.sum(centered ** 2, axis=1),
        count - 1,
        out=np.full(len(values), np.nan),
        where=count > 1,
    )
    return np.sqrt(variance)


def _serving_history(monthly: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    last_month = pd.Timestamp(monthly["month"].max())
    ordered = monthly.sort_values(["inn", "month"])
    inns = ordered["inn"].astype(str).drop_duplicates().to_numpy()
    # Keep history in float64 so serving features match pandas training features
    # before the final shared float32 cast used by every model.
    history = np.full((len(inns), 12, len(SOURCE_COLUMNS)), np.nan, dtype=np.float64)
    counts = np.zeros(len(inns), dtype=np.int32)
    positions = {inn: index for index, inn in enumerate(inns)}
    for inn, group in ordered.groupby("inn", sort=False):
        index = positions[str(inn)]
        values = group[SOURCE_COLUMNS].to_numpy(dtype=np.float64)
        company_last_month = pd.Timestamp(group["month"].max())
        inactive_months = (
            (last_month.year - company_last_month.year) * 12
            + last_month.month - company_last_month.month
        )
        if inactive_months >= 12:
            values = np.zeros((12, len(SOURCE_COLUMNS)), dtype=np.float64)
        elif inactive_months > 0:
            values = np.concatenate([
                values,
                np.zeros((inactive_months, len(SOURCE_COLUMNS)), dtype=np.float64),
            ], axis=0)
        values = values[-12:]
        history[index, -len(values):, :] = values
        counts[index] = len(group) + inactive_months

    recent_months = pd.date_range(end=last_month, periods=12, freq="MS")
    recent_history = pd.DataFrame({
        "inn": np.repeat(inns, 12),
        "month": np.tile(recent_months.to_numpy(), len(inns)),
    })
    flat_history = history.reshape(len(inns) * 12, len(SOURCE_COLUMNS))
    for source_index, source in enumerate(SOURCE_COLUMNS):
        recent_history[source] = flat_history[:, source_index]
    recent_history = recent_history.dropna(subset=SOURCE_COLUMNS, how="all")
    return inns, history, counts, recent_history


def _future_features(
    history: np.ndarray,
    counts: np.ndarray,
    period: pd.Timestamp,
    features: Sequence[str],
) -> pd.DataFrame:
    values: Dict[str, np.ndarray] = {}
    for source_index, source in enumerate(SOURCE_COLUMNS):
        source_history = history[:, :, source_index]
        for lag in (1, 2, 3, 6, 12):
            values["{}_lag_{}".format(source, lag)] = source_history[:, -lag]
        for window in (3, 6, 12):
            window_values = source_history[:, -window:]
            values["{}_mean_{}".format(source, window)] = _nan_mean(window_values)
            values["{}_std_{}".format(source, window)] = _nan_std(window_values)
    values["inflow_change_1"] = (
        values["target_inflow_lag_1"] - values["target_inflow_lag_2"]
    )
    values["outflow_change_1"] = (
        values["target_outflow_lag_1"] - values["target_outflow_lag_2"]
    )
    values["inflow_ratio_to_mean_6"] = values["target_inflow_lag_1"] / np.maximum(
        values["target_inflow_mean_6"], 1.0
    )
    values["outflow_ratio_to_mean_6"] = values["target_outflow_lag_1"] / np.maximum(
        values["target_outflow_mean_6"], 1.0
    )
    values["month_sin"] = np.full(len(history), math.sin(2 * math.pi * period.month / 12.0))
    values["month_cos"] = np.full(len(history), math.cos(2 * math.pi * period.month / 12.0))
    values["history_months"] = counts.astype(np.float32)
    missing = [column for column in features if column not in values]
    if missing:
        raise ValueError("Не удалось построить признаки для API: {}".format(missing))
    return pd.DataFrame(values)[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
        np.float32, copy=False
    )


def build_future_forecasts(
    monthly: pd.DataFrame,
    features: Sequence[str],
    predict: Callable[[pd.DataFrame], np.ndarray],
    model_id: str,
    forecast_months: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if forecast_months < 1:
        raise ValueError("--forecast-months должен быть положительным.")
    inns, history, counts, recent_history = _serving_history(monthly)
    if len(inns) == 0:
        raise ValueError("Нет ИНН для формирования прогнозов API.")
    last_month = pd.Timestamp(monthly["month"].max())
    parts = []
    for step in range(1, forecast_months + 1):
        period = last_month + pd.offsets.MonthBegin(step)
        X_future = _future_features(history, counts, period, features)
        prediction = predict(X_future)
        net = prediction[:, 0] - prediction[:, 1]
        parts.append(pd.DataFrame({
            "inn": inns,
            "period": period.strftime("%Y-%m"),
            "model": model_id,
            "forecast_step": step,
            "forecast_type": "direct" if step == 1 else "recursive",
            "predicted_inflow": prediction[:, 0],
            "predicted_outflow": prediction[:, 1],
            "predicted_net_flow": net,
            "negative_net_flow": net < 0,
        }))
        days_in_month = int(period.days_in_month)
        activity = np.rint(np.column_stack([
            _nan_mean(history[:, -3:, 3]),
            _nan_mean(history[:, -3:, 4]),
            _nan_mean(history[:, -3:, 5]),
        ]))
        activity = np.clip(np.nan_to_num(activity, nan=0.0), 0, days_in_month)
        next_sources = np.column_stack([
            prediction[:, 0], prediction[:, 1], net,
            activity[:, 0], activity[:, 1], activity[:, 2],
        ]).astype(np.float64)
        history[:, :-1, :] = history[:, 1:, :]
        history[:, -1, :] = next_sources
        counts += 1
        print("  Прогноз {} из {}: {} | ИНН {:,}".format(
            step, forecast_months, period.strftime("%Y-%m"), len(inns)
        ))
    return pd.concat(parts, ignore_index=True), recent_history


def _write_forecast_reports(forecasts: pd.DataFrame, output: Path) -> None:
    forecasts.to_parquet(output / "forecasts_api.parquet", index=False)
    russian = forecasts.copy()
    russian["model_name_ru"] = russian["model"].map(model_name_ru)
    russian["forecast_type"] = russian["forecast_type"].map({
        "direct": "прямой",
        "recursive": "рекурсивный",
    })
    russian = russian.rename(columns={
        "inn": "инн",
        "period": "период",
        "model": "код_модели",
        "model_name_ru": "название_модели",
        "forecast_step": "шаг_прогноза_месяцев",
        "forecast_type": "тип_прогноза",
        "predicted_inflow": "прогноз_зачислений",
        "predicted_outflow": "прогноз_списаний",
        "predicted_net_flow": "прогноз_чистого_потока",
        "negative_net_flow": "отрицательный_чистый_поток",
    })
    russian.to_parquet(output / "прогнозы_для_api.parquet", index=False)
    russian.to_csv(
        output / "прогнозы_для_api.csv", index=False, sep=";", encoding="utf-8-sig",
        float_format="%.2f",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("\n=== ОБУЧЕНИЕ И СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ ===")
    observed_daily = build_observed_daily(args)
    monthly, features = build_monthly_dataset(observed_daily)
    del observed_daily
    model_id = args.model
    device_name = "cpu"
    layers: List[int] = []
    started = time.perf_counter()
    if model_id == "trailing_mean":
        model_file = output / "model.json"
        model_file.write_text(json.dumps({
            "model": model_id,
            "rule": "mean of previous 3 months with previous month fallback",
            "objective_version": MONTHLY_OBJECTIVE_VERSION,
            "objective_name": MONTHLY_OBJECTIVE_NAME,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        def predictor(X_future: pd.DataFrame) -> np.ndarray:
            inflow = X_future["target_inflow_mean_3"].to_numpy(dtype=float)
            outflow = X_future["target_outflow_mean_3"].to_numpy(dtype=float)
            return np.maximum(np.column_stack([inflow, outflow]), 0.0)

        model_filename = model_file.name
    elif model_id in {"linear_regression", "gradient_boosting"}:
        predictor, model_filename = _fit_sklearn(model_id, monthly, features, args, output)
    else:
        predictor, model_filename, device_name, layers = _fit_torch(
            model_id, monthly, features, args, output
        )

    forecasts, recent_history = build_future_forecasts(
        monthly, features, predictor, model_id, args.forecast_months
    )
    recent_history.to_parquet(output / "monthly_history.parquet", index=False)
    _write_forecast_reports(forecasts, output)
    last_month = pd.Timestamp(monthly["month"].max())
    model_display_name = model_name_ru(model_id)
    if layers:
        model_display_name = "{} ({})".format(
            model_display_name, " → ".join(str(value) for value in layers)
        )
    metadata = {
        "format_version": 2,
        "model_id": model_id,
        "objective_version": MONTHLY_OBJECTIVE_VERSION,
        "objective_name": MONTHLY_OBJECTIVE_NAME,
        "model_name_ru": model_display_name,
        "model_file": model_filename,
        "features": list(features),
        "feature_count": len(features),
        "last_complete_month": last_month.strftime("%Y-%m"),
        "first_forecast_month": (last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "forecast_months": args.forecast_months,
        "forecast_inns": int(forecasts["inn"].nunique()),
        "device_used_for_training": device_name,
        "layers": layers,
        "forecast_note_ru": (
            "Первый месяц — прямой прогноз. Последующие месяцы используют предыдущие "
            "прогнозы как историю и являются рекурсивными."
        ),
        "cash_gap_note_ru": (
            "Отрицательный чистый поток не равен кассовому разрыву: в данных нет остатков на счетах."
        ),
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    russian_metadata = {
        "код_модели": model_id,
        "название_модели": model_display_name,
        "файл_модели": model_filename,
        "последний_полный_месяц_данных": metadata["last_complete_month"],
        "первый_месяц_прогноза": metadata["first_forecast_month"],
        "месяцев_прогноза": args.forecast_months,
        "количество_инн": metadata["forecast_inns"],
        "целевая_функция": MONTHLY_OBJECTIVE_NAME,
        "примечание": metadata["forecast_note_ru"],
        "ограничение": metadata["cash_gap_note_ru"],
    }
    (output / "описание_модели.json").write_text(
        json.dumps(russian_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nМодель: {}".format((output / model_filename).resolve()))
    print("Прогнозы API: {}".format((output / "forecasts_api.parquet").resolve()))
    print("Общее время: {:.1f} сек".format(time.perf_counter() - started))


if __name__ == "__main__":
    main()
