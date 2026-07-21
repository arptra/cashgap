from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


FORECAST_MODELS = {"seasonal_naive", "auto_ets", "auto_arima", "lightgbm_forecast"}


@dataclass
class ForecastResult:
    model: Any
    predictions: pd.DataFrame
    metrics: dict[str, float | None]
    parameters: dict[str, Any]


def prepare_flow_series(canonical: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    monthly = canonical.groupby("month", as_index=False)[["debit_sum", "credit_sum"]].sum()
    monthly["date"] = pd.to_datetime(monthly["month"] + "-01")
    if (monthly["credit_sum"] > 0).any() and (monthly["debit_sum"] > 0).any():
        monthly["y"] = monthly["credit_sum"] - monthly["debit_sum"]
        target = "net_flow"
    elif (monthly["debit_sum"] > 0).any():
        monthly["y"] = monthly["debit_sum"]
        target = "debit_flow"
    else:
        monthly["y"] = monthly["credit_sum"]
        target = "credit_flow"
    return monthly[["date", "y"]].sort_values("date").reset_index(drop=True), target


def temporal_forecast_split(series: pd.DataFrame, horizon: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) < 6:
        raise ValueError("Forecasting requires at least six monthly observations")
    test_size = horizon or max(2, int(np.ceil(len(series) * 0.2)))
    test_size = min(test_size, len(series) - 4)
    return series.iloc[:-test_size].copy(), series.iloc[-test_size:].copy()


def forecasting_metrics(train: np.ndarray, actual: np.ndarray, forecast: np.ndarray, seconds: float) -> dict:
    errors = actual - forecast
    denominator = float(np.abs(actual).sum())
    naive_scale = float(np.abs(np.diff(train)).mean()) if len(train) > 1 else 0.0
    return {
        "mae": float(mean_absolute_error(actual, forecast)),
        "rmse": float(np.sqrt(mean_squared_error(actual, forecast))),
        "wape": float(np.abs(errors).sum() / denominator) if denominator else None,
        "mase": float(np.abs(errors).mean() / naive_scale) if naive_scale > 0 else None,
        "training_seconds": float(seconds),
        "test_rows": int(len(actual)),
    }


def _statsforecast(model_name: str, train: pd.DataFrame, horizon: int, params: dict[str, Any]):
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    except ImportError as exc:
        raise RuntimeError("StatsForecast is not installed. Run make setup.") from exc
    season_length = int(params.get("season_length", min(12, max(1, len(train) // 2))))
    if model_name == "seasonal_naive":
        model = SeasonalNaive(season_length=season_length)
    elif model_name == "auto_ets":
        model = AutoETS(season_length=season_length)
    else:
        model = AutoARIMA(season_length=season_length)
    sf = StatsForecast(models=[model], freq="MS", n_jobs=1)
    training = train.rename(columns={"date": "ds"}).assign(unique_id="CASHFLOW")
    started = time.perf_counter()
    output = sf.forecast(df=training[["unique_id", "ds", "y"]], h=horizon, level=[80])
    seconds = time.perf_counter() - started
    value_columns = [column for column in output.columns if column not in {"unique_id", "ds"} and "-lo-" not in column and "-hi-" not in column]
    value_column = value_columns[0]
    lower_column = next((column for column in output.columns if "-lo-80" in column), None)
    upper_column = next((column for column in output.columns if "-hi-80" in column), None)
    return sf, output[value_column].to_numpy(float), (
        output[lower_column].to_numpy(float) if lower_column else output[value_column].to_numpy(float)
    ), (
        output[upper_column].to_numpy(float) if upper_column else output[value_column].to_numpy(float)
    ), seconds, {"season_length": season_length}


def _lag_matrix(values: list[float], lags: tuple[int, ...]) -> list[float]:
    return [values[-lag] if len(values) >= lag else values[0] for lag in lags]


def _lightgbm_forecast(train: pd.DataFrame, horizon: int, params: dict[str, Any]):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("LightGBM is not installed. Run make setup.") from exc
    lags = (1, 2, 3, 6)
    values = train["y"].astype(float).tolist()
    rows, targets = [], []
    for index in range(max(lags), len(values)):
        rows.append([values[index - lag] for lag in lags])
        targets.append(values[index])
    if len(rows) < 2:
        lags = (1, 2)
        rows, targets = [], []
        for index in range(max(lags), len(values)):
            rows.append([values[index - lag] for lag in lags])
            targets.append(values[index])
    effective = {
        "n_estimators": int(params.get("n_estimators", 150)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "num_leaves": int(params.get("num_leaves", 15)),
        "verbosity": -1,
        "n_jobs": 2,
        "random_state": 42,
    }
    model = LGBMRegressor(**effective)
    started = time.perf_counter()
    model.fit(np.asarray(rows), np.asarray(targets))
    seconds = time.perf_counter() - started
    history = values.copy()
    forecast = []
    for _ in range(horizon):
        prediction = float(model.predict(np.asarray([_lag_matrix(history, lags)]))[0])
        forecast.append(prediction)
        history.append(prediction)
    residuals = np.asarray(targets) - model.predict(np.asarray(rows))
    spread = float(np.std(residuals)) * 1.28 if len(residuals) > 1 else 0.0
    forecast_values = np.asarray(forecast)
    return model, forecast_values, forecast_values - spread, forecast_values + spread, seconds, {**effective, "lags": list(lags)}


def train_forecaster(model_name: str, series: pd.DataFrame, params: dict[str, Any] | None = None) -> ForecastResult:
    if model_name not in FORECAST_MODELS:
        raise ValueError(f"Unknown forecasting model: {model_name}")
    params = params or {}
    train, test = temporal_forecast_split(series, params.get("horizon"))
    if model_name == "lightgbm_forecast":
        model, forecast, lower, upper, seconds, effective = _lightgbm_forecast(train, len(test), params)
    else:
        model, forecast, lower, upper, seconds, effective = _statsforecast(model_name, train, len(test), params)
    predictions = pd.DataFrame(
        {
            "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "actual": test["y"].to_numpy(float),
            "forecast": forecast,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )
    metrics = forecasting_metrics(train["y"].to_numpy(float), test["y"].to_numpy(float), forecast, seconds)
    return ForecastResult(model=model, predictions=predictions, metrics=metrics, parameters=effective)

