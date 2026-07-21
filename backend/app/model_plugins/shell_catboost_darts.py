from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from app.ml.forecasting_data import PreparedForecastData
from app.model_plugins.competition_recipe import CompetitionRecipePlugin
from app.models_registry.schemas import EnvironmentReport, ModelStatus


LAGS = (1, 2, 3, 6)


def _features(history: list[float], month_number: int) -> list[float]:
    def lag(value: int) -> float:
        return history[-value] if len(history) >= value else history[0]

    roll3 = history[-3:]
    roll6 = history[-6:]
    return [
        *(lag(value) for value in LAGS),
        float(np.mean(roll3)), float(np.mean(roll6)),
        float(roll3[-1] - roll3[0]) if len(roll3) > 1 else 0.0,
        float(np.std(roll6)),
        float(np.sin(2 * np.pi * month_number / 12)),
        float(np.cos(2 * np.pi * month_number / 12)),
    ]


def _training_matrix(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for _, group in frame.groupby("series_id", sort=False):
        group = group.sort_values("month")
        values = group[column].astype(float).tolist()
        dates = pd.to_datetime(group["month"] + "-01")
        for index in range(max(LAGS), len(values)):
            rows.append(_features(values[:index], int(dates.iloc[index].month)))
            targets.append(values[index])
    if not rows:
        raise ValueError("Shell-style CatBoost requires at least seven context months per series")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


class ShellCatboostDartsPlugin(CompetitionRecipePlugin):
    def check_environment(self) -> EnvironmentReport:
        if self.spec.type == "local_trainable_model":
            return EnvironmentReport(status=ModelStatus.INSTALLED, message="Безопасный внутренний lag/rolling adapter готов", installed=True)
        return super().check_environment()

    def fit(self, prepared: PreparedForecastData, parameters: dict[str, Any] | None = None):
        parameters = parameters or {}
        self._started = time.perf_counter()
        effective = {
            "iterations": int(parameters.get("iterations", 120)),
            "depth": int(parameters.get("depth", 5)),
            "learning_rate": float(parameters.get("learning_rate", 0.05)),
            "loss_function": "RMSE",
            "verbose": False,
            "thread_count": 2,
            "random_seed": 42,
        }
        models: dict[str, Any] = {}
        residual_spread: dict[str, float] = {}
        for column in ("total_credit_sum", "total_debit_sum"):
            x, y = _training_matrix(prepared.context, column)
            model = CatBoostRegressor(**effective)
            model.fit(x, y)
            residual_spread[column] = float(np.std(y - model.predict(x))) * 1.28
            models[column] = model
        self._model = {"models": models, "spread": residual_spread, "parameters": effective}
        return self._model

    def predict(self, prepared: PreparedForecastData, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        if not self._model:
            raise RuntimeError("Shell recipe is not fitted")
        output: list[dict[str, Any]] = []
        attempted = len(prepared.series_ids)
        for series_id, context in prepared.context.groupby("series_id", sort=False):
            context = context.sort_values("month")
            test = prepared.test[prepared.test["series_id"] == series_id].sort_values("month")
            histories = {
                column: context[column].astype(float).tolist()
                for column in ("total_credit_sum", "total_debit_sum")
            }
            for _, actual_row in test.iterrows():
                month_number = int(pd.Timestamp(f"{actual_row['month']}-01").month)
                predicted: dict[str, float] = {}
                for column in ("total_credit_sum", "total_debit_sum"):
                    value = float(self._model["models"][column].predict(np.asarray([_features(histories[column], month_number)]))[0])
                    value = max(value, 0.0)
                    predicted[column] = value
                    histories[column].append(value)
                point = predicted[prepared.target] if prepared.target != "net_flow" else predicted["total_credit_sum"] - predicted["total_debit_sum"]
                spread = self._model["spread"].get(prepared.target, sum(self._model["spread"].values()))
                output.append(
                    {
                        "series_id": str(series_id), "date": f"{actual_row['month']}-01",
                        "actual": float(actual_row[prepared.target]), "forecast": point,
                        "lower_bound": point - spread, "upper_bound": point + spread,
                        "predicted_credit": predicted["total_credit_sum"],
                        "predicted_debit": predicted["total_debit_sum"],
                        "predicted_net_flow": predicted["total_credit_sum"] - predicted["total_debit_sum"],
                        "target": prepared.target,
                    }
                )
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = attempted
        return result

    def save_artifacts(self, output_dir, result):
        paths = super().save_artifacts(output_dir, result)
        model_path = output_dir / "model.joblib"
        joblib.dump(self._model, model_path)
        paths["model"] = str(model_path)
        return paths

    def load_artifacts(self, output_dir):
        result = super().load_artifacts(output_dir)
        self._model = joblib.load(output_dir / "model.joblib")
        result.model = self._model
        return result
