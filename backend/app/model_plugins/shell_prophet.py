from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.ml.forecasting_data import PreparedForecastData
from app.model_plugins.competition_recipe import CompetitionRecipePlugin


class ShellProphetPlugin(CompetitionRecipePlugin):
    optional_dependency = "prophet"

    def fit(self, prepared: PreparedForecastData, parameters: dict[str, Any] | None = None):
        from prophet import Prophet

        self._started = time.perf_counter()
        models: dict[str, dict[str, Any]] = {}
        for series_id, context in prepared.context.groupby("series_id", sort=False):
            models[str(series_id)] = {}
            for column in ("total_credit_sum", "total_debit_sum"):
                train = pd.DataFrame({"ds": pd.to_datetime(context["month"] + "-01"), "y": context[column].astype(float)})
                model = Prophet(yearly_seasonality=len(train) >= 18, weekly_seasonality=False, daily_seasonality=False, interval_width=0.8)
                model.fit(train)
                models[str(series_id)][column] = model
        self._model = models
        return models

    def predict(self, prepared: PreparedForecastData, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        output: list[dict[str, Any]] = []
        for series_id, test in prepared.test.groupby("series_id", sort=False):
            future = pd.DataFrame({"ds": pd.to_datetime(test.sort_values("month")["month"] + "-01")})
            forecasts = {column: self._model[str(series_id)][column].predict(future) for column in ("total_credit_sum", "total_debit_sum")}
            for index, (_, actual) in enumerate(test.sort_values("month").iterrows()):
                credit = max(float(forecasts["total_credit_sum"].iloc[index]["yhat"]), 0.0)
                debit = max(float(forecasts["total_debit_sum"].iloc[index]["yhat"]), 0.0)
                point = credit if prepared.target == "total_credit_sum" else debit if prepared.target == "total_debit_sum" else credit - debit
                selected = forecasts[prepared.target].iloc[index] if prepared.target != "net_flow" else None
                lower = float(selected["yhat_lower"]) if selected is not None else point
                upper = float(selected["yhat_upper"]) if selected is not None else point
                output.append({"series_id": str(series_id), "date": f"{actual['month']}-01", "actual": float(actual[prepared.target]), "forecast": point, "lower_bound": lower, "upper_bound": upper, "predicted_credit": credit, "predicted_debit": debit, "predicted_net_flow": credit - debit, "target": prepared.target})
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = len(prepared.series_ids)
        return result
