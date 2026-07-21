from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from app.model_plugins.base import BaseModelPlugin


class SeasonalNaivePlugin(BaseModelPlugin):
    def fit(self, prepared, parameters=None):
        self._started = time.perf_counter()
        self._model = {"season_length": int((parameters or {}).get("season_length", 12))}
        return self._model

    def predict(self, prepared, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        season = self._model["season_length"]
        output = []
        for series_id, context in prepared.context.groupby("series_id", sort=False):
            history = context.sort_values("month")[prepared.target].astype(float).tolist()
            residual = np.diff(history)
            spread = float(np.std(residual)) * 1.28 if len(residual) else 0.0
            test = prepared.test[prepared.test["series_id"] == series_id].sort_values("month")
            for _, actual in test.iterrows():
                point = history[-season] if len(history) >= season else history[-1]
                output.append({"series_id": str(series_id), "date": f"{actual['month']}-01", "actual": float(actual[prepared.target]), "forecast": float(point), "lower_bound": float(point - spread), "upper_bound": float(point + spread), "target": prepared.target})
                history.append(float(point))
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = len(prepared.series_ids)
        return result
