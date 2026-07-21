from __future__ import annotations

import importlib.util
import time
from typing import Any

import pandas as pd

from app.model_plugins.base import BaseModelPlugin
from app.models_registry.installer import cache_metadata, model_install_dir
from app.models_registry.schemas import EnvironmentReport, ModelStatus


class Chronos2Plugin(BaseModelPlugin):
    def check_environment(self) -> EnvironmentReport:
        dependency = importlib.util.find_spec("chronos") is not None
        info = cache_metadata(self.spec)
        if dependency and info["installed"]:
            return EnvironmentReport(status=ModelStatus.INSTALLED, message="Chronos-2 DataFrame adapter готов для CPU", installed=True, weights_cached=True, size_bytes=info["size_bytes"], revision=info.get("revision"))
        return EnvironmentReport(status=ModelStatus.NOT_INSTALLED if not dependency else ModelStatus.AVAILABLE, message="Установите chronos-forecasting" if not dependency else "Скачайте Chronos-2 weights", dependency_installed=dependency, weights_cached=info["installed"], size_bytes=info["size_bytes"], revision=info.get("revision"), install_command=self.spec.install_command)

    def fit(self, prepared, parameters=None):
        from chronos import Chronos2Pipeline
        import torch

        if not cache_metadata(self.spec)["installed"]:
            raise RuntimeError("Chronos-2 weights are not installed")
        self._started = time.perf_counter()
        self._model = Chronos2Pipeline.from_pretrained(str(model_install_dir(self.spec)), device_map="cpu", torch_dtype=torch.float32, local_files_only=True)
        return self._model

    def predict(self, prepared, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        parameters = parameters or {}
        batch_size = int(parameters.get("batch_size", 16))
        output = []
        attempted = len(prepared.series_ids)
        errors: list[str] = []
        for ids in prepared.batches(batch_size):
            context = prepared.context[prepared.context["series_id"].isin(ids)][["series_id", "month", prepared.target]].copy()
            context["timestamp"] = pd.to_datetime(context.pop("month") + "-01")
            context["target"] = context.pop(prepared.target).astype(float)
            try:
                forecast = self._model.predict_df(context, prediction_length=prepared.horizon, quantile_levels=[0.1, 0.5, 0.9], id_column="series_id", timestamp_column="timestamp", target="target")
                for _, row in forecast.iterrows():
                    series_id = str(row["series_id"])
                    month = pd.Timestamp(row["timestamp"]).strftime("%Y-%m")
                    actual_rows = prepared.test[(prepared.test["series_id"] == series_id) & (prepared.test["month"] == month)]
                    if actual_rows.empty:
                        continue
                    output.append({"series_id": series_id, "date": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d"), "actual": float(actual_rows.iloc[0][prepared.target]), "forecast": float(row["predictions"]), "lower_bound": float(row.get("0.1", row["predictions"])), "median": float(row.get("0.5", row["predictions"])), "upper_bound": float(row.get("0.9", row["predictions"])), "target": prepared.target})
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = attempted
        if result.empty:
            detail = errors[-1] if errors else "unknown adapter error"
            raise RuntimeError(f"Chronos-2 failed to process every series; no fallback values were generated: {detail}")
        return result
