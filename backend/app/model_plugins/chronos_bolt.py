from __future__ import annotations

import importlib.util
import time
from typing import Any

import numpy as np
import pandas as pd

from app.model_plugins.base import BaseModelPlugin
from app.models_registry.installer import cache_metadata, model_install_dir
from app.models_registry.schemas import EnvironmentReport, ModelStatus


class ChronosBoltPlugin(BaseModelPlugin):
    def check_environment(self) -> EnvironmentReport:
        dependency = importlib.util.find_spec("chronos") is not None and importlib.util.find_spec("torch") is not None
        info = cache_metadata(self.spec)
        if dependency and info["installed"]:
            return EnvironmentReport(status=ModelStatus.INSTALLED, message="Chronos и safetensors weights готовы для CPU inference", installed=True, weights_cached=True, size_bytes=info["size_bytes"], revision=info.get("revision"))
        if not dependency:
            return EnvironmentReport(status=ModelStatus.NOT_INSTALLED, message="Не установлен пакет chronos-forecasting", installed=False, dependency_installed=False, weights_cached=info["installed"], size_bytes=info["size_bytes"], revision=info.get("revision"), install_command=self.spec.install_command)
        return EnvironmentReport(status=ModelStatus.AVAILABLE, message="Библиотека готова; скачайте safetensors weights", installed=False, dependency_installed=True)

    def fit(self, prepared, parameters=None):
        from chronos import BaseChronosPipeline
        import torch

        info = cache_metadata(self.spec)
        if not info["installed"]:
            raise RuntimeError("Chronos weights are not installed")
        self._started = time.perf_counter()
        self._model = BaseChronosPipeline.from_pretrained(
            str(model_install_dir(self.spec)), device_map="cpu", torch_dtype=torch.float32, local_files_only=True
        )
        return self._model

    def predict(self, prepared, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        import torch

        parameters = parameters or {}
        batch_size = int(parameters.get("batch_size", 16))
        output: list[dict[str, Any]] = []
        attempted = len(prepared.series_ids)
        errors: list[str] = []
        for ids in prepared.batches(batch_size):
            contexts = []
            for series_id in ids:
                values = prepared.context.loc[prepared.context["series_id"] == series_id, prepared.target].astype(float).to_numpy()
                contexts.append(torch.tensor(values, dtype=torch.float32))
            try:
                quantiles, mean = self._model.predict_quantiles(
                    contexts, prediction_length=prepared.horizon, quantile_levels=[0.1, 0.5, 0.9]
                )
                quantiles = np.asarray(quantiles)
                mean = np.asarray(mean)
                for item_index, series_id in enumerate(ids):
                    test = prepared.test[prepared.test["series_id"] == series_id].sort_values("month")
                    for step, (_, actual) in enumerate(test.iterrows()):
                        output.append({"series_id": series_id, "date": f"{actual['month']}-01", "actual": float(actual[prepared.target]), "forecast": float(mean[item_index, step]), "lower_bound": float(quantiles[item_index, step, 0]), "median": float(quantiles[item_index, step, 1]), "upper_bound": float(quantiles[item_index, step, 2]), "target": prepared.target})
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = attempted
        if result.empty:
            detail = errors[-1] if errors else "unknown adapter error"
            raise RuntimeError(f"Chronos failed to process every series; no fallback values were generated: {detail}")
        return result
