from __future__ import annotations

import importlib.util
import time
from typing import Any

import numpy as np
import pandas as pd

from app.model_plugins.base import BaseModelPlugin
from app.models_registry.installer import cache_metadata, model_install_dir
from app.models_registry.schemas import EnvironmentReport, ModelStatus


class TimesFmPlugin(BaseModelPlugin):
    def check_environment(self) -> EnvironmentReport:
        dependency = importlib.util.find_spec("timesfm") is not None and importlib.util.find_spec("torch") is not None
        info = cache_metadata(self.spec)
        if dependency:
            try:
                import timesfm
                compatible = hasattr(timesfm, "TimesFM_2p5_200M_torch")
            except Exception:
                compatible = False
        else:
            compatible = False
        if dependency and compatible and info["installed"]:
            return EnvironmentReport(status=ModelStatus.INSTALLED, message="TimesFM 2.5 PyTorch adapter и weights готовы", installed=True, weights_cached=True, size_bytes=info["size_bytes"], revision=info.get("revision"))
        message = "Установите timesfm[torch]" if not dependency else "Версия timesfm несовместима с TimesFM_2p5_200M_torch" if not compatible else "Скачайте TimesFM weights"
        return EnvironmentReport(status=ModelStatus.NOT_INSTALLED if not compatible else ModelStatus.AVAILABLE, message=message, dependency_installed=compatible, weights_cached=info["installed"], size_bytes=info["size_bytes"], revision=info.get("revision"), install_command=self.spec.install_command)

    def fit(self, prepared, parameters=None):
        import timesfm
        import torch

        if not cache_metadata(self.spec)["installed"]:
            raise RuntimeError("TimesFM weights are not installed")
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise RuntimeError("Installed timesfm package does not expose TimesFM_2p5_200M_torch")
        self._started = time.perf_counter()
        torch.set_float32_matmul_precision("high")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(model_install_dir(self.spec)))
        self._model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=3, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=prepared.target != "net_flow", fix_quantile_crossing=True))
        return self._model

    def predict(self, prepared, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        parameters = parameters or {}
        batch_size = int(parameters.get("batch_size", 8))
        output = []
        attempted = len(prepared.series_ids)
        errors: list[str] = []
        for ids in prepared.batches(batch_size):
            inputs = [prepared.context.loc[prepared.context["series_id"] == series_id, prepared.target].astype(float).to_numpy() for series_id in ids]
            try:
                point, quantile = self._model.forecast(horizon=prepared.horizon, inputs=inputs)
                for index, series_id in enumerate(ids):
                    test = prepared.test[prepared.test["series_id"] == series_id].sort_values("month")
                    for step, (_, actual) in enumerate(test.iterrows()):
                        output.append({"series_id": series_id, "date": f"{actual['month']}-01", "actual": float(actual[prepared.target]), "forecast": float(point[index, step]), "lower_bound": float(quantile[index, step, 1]), "median": float(quantile[index, step, 5]), "upper_bound": float(quantile[index, step, 9]), "target": prepared.target})
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
        result = pd.DataFrame(output)
        result.attrs["attempted_series"] = attempted
        if result.empty:
            detail = errors[-1] if errors else "unknown adapter error"
            raise RuntimeError(f"TimesFM failed to process every series; check the installed package version: {detail}")
        return result
