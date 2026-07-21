from __future__ import annotations

import json
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.forecasting_data import PreparedForecastData, evaluate_forecast_predictions, prepare_forecast_data
from app.models_registry.compatibility import check_model_compatibility
from app.models_registry.installer import cache_metadata, install_model, uninstall_model
from app.models_registry.schemas import EnvironmentReport, ModelCompatibility, ModelSpec, ModelStatus


@dataclass
class PluginRunResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    model: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)


class BaseModelPlugin(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._model: Any = None
        self._started = 0.0

    def check_environment(self) -> EnvironmentReport:
        return EnvironmentReport(status=ModelStatus.AVAILABLE, message="Модель доступна локально", installed=True)

    def check_compatibility(self, dataset: dict[str, Any], **options: Any) -> ModelCompatibility:
        return check_model_compatibility(self.spec, dataset, **options)

    def install(self, cancel_event=None) -> dict[str, Any]:
        return install_model(self.spec, cancel_event)

    def uninstall(self) -> None:
        uninstall_model(self.spec)

    def prepare_data(self, dataset: dict[str, Any], **options: Any) -> PreparedForecastData:
        path = (dataset.get("paths") or {}).get("monthly_aggregates")
        if not path:
            raise ValueError("Dataset has no canonical monthly table")
        return prepare_forecast_data(Path(path), **options)

    def fit(self, prepared: Any, parameters: dict[str, Any] | None = None) -> Any:
        self._started = time.perf_counter()
        return None

    def predict(self, prepared: Any, parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        raise NotImplementedError(f"predict() is not implemented for {self.spec.id}")

    def evaluate(self, predictions: pd.DataFrame, prepared: PreparedForecastData) -> dict[str, Any]:
        context = prepared.context.assign(target_value=prepared.context[prepared.target])
        return evaluate_forecast_predictions(predictions, context, time.perf_counter() - self._started)

    def save_artifacts(self, output_dir: Path, result: PluginRunResult) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.parquet"
        metrics_path = output_dir / "metrics.json"
        parameters_path = output_dir / "parameters.json"
        result.predictions.to_parquet(predictions_path, index=False)
        metrics_path.write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        parameters_path.write_text(json.dumps(result.parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"predictions": str(predictions_path), "metrics": str(metrics_path), "parameters": str(parameters_path)}

    def load_artifacts(self, output_dir: Path) -> PluginRunResult:
        return PluginRunResult(
            predictions=pd.read_parquet(output_dir / "predictions.parquet"),
            metrics=json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")),
            parameters=json.loads((output_dir / "parameters.json").read_text(encoding="utf-8")),
        )

    def run(self, dataset: dict[str, Any], options: dict[str, Any]) -> PluginRunResult:
        compatibility = self.check_compatibility(dataset, **{key: options[key] for key in ("target", "series_level", "horizon", "min_history")})
        if not compatibility.compatible:
            raise ValueError("; ".join(compatibility.reasons))
        prepared = self.prepare_data(
            dataset,
            target=options["target"],
            series_level=options["series_level"],
            horizon=options["horizon"],
            min_history=options["min_history"],
        )
        self.fit(prepared, options.get("parameters") or {})
        predictions = self.predict(prepared, options.get("parameters") or {})
        metrics = self.evaluate(predictions, prepared)
        return PluginRunResult(predictions=predictions, metrics=metrics, model=self._model, parameters=options)

    def cache_info(self) -> dict[str, Any]:
        return cache_metadata(self.spec)
