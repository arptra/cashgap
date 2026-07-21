from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.features import build_feature_frame, feature_columns
from app.ml.training import train_and_evaluate
from app.model_plugins.base import BaseModelPlugin, PluginRunResult
from app.models_registry.schemas import EnvironmentReport, ModelStatus
from app.services.training import _split_months


class LocalCashGapPlugin(BaseModelPlugin):
    legacy_model_name = "logistic_regression"

    def check_environment(self) -> EnvironmentReport:
        return EnvironmentReport(status=ModelStatus.INSTALLED, message="Локальный CPU classifier готов", installed=True)

    def run(self, dataset: dict[str, Any], options: dict[str, Any]) -> PluginRunResult:
        target_name = options.get("target", "cash_gap_next_month")
        compatibility = self.check_compatibility(dataset, target=target_name, series_level="client", horizon=1, min_history=6)
        if not compatibility.compatible:
            raise ValueError("; ".join(compatibility.reasons))
        paths = dataset.get("paths") or {}
        monthly = pd.read_parquet(paths["monthly_aggregates"])
        target = pd.read_parquet(paths["target"])
        frame = build_feature_frame(monthly, target)
        columns = feature_columns(frame)
        months = sorted(frame["month"].unique().tolist())
        params = options.get("parameters") or {}
        train_months, validation_months, test_months = _split_months(months, float(options.get("train_ratio", 0.6)), float(options.get("validation_ratio", 0.2)))
        masks = {"train": frame["month"].isin(train_months), "validation": frame["month"].isin(validation_months), "test": frame["month"].isin(test_months)}
        x = frame[columns].astype(float).fillna(0.0)
        y = frame["cash_gap_next_month"].to_numpy(dtype=np.int8)
        started = time.perf_counter()
        trained = train_and_evaluate(self.legacy_model_name, params, x.loc[masks["train"]], y[masks["train"].to_numpy()], x.loc[masks["validation"]], y[masks["validation"].to_numpy()], x.loc[masks["test"]], y[masks["test"].to_numpy()], int((dataset.get("config") or {}).get("random_seed", 42)))
        test_frame = frame.loc[masks["test"]]
        predictions = pd.DataFrame({"client_id": test_frame["client_id"].to_numpy(), "scoring_month": test_frame["month"].to_numpy(), "risk_score": trained.predictions, "actual_cash_gap_next_month": test_frame["cash_gap_next_month"].astype(int).to_numpy()})
        predictions["predicted_cash_gap"] = (predictions["risk_score"] >= trained.threshold).astype(int)
        predictions["risk_rank"] = predictions.groupby("scoring_month")["risk_score"].rank(method="first", ascending=False).astype(int)
        trained.metrics["training_seconds"] = float(time.perf_counter() - started)
        self._model = trained.model
        return PluginRunResult(model=trained.model, predictions=predictions, metrics=trained.metrics, parameters={**options, "feature_columns": columns, "split": {"train_months": train_months, "validation_months": validation_months, "test_months": test_months}, "feature_importance": trained.feature_importance})

    def save_artifacts(self, output_dir: Path, result: PluginRunResult) -> dict[str, str]:
        paths = super().save_artifacts(output_dir, result)
        model_path = output_dir / "model.joblib"
        joblib.dump(result.model, model_path)
        paths["model"] = str(model_path)
        return paths

    def load_artifacts(self, output_dir: Path) -> PluginRunResult:
        result = super().load_artifacts(output_dir)
        result.model = joblib.load(output_dir / "model.joblib")
        self._model = result.model
        return result


class LogisticCashGapPlugin(LocalCashGapPlugin):
    legacy_model_name = "logistic_regression"
