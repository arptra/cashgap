from __future__ import annotations

from app.model_plugins.base import BaseModelPlugin
from app.model_plugins.catboost_cashgap import CatBoostCashGapPlugin
from app.model_plugins.chronos_2 import Chronos2Plugin
from app.model_plugins.chronos_bolt import ChronosBoltPlugin
from app.model_plugins.lightgbm_cashgap import LightGbmCashGapPlugin
from app.model_plugins.logistic_cashgap import LogisticCashGapPlugin
from app.model_plugins.random_forest_cashgap import RandomForestCashGapPlugin
from app.model_plugins.seasonal_naive import SeasonalNaivePlugin
from app.model_plugins.shell_catboost_darts import ShellCatboostDartsPlugin
from app.model_plugins.shell_prophet import ShellProphetPlugin
from app.model_plugins.timesfm import TimesFmPlugin
from app.models_registry.schemas import ModelSpec


PLUGINS: dict[str, type[BaseModelPlugin]] = {
    "shell_catboost_darts": ShellCatboostDartsPlugin,
    "shell_prophet": ShellProphetPlugin,
    "chronos_bolt": ChronosBoltPlugin,
    "chronos_2": Chronos2Plugin,
    "timesfm": TimesFmPlugin,
    "seasonal_naive": SeasonalNaivePlugin,
    "logistic_cashgap": LogisticCashGapPlugin,
    "catboost_cashgap": CatBoostCashGapPlugin,
    "lightgbm_cashgap": LightGbmCashGapPlugin,
    "random_forest_cashgap": RandomForestCashGapPlugin,
}


def create_model_plugin(spec: ModelSpec) -> BaseModelPlugin:
    plugin_class = PLUGINS.get(spec.plugin)
    if plugin_class is None:
        raise ValueError(f"Unknown model plugin: {spec.plugin}")
    return plugin_class(spec)


__all__ = ["BaseModelPlugin", "create_model_plugin"]
