from app.model_plugins.logistic_cashgap import LocalCashGapPlugin


class LightGbmCashGapPlugin(LocalCashGapPlugin):
    legacy_model_name = "lightgbm"
