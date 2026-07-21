from app.model_plugins.logistic_cashgap import LocalCashGapPlugin


class RandomForestCashGapPlugin(LocalCashGapPlugin):
    legacy_model_name = "random_forest"
