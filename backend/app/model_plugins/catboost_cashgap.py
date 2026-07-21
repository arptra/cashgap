from app.model_plugins.logistic_cashgap import LocalCashGapPlugin


class CatBoostCashGapPlugin(LocalCashGapPlugin):
    legacy_model_name = "catboost"
