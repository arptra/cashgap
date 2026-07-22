from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.evaluation import classification_metrics, select_threshold


@dataclass
class TrainingResult:
    model: Any
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    predictions: np.ndarray
    threshold: float
    training_seconds: float
    effective_params: dict[str, Any]


def _filtered(params: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key in allowed}


def create_model(model_name: str, params: dict[str, Any], random_seed: int = 42) -> tuple[Any, dict[str, Any]]:
    if model_name == "dummy":
        effective = {"strategy": "prior", **_filtered(params, {"strategy"})}
        return DummyClassifier(**effective), effective
    if model_name == "logistic_regression":
        options = {
            "C": 1.0,
            "max_iter": 500,
            "class_weight": "balanced",
            "random_state": random_seed,
            **_filtered(params, {"C", "max_iter", "class_weight"}),
        }
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(**options))]), options
    if model_name == "random_forest":
        effective = {
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_seed,
            **_filtered(params, {"n_estimators", "max_depth", "min_samples_leaf", "class_weight"}),
        }
        return RandomForestClassifier(**effective), effective
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("CatBoost is not installed. Run make setup.") from exc
        effective = {
            "iterations": 350,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "verbose": False,
            "thread_count": 2,
            "random_seed": random_seed,
            **_filtered(params, {"iterations", "depth", "learning_rate", "l2_leaf_reg"}),
        }
        return CatBoostClassifier(**effective), effective
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is optional and is not installed. Install backend/requirements-lightgbm.txt if your package index allows it."
            ) from exc
        effective = {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "class_weight": "balanced",
            "n_jobs": 2,
            "verbosity": -1,
            "random_state": random_seed,
            **_filtered(params, {"n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples"}),
        }
        return LGBMClassifier(**effective), effective
    raise ValueError(f"Unknown model: {model_name}")


def _importance(model_name: str, model: Any, columns: list[str]) -> list[dict[str, Any]]:
    if model_name == "logistic_regression":
        values = np.abs(model.named_steps["model"].coef_[0])
    elif model_name == "dummy":
        values = np.zeros(len(columns))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance())
    else:
        values = np.zeros(len(columns))
    order = np.argsort(-values)
    return [
        {"feature": columns[index], "importance": float(values[index])}
        for index in order[:50]
    ]


def train_and_evaluate(
    model_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    random_seed: int = 42,
) -> TrainingResult:
    model, effective_params = create_model(model_name, params, random_seed)
    started = time.perf_counter()
    if model_name == "catboost":
        model.fit(x_train, y_train, eval_set=(x_validation, y_validation), early_stopping_rounds=40)
    elif model_name == "lightgbm":
        try:
            from lightgbm import early_stopping

            model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                eval_metric="average_precision",
                callbacks=[early_stopping(40, verbose=False)],
            )
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probabilities, threshold, training_seconds)
    importance = _importance(model_name, model, list(x_train.columns))
    return TrainingResult(
        model=model,
        metrics=metrics,
        feature_importance=importance,
        predictions=test_probabilities,
        threshold=threshold,
        training_seconds=training_seconds,
        effective_params=effective_params,
    )
