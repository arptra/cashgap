from __future__ import annotations

import numpy as np
import pandas as pd


REASON_LABELS = {
    "credit_decline": "снижение поступлений",
    "debit_growth": "рост расходов",
    "negative_flow": "отрицательный денежный поток",
    "negative_streak": "несколько отрицательных месяцев подряд",
    "credit_count_decline": "снижение количества входящих операций",
    "average_debit_growth": "рост среднего исходящего платежа",
    "credit_volatility": "высокая волатильность поступлений",
}


def _safe_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame))
    return np.nan_to_num(frame[name].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


REASON_FEATURE_PATTERNS = {
    "credit_decline": ("total_credit_change", "credit_trend", "credit_sum__"),
    "debit_growth": ("total_debit_change", "debit_trend", "debit_sum__"),
    "negative_flow": ("net_flow",),
    "negative_streak": ("negative_flow_months", "negative_flow_streak"),
    "credit_count_decline": ("credit_count_change", "credit_count_decline"),
    "average_debit_growth": ("avg_debit_operation_change",),
    "credit_volatility": ("credit_volatility", "total_credit_std"),
}


def model_contributions(model_name: str, model, frame: pd.DataFrame) -> np.ndarray | None:
    """Return signed per-row feature contributions for explainable model families."""

    if model_name == "catboost":
        from catboost import Pool

        values = model.get_feature_importance(
            Pool(frame), type="ShapValues", shap_calc_type="Approximate"
        )
        return np.asarray(values)[:, :-1]
    if model_name == "lightgbm":
        values = model.predict(frame, pred_contrib=True)
        return np.asarray(values)[:, :-1]
    if model_name == "logistic_regression":
        transformed = model.named_steps["scale"].transform(frame)
        coefficients = model.named_steps["model"].coef_[0]
        return np.asarray(transformed) * coefficients
    return None


def explain_rows(
    frame: pd.DataFrame,
    contributions: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Translate point-in-time severity and signed model contributions into reasons."""

    scale = np.maximum(np.abs(_safe_column(frame, "total_credit_mean_6")), 1.0)
    severities = np.column_stack(
        [
            np.maximum(-_safe_column(frame, "total_credit_change_pct"), 0),
            np.maximum(_safe_column(frame, "total_debit_change_pct"), 0),
            np.maximum(-_safe_column(frame, "net_flow") / scale, 0),
            _safe_column(frame, "negative_flow_streak") / 3.0,
            np.maximum(-_safe_column(frame, "credit_count_change_pct"), 0),
            np.maximum(_safe_column(frame, "avg_debit_operation_change_pct"), 0),
            _safe_column(frame, "credit_volatility_6"),
        ]
    )
    reason_keys = list(REASON_LABELS)
    if contributions is not None and feature_names is not None:
        contribution_scores = np.zeros_like(severities)
        positive = np.maximum(np.nan_to_num(contributions, nan=0.0), 0.0)
        for reason_index, reason_key in enumerate(reason_keys):
            patterns = REASON_FEATURE_PATTERNS[reason_key]
            indices = [
                index for index, name in enumerate(feature_names)
                if any(pattern in name for pattern in patterns)
            ]
            if indices:
                contribution_scores[:, reason_index] = positive[:, indices].sum(axis=1)
        row_max = contribution_scores.max(axis=1, keepdims=True)
        contribution_scores = contribution_scores / np.where(row_max > 0, row_max, 1.0)
        severities = severities + contribution_scores

    labels = [REASON_LABELS[key] for key in reason_keys]
    order = np.argsort(-severities, axis=1)
    rows: list[list[str]] = []
    for row_index, indices in enumerate(order):
        chosen = [labels[index] for index in indices if severities[row_index, index] > 0][:3]
        fallback = ["нет выраженного негативного сигнала", "стабильная динамика", "низкая текущая волатильность"]
        chosen.extend(fallback[: 3 - len(chosen)])
        rows.append(chosen)
    return pd.DataFrame(rows, columns=["top_reason_1", "top_reason_2", "top_reason_3"], index=frame.index)
