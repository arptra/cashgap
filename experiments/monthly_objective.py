#!/usr/bin/env python3
"""Shared monetary target and metrics for monthly cash-flow models.

The business KPI is the error of monthly totals in rubles. Models therefore
learn a linearly scaled ruble correction to the transparent three-month-mean
baseline, not log1p(amount): a log target optimizes a different, per-company
objective and systematically underestimates a skewed monetary sum. A zero
correction exactly reproduces the baseline, so an untrained network cannot emit
an arbitrary monetary forecast.

This module intentionally depends only on NumPy so the isolated CUDA worker can
import it without loading pandas, PyArrow, sklearn or scipy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


MONTHLY_OBJECTIVE_VERSION = 2
MONTHLY_OBJECTIVE_NAME = "baseline_residual_rubles_rms_scaled_mse_v2"

BASELINE_FEATURES: Tuple[str, str] = (
    "target_inflow_mean_3",
    "target_outflow_mean_3",
)


def validate_money_targets(values: np.ndarray, name: str = "targets") -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("{} должен иметь форму (строки, 2); получено {}.".format(
            name, result.shape
        ))
    if len(result) == 0:
        raise ValueError("{} пуст.".format(name))
    if not np.isfinite(result).all():
        raise ValueError("{} содержит NaN или infinity.".format(name))
    if (result < 0).any():
        minimum = float(result.min())
        raise ValueError(
            "{} содержит отрицательные зачисления/списания (минимум {}). "
            "Проверьте знак tr_sum до обучения.".format(name, minimum)
        )
    return result


def fit_money_scale(values: np.ndarray) -> np.ndarray:
    """Return one stable linear scale per flow while preserving ruble geometry."""
    targets = validate_money_targets(values)
    scale = np.sqrt(np.mean(np.square(targets), axis=0, dtype=np.float64))
    return np.maximum(scale, 1.0).astype(np.float32)


def baseline_from_feature_matrix(
    values: np.ndarray, feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("Матрица признаков должна быть двумерной.")
    missing = [name for name in BASELINE_FEATURES if name not in feature_names]
    if missing:
        raise ValueError("Нет признаков baseline: {}.".format(missing))
    indices = [list(feature_names).index(name) for name in BASELINE_FEATURES]
    baseline = np.asarray(matrix[:, indices], dtype=np.float64)
    return np.maximum(np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def fit_feature_normalizer(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Признаки train должны быть конечной непустой матрицей.")
    mean = matrix.mean(axis=0, dtype=np.float64)
    raw_scale = matrix.std(axis=0, dtype=np.float64)
    active = raw_scale > 1e-6
    scale = np.where(active, raw_scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32), active.astype(bool)


def normalize_features(
    values: np.ndarray,
    mean: Sequence[float],
    scale: Sequence[float],
    active: Sequence[bool],
    clip: float = 10.0,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    divisor = np.asarray(scale, dtype=np.float64)
    active_mask = np.asarray(active, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != len(center):
        raise ValueError("Размер признаков не совпадает с сохранённым normalizer.")
    if len(center) != len(divisor) or len(center) != len(active_mask):
        raise ValueError("Некорректные параметры normalizer.")
    normalized = (matrix - center) / divisor
    # A feature constant in train is unknown to the model. Letting a future
    # lag suddenly enter as millions (scale=1 in StandardScaler) creates
    # catastrophic extrapolation, so such coordinates remain exactly zero.
    normalized[:, ~active_mask] = 0.0
    normalized = np.clip(normalized, -abs(float(clip)), abs(float(clip)))
    if not np.isfinite(normalized).all():
        raise ValueError("После нормализации признаков появились NaN или infinity.")
    return normalized.astype(np.float32)


def fit_residual_scale(values: np.ndarray) -> np.ndarray:
    residuals = np.asarray(values, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] != 2 or len(residuals) == 0:
        raise ValueError("Остатки должны иметь форму (строки, 2).")
    if not np.isfinite(residuals).all():
        raise ValueError("Остатки содержат NaN или infinity.")
    scale = np.sqrt(np.mean(np.square(residuals), axis=0, dtype=np.float64))
    return np.maximum(scale, 1.0).astype(np.float32)


def scale_residual_targets(values: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    residuals = np.asarray(values, dtype=np.float64)
    divisor = np.asarray(scale, dtype=np.float64).reshape(1, 2)
    if residuals.ndim != 2 or residuals.shape[1] != 2:
        raise ValueError("Остатки должны иметь форму (строки, 2).")
    if not np.isfinite(residuals).all() or not np.isfinite(divisor).all() or (divisor <= 0).any():
        raise ValueError("Некорректные остатки или их масштаб.")
    return (residuals / divisor).astype(np.float32)


def restore_residual_predictions(
    baseline: np.ndarray, scaled_residual: np.ndarray, scale: Sequence[float],
) -> np.ndarray:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    correction = np.asarray(scaled_residual, dtype=np.float64)
    if baseline_values.shape != correction.shape or correction.ndim != 2 or correction.shape[1] != 2:
        raise ValueError("Baseline и поправка должны иметь одинаковую форму (строки, 2).")
    restored = baseline_values + correction * np.asarray(scale, dtype=np.float64).reshape(1, 2)
    return np.maximum(restored, 0.0)


def scale_money_targets(values: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    targets = validate_money_targets(values)
    divisor = np.asarray(scale, dtype=np.float64).reshape(1, 2)
    if not np.isfinite(divisor).all() or (divisor <= 0).any():
        raise ValueError("Некорректный масштаб денежных таргетов: {}.".format(divisor.tolist()))
    return (targets / divisor).astype(np.float32)


def restore_money_predictions(values: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    predictions = np.asarray(values, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Прогноз должен иметь форму (строки, 2); получено {}.".format(
            predictions.shape
        ))
    restored = predictions * np.asarray(scale, dtype=np.float64).reshape(1, 2)
    # Negative credits/debits have no business meaning. Linear models may emit
    # them outside the training range, so all model families use the same rule.
    return np.maximum(restored, 0.0)


def monthly_regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, zero_floor: float,
) -> List[Dict[str, float]]:
    """Metrics for aggregate liquidity and robust per-company diagnostics."""
    truth_matrix = validate_money_targets(actual, "actual")
    forecast_matrix = np.asarray(predicted, dtype=np.float64)
    if forecast_matrix.shape != truth_matrix.shape:
        raise ValueError(
            "Размер прогноза {} не совпадает с фактом {}.".format(
                forecast_matrix.shape, truth_matrix.shape
            )
        )
    if not np.isfinite(forecast_matrix).all():
        raise ValueError("Прогноз содержит NaN или infinity.")
    if zero_floor <= 0:
        raise ValueError("zero_floor должен быть положительным.")

    rows: List[Dict[str, float]] = []
    for index, flow in enumerate(("inflow_credit", "outflow_debit")):
        truth = truth_matrix[:, index]
        forecast = np.maximum(forecast_matrix[:, index], 0.0)
        nonzero = np.abs(truth) > zero_floor
        total_truth = float(truth.sum())
        total_forecast = float(forecast.sum())
        aggregate_ape = abs(total_forecast - total_truth) / max(abs(total_truth), zero_floor)
        absolute_errors = np.abs(forecast - truth)
        wape = absolute_errors.sum() / max(np.abs(truth).sum(), zero_floor)
        company_median_ape = (
            float(np.median(absolute_errors[nonzero] / np.abs(truth[nonzero])))
            if nonzero.any() else float("nan")
        )
        rows.append({
            "flow": flow,
            # Kept for compatibility: one fold contains one test month, so this
            # is APE of that monthly total; its mean over folds is monthly MAPE.
            "aggregate_mape": float(aggregate_ape),
            "aggregate_mape_percent": float(aggregate_ape * 100),
            "company_median_ape": company_median_ape,
            "company_median_ape_percent": float(company_median_ape * 100),
            "wape": float(wape),
            "wape_percent": float(wape * 100),
            "mae": float(np.mean(absolute_errors)),
            "bias_percent": float(
                (total_forecast - total_truth) / max(abs(total_truth), zero_floor) * 100
            ),
            "actual_total": total_truth,
            "predicted_total": total_forecast,
            "nonzero_companies": int(nonzero.sum()),
            "companies": int(len(truth)),
        })
    return rows


def target_total_diagnostics(values: np.ndarray) -> Dict[str, object]:
    targets = validate_money_targets(values)
    return {
        "rows": int(len(targets)),
        "inflow_total": float(targets[:, 0].sum()),
        "outflow_total": float(targets[:, 1].sum()),
        "inflow_zero_share_percent": float(np.mean(targets[:, 0] == 0) * 100),
        "outflow_zero_share_percent": float(np.mean(targets[:, 1] == 0) * 100),
        "inflow_median_nonzero": float(
            np.median(targets[targets[:, 0] > 0, 0]) if np.any(targets[:, 0] > 0) else 0.0
        ),
        "outflow_median_nonzero": float(
            np.median(targets[targets[:, 1] > 0, 1]) if np.any(targets[:, 1] > 0) else 0.0
        ),
    }
