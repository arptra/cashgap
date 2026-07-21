from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(scores))])


def _safe_auc(metric, y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(metric(y_true, probabilities))


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    training_seconds: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    top_count = max(1, int(np.ceil(len(y_true) * 0.10)))
    top_indices = np.argsort(-probabilities)[:top_count]
    positives = int(y_true.sum())
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "pr_auc": _safe_auc(average_precision_score, y_true, probabilities),
        "roc_auc": _safe_auc(roc_auc_score, y_true, probabilities),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision_at_top_10": float(y_true[top_indices].mean()),
        "recall_at_top_10": float(y_true[top_indices].sum() / positives) if positives else 0.0,
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
        "training_seconds": float(training_seconds),
        "test_rows": int(len(y_true)),
        "test_positive_rate": float(y_true.mean()),
    }

