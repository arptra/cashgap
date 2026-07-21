import numpy as np

from app.ml.evaluation import classification_metrics, select_threshold


def test_threshold_and_metrics() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.30, 0.55, 0.75, 0.95])
    threshold = select_threshold(y_true, probabilities)
    metrics = classification_metrics(y_true, probabilities, threshold, training_seconds=0.25)
    assert 0 <= threshold <= 1
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]
    assert metrics["training_seconds"] == 0.25

