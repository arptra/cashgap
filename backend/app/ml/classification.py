"""Cash-gap classification public module.

The implementation lives in training.py for backwards compatibility with the
first CashGap Lab release.
"""

from app.ml.training import TrainingResult, create_model, train_and_evaluate

__all__ = ["TrainingResult", "create_model", "train_and_evaluate"]

