from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class CategorizationResult:
    model: Pipeline
    predictions: pd.DataFrame
    metrics: dict[str, float | int]


def train_categorizer(frame: pd.DataFrame, params: dict[str, Any] | None = None) -> CategorizationResult:
    params = params or {}
    required = {"transaction_description", "category"}
    if not required <= set(frame.columns):
        raise ValueError("Categorization data needs transaction_description and category")
    clean = frame.dropna(subset=list(required)).copy()
    if len(clean) < 20 or clean["category"].nunique() < 2:
        raise ValueError("Categorization requires at least 20 rows and two categories")
    train, test = train_test_split(
        clean,
        test_size=0.2,
        random_state=42,
        stratify=clean["category"] if clean["category"].value_counts().min() >= 2 else None,
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=int(params.get("max_features", 30_000)))),
            ("model", LogisticRegression(max_iter=int(params.get("max_iter", 500)), class_weight="balanced")),
        ]
    )
    started = time.perf_counter()
    model.fit(train["transaction_description"].astype(str), train["category"].astype(str))
    seconds = time.perf_counter() - started
    predicted = model.predict(test["transaction_description"].astype(str))
    predictions = pd.DataFrame(
        {
            "transaction_description": test["transaction_description"].astype(str).to_numpy(),
            "actual_category": test["category"].astype(str).to_numpy(),
            "predicted_category": predicted,
        }
    )
    return CategorizationResult(
        model=model,
        predictions=predictions,
        metrics={
            "accuracy": float(accuracy_score(test["category"].astype(str), predicted)),
            "f1_macro": float(f1_score(test["category"].astype(str), predicted, average="macro")),
            "training_seconds": float(seconds),
            "test_rows": int(len(test)),
        },
    )

