from __future__ import annotations

from typing import Any

from app.canonical.schemas import assert_valid_cash_gap_target
from app.models_registry.schemas import ModelCompatibility, ModelSpec


TARGET_ALIASES = {
    "credit": "total_credit_sum",
    "debit": "total_debit_sum",
    "net": "net_flow",
    "net_flow": "net_flow",
    "total_credit_sum": "total_credit_sum",
    "total_debit_sum": "total_debit_sum",
    "cash_gap_next_month": "cash_gap_next_month",
}


def normalize_target(target: str) -> str:
    assert_valid_cash_gap_target(target)
    normalized = TARGET_ALIASES.get(target)
    if normalized is None:
        raise ValueError(f"Unsupported target: {target}")
    return normalized


def check_model_compatibility(
    spec: ModelSpec,
    dataset: dict[str, Any],
    *,
    target: str,
    series_level: str = "client",
    horizon: int = 1,
    min_history: int = 6,
) -> ModelCompatibility:
    reasons: list[str] = []
    try:
        target = normalize_target(target)
    except ValueError as exc:
        return ModelCompatibility(compatible=False, reasons=[str(exc)], target=target, task=spec.task)
    summary = dataset.get("summary") or {}
    paths = dataset.get("paths") or {}
    if dataset.get("status") != "completed":
        reasons.append("Dataset ещё не готов")
    if target not in spec.compatible_targets:
        reasons.append(f"Модель не поддерживает target {target}")
    if spec.task == "cash_gap_classification":
        if target != "cash_gap_next_month":
            reasons.append("Cash-gap classifier требует target cash_gap_next_month")
        if not summary.get("has_cash_gap_target") or not paths.get("target"):
            reasons.append("Для обучения классификатора необходим исторический признак кассового разрыва")
        estimated_series = int(summary.get("clients") or summary.get("n_clients") or 0)
    else:
        if target == "cash_gap_next_month":
            reasons.append("Forecasting model прогнозирует credit, debit или net_flow")
        if not paths.get("monthly_aggregates") and not paths.get("forecast_series"):
            reasons.append("Нужна normalized месячная таблица денежных потоков")
        months = int(summary.get("months") or summary.get("n_months") or 0)
        if months < min_history + horizon:
            reasons.append(f"Нужно минимум {min_history + horizon} месяцев истории")
        clients = int(summary.get("clients") or summary.get("n_clients") or 1)
        labels = int(summary.get("labels") or 1)
        estimated_series = clients * labels if series_level == "client_category" else clients
    memory_mb = round(max(estimated_series, 1) * max(min_history + horizon, 1) * 8 * 6 / 1024**2, 2)
    return ModelCompatibility(
        compatible=not reasons,
        reasons=reasons,
        target=target,
        task=spec.task,
        estimated_series=estimated_series,
        estimated_memory_mb=memory_mb,
        requires_training=spec.requires_training,
        requires_target=spec.task == "cash_gap_classification",
        details={"series_level": series_level, "horizon": horizon, "min_history": min_history},
    )
