#!/usr/bin/env python3
"""Русскоязычные бизнес-отчёты для месячного cash-flow benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_NAMES_RU: Dict[str, str] = {
    "trailing_mean": "Прогноз по среднему",
    "linear_regression": "Линейная регрессия",
    "gradient_boosting": "Градиентный бустинг",
    "torch_mlp_2_layers": "Полносвязная сеть — 2 слоя",
    "torch_mlp_3_layers": "Полносвязная сеть — 3 слоя",
    "torch_mlp_tuned": "Полносвязная сеть — настроенная",
}

FLOW_NAMES_RU = {
    "inflow_credit": "Зачисления",
    "outflow_debit": "Списания",
}


def model_name_ru(model_id: str) -> str:
    return MODEL_NAMES_RU.get(str(model_id), str(model_id))


def _format_month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("test_month", "train_start", "train_end", "validation_month"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m")
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # UTF-8 BOM and semicolon make the report open correctly in Russian Excel.
    frame.to_csv(path, index=False, sep=";", encoding="utf-8-sig", float_format="%.4f")


def russian_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = _format_month_columns(metrics)
    result.insert(3, "название_модели", result["model"].map(model_name_ru))
    result["flow"] = result["flow"].map(FLOW_NAMES_RU).fillna(result["flow"])
    columns = {
        "fold": "номер_тестового_периода",
        "test_month": "тестовый_месяц",
        "model": "код_модели",
        "flow": "денежный_поток",
        "training_seconds": "время_обучения_сек",
        "aggregate_mape": "MAPE_совокупный",
        "aggregate_mape_percent": "MAPE_совокупный_процент",
        "company_mape_nonzero": "MAPE_по_компаниям",
        "company_mape_nonzero_percent": "MAPE_по_компаниям_процент",
        "wape": "WAPE",
        "wape_percent": "WAPE_процент",
        "mae": "MAE_рублей",
        "bias_percent": "смещение_прогноза_процент",
        "actual_total": "фактическая_сумма",
        "predicted_total": "прогнозная_сумма",
        "nonzero_companies": "компаний_с_ненулевым_фактом",
        "companies": "компаний_всего",
    }
    return result.rename(columns=columns)


def russian_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result.insert(1, "название_модели", result["model"].map(model_name_ru))
    result["flow"] = result["flow"].map(FLOW_NAMES_RU).fillna(result["flow"])
    return result.rename(columns={
        "model": "код_модели",
        "flow": "денежный_поток",
        "aggregate_mape_mean_percent": "MAPE_средний_процент",
        "aggregate_mape_std_percent": "MAPE_стандартное_отклонение_процент",
        "aggregate_mape_worst_percent": "MAPE_худший_месяц_процент",
        "wape_mean_percent": "WAPE_средний_процент",
        "company_mape_mean_percent": "MAPE_по_компаниям_средний_процент",
        "bias_mean_percent": "среднее_смещение_процент",
        "folds": "количество_тестовых_периодов",
    })


def russian_windows(windows: pd.DataFrame) -> pd.DataFrame:
    result = _format_month_columns(windows)
    return result.rename(columns={
        "fold": "номер_тестового_периода",
        "train_start": "обучение_с",
        "train_end": "обучение_по",
        "validation_month": "месяц_валидации",
        "test_month": "тестовый_месяц",
        "train_rows": "строк_обучения",
        "validation_rows": "строк_валидации",
        "test_rows": "строк_теста",
    })


def _write_russian_predictions(source: Path, destination: Path) -> None:
    parquet = pq.ParquetFile(source)
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            frame = batch.to_pandas()
            frame["model_name_ru"] = frame["model"].map(model_name_ru)
            frame = frame.rename(columns={
                "fold": "номер_тестового_периода",
                "test_month": "тестовый_месяц",
                "inn": "инн",
                "model": "код_модели",
                "model_name_ru": "название_модели",
                "actual_inflow": "фактические_зачисления",
                "predicted_inflow": "прогноз_зачислений",
                "actual_outflow": "фактические_списания",
                "predicted_outflow": "прогноз_списаний",
            })
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _business_conclusion(summary: pd.DataFrame, periods: int) -> str:
    lines = [
        "БИЗНЕС-ВЫВОД ПО СТАБИЛЬНОСТИ МОДЕЛЕЙ",
        "Тестовых периодов: {}".format(periods),
        "",
        "MAPE совокупного потока показывает ошибку общей суммы по всем компаниям за месяц.",
        "Чем меньше значение, тем точнее прогноз.",
        "",
    ]
    for flow_id, flow_name in FLOW_NAMES_RU.items():
        candidates = summary[summary["flow"].eq(flow_id)].sort_values(
            "aggregate_mape_mean_percent"
        )
        if candidates.empty:
            continue
        best = candidates.iloc[0]
        lines.extend([
            "{}:".format(flow_name),
            "  Лучшая модель: {} ({})".format(model_name_ru(best["model"]), best["model"]),
            "  Средний MAPE: {:.2f}%".format(best["aggregate_mape_mean_percent"]),
            "  Худший месяц: {:.2f}%".format(best["aggregate_mape_worst_percent"]),
            "  Среднее смещение: {:+.2f}%".format(best["bias_mean_percent"]),
            "",
        ])
    lines.extend([
        "Важно: прогноз описывает зачисления и списания. Для настоящего кассового разрыва",
        "нужны начальные остатки на счетах и обязательные будущие платежи.",
    ])
    return "\n".join(lines)


def write_russian_reports(
    output: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    predictions_path: Path,
) -> None:
    """Write human-facing Russian artifacts while keeping technical files compatible."""
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(russian_summary(summary), output / "отчет_стабильность_моделей.csv")
    _write_csv(russian_metrics(metrics), output / "отчет_метрики_по_периодам.csv")
    _write_csv(russian_windows(windows), output / "отчет_окна_тестирования.csv")
    _write_russian_predictions(predictions_path, output / "отчет_прогнозы_по_инн.parquet")
    periods = int(windows["fold"].nunique()) if not windows.empty else 0
    (output / "бизнес_вывод.txt").write_text(
        _business_conclusion(summary, periods), encoding="utf-8"
    )
