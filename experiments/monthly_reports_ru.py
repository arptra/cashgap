#!/usr/bin/env python3
"""Русскоязычные бизнес-отчёты для месячного cash-flow benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_NAMES_RU: Dict[str, str] = {
    "trailing_mean": "Прогноз по среднему",
    "linear_regression": "Линейная регрессия",
    "gradient_boosting": "Градиентный бустинг",
    "torch_mlp_2_layers": "Полносвязная сеть — вариант A",
    "torch_mlp_3_layers": "Полносвязная сеть — вариант B",
    "torch_mlp_tuned": "Полносвязная сеть — настроенная",
}

FLOW_NAMES_RU = {
    "inflow_credit": "Зачисления",
    "outflow_debit": "Списания",
}


def model_name_ru(model_id: str) -> str:
    return MODEL_NAMES_RU.get(str(model_id), str(model_id))


def model_display_ru(row: pd.Series) -> str:
    name = model_name_ru(str(row.get("model", "")))
    architecture = row.get("architecture")
    if architecture is None or pd.isna(architecture) or not str(architecture).strip():
        return name
    return "{} ({})".format(name, architecture)


def _month_values(values: pd.Series) -> pd.Series:
    """Format Timestamp and compact YYYYMM values without turning them into 1970 dates."""
    text = values.astype("string").str.strip()
    compact = text.str.extract(r"(^|\D)(\d{6})(?:\.0)?($|\D)", expand=True)[1]
    parsed_compact = pd.to_datetime(compact, format="%Y%m", errors="coerce")
    parsed_regular = pd.to_datetime(values, errors="coerce")
    parsed = parsed_compact.fillna(parsed_regular)
    formatted = parsed.dt.strftime("%Y-%m")
    return formatted.fillna(text)


def _format_month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("test_month", "train_start", "train_end", "validation_month"):
        if column in result.columns:
            result[column] = _month_values(result[column])
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # BOM + semicolon + decimal comma are understood by Russian Excel without import wizard.
    frame.to_csv(
        path, index=False, sep=";", decimal=",", encoding="utf-8-sig", float_format="%.2f"
    )


def _stability_label(mean_value: object, std_value: object) -> str:
    try:
        mean = abs(float(mean_value))
        deviation = abs(float(std_value))
    except (TypeError, ValueError):
        return "Недостаточно данных"
    if pd.isna(mean) or pd.isna(deviation):
        return "Недостаточно данных"
    ratio = deviation / max(mean, 1e-9)
    if ratio <= 0.25:
        return "Стабильная"
    if ratio <= 0.50:
        return "Умеренно стабильная"
    return "Нестабильная"


def russian_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = _format_month_columns(metrics)
    business = pd.DataFrame({
        "Номер тестового периода": result["fold"],
        "Тестовый месяц": result["test_month"],
        "Денежный поток": result["flow"].map(FLOW_NAMES_RU).fillna(result["flow"]),
        "Модель": [model_display_ru(row) for _, row in result.iterrows()],
        "Архитектура/метод": result.get(
            "architecture", pd.Series(["Не записана"] * len(result), index=result.index)
        ),
        "Фактическая сумма, руб.": result["actual_total"],
        "Прогнозная сумма, руб.": result["predicted_total"],
        "Отклонение прогноза, руб.": result["predicted_total"] - result["actual_total"],
        "Абсолютная ошибка, руб.": (result["predicted_total"] - result["actual_total"]).abs(),
        "Ошибка месячного итога, %": result["aggregate_mape_percent"],
        "Ошибка прогноза отдельных компаний, %": result["wape_percent"],
        "Завышение (+) или занижение (-), %": result["bias_percent"],
        "Компаний в тесте": result["companies"],
        "Компаний с ненулевым фактом": result["nonzero_companies"],
    })
    if "training_seconds" in result.columns:
        business["Время обучения, сек."] = result["training_seconds"]
    return business.sort_values(
        ["Тестовый месяц", "Денежный поток", "Ошибка месячного итога, %", "Модель"]
    ).reset_index(drop=True)


def russian_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.sort_values(["flow", "aggregate_mape_mean_percent"]).copy()
    result["rank"] = result.groupby("flow").cumcount() + 1
    business = pd.DataFrame({
        "Место": result["rank"],
        "Денежный поток": result["flow"].map(FLOW_NAMES_RU).fillna(result["flow"]),
        "Модель": [model_display_ru(row) for _, row in result.iterrows()],
        "Архитектура/метод": result.get(
            "architecture", pd.Series(["Не записана"] * len(result), index=result.index)
        ),
        "Средняя ошибка месячного итога, %": result["aggregate_mape_mean_percent"],
        "Насколько ошибка скачет по месяцам, п.п.": result["aggregate_mape_std_percent"],
        "Ошибка в самом плохом месяце, %": result["aggregate_mape_worst_percent"],
        "Ошибка прогноза отдельных компаний, %": result["wape_mean_percent"],
        "Завышение (+) или занижение (-), %": result["bias_mean_percent"],
        "Количество тестовых месяцев": result["folds"],
    })
    business["Оценка стабильности"] = [
        _stability_label(mean, deviation)
        for mean, deviation in zip(
            result["aggregate_mape_mean_percent"], result["aggregate_mape_std_percent"]
        )
    ]
    business["Результат"] = business["Место"].map(
        lambda value: "Лучшая модель" if int(value) == 1 else "Альтернатива"
    )
    return business.reset_index(drop=True)


def russian_windows(windows: pd.DataFrame) -> pd.DataFrame:
    result = _format_month_columns(windows)
    return result.rename(columns={
        "fold": "Номер тестового периода",
        "train_start": "Начало истории обучения",
        "train_end": "Конец истории обучения",
        "validation_month": "Месяц валидации",
        "test_month": "Тестовый месяц",
        "train_rows": "Строк в обучении",
        "validation_rows": "Строк в валидации",
        "test_rows": "Строк в тесте",
    })


def russian_net_flow(metrics: pd.DataFrame) -> pd.DataFrame:
    result = _format_month_columns(metrics)
    keys = ["fold", "test_month", "model"]
    actual = result.pivot_table(
        index=keys, columns="flow", values="actual_total", aggfunc="first"
    )
    predicted = result.pivot_table(
        index=keys, columns="flow", values="predicted_total", aggfunc="first"
    )
    required = set(FLOW_NAMES_RU)
    if not required.issubset(actual.columns) or not required.issubset(predicted.columns):
        return pd.DataFrame()
    actual_net = actual["inflow_credit"] - actual["outflow_debit"]
    predicted_net = predicted["inflow_credit"] - predicted["outflow_debit"]
    business = pd.DataFrame({
        "Номер тестового периода": actual.index.get_level_values("fold"),
        "Тестовый месяц": actual.index.get_level_values("test_month"),
        "Модель": actual.index.get_level_values("model").map(model_name_ru),
        "Фактический чистый поток, руб.": actual_net.to_numpy(),
        "Прогнозный чистый поток, руб.": predicted_net.to_numpy(),
        "Отклонение прогноза, руб.": (predicted_net - actual_net).to_numpy(),
        "Фактический поток отрицательный": actual_net.lt(0).map(
            {True: "Да", False: "Нет"}
        ).to_numpy(),
        "Прогноз отрицательного потока": predicted_net.lt(0).map(
            {True: "Да", False: "Нет"}
        ).to_numpy(),
        "Знак потока определён верно": actual_net.lt(0).eq(predicted_net.lt(0)).map(
            {True: "Да", False: "Нет"}
        ).to_numpy(),
    })
    return business.sort_values(["Тестовый месяц", "Модель"]).reset_index(drop=True)


def _write_russian_predictions(source: Path, destination: Path) -> None:
    parquet = pq.ParquetFile(source)
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            frame = batch.to_pandas()
            business = pd.DataFrame({
                "Номер тестового периода": frame["fold"],
                "Тестовый месяц": _month_values(frame["test_month"]),
                "ИНН": frame["inn"].astype("string"),
                "Модель": frame["model"].map(model_name_ru),
                "Фактические зачисления, руб.": frame["actual_inflow"],
                "Прогноз зачислений, руб.": frame["predicted_inflow"],
                "Ошибка зачислений, руб.": frame["predicted_inflow"] - frame["actual_inflow"],
                "Фактические списания, руб.": frame["actual_outflow"],
                "Прогноз списаний, руб.": frame["predicted_outflow"],
                "Ошибка списаний, руб.": frame["predicted_outflow"] - frame["actual_outflow"],
            })
            business["Фактический чистый поток, руб."] = (
                business["Фактические зачисления, руб."]
                - business["Фактические списания, руб."]
            )
            business["Прогнозный чистый поток, руб."] = (
                business["Прогноз зачислений, руб."]
                - business["Прогноз списаний, руб."]
            )
            frame = business
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _validate_prediction_totals(
    metrics: pd.DataFrame, predictions_path: Path,
) -> Dict[str, object]:
    """Recompute every reported total from durable predictions before reporting."""
    required_prediction_columns = {
        "fold", "model", "actual_inflow", "predicted_inflow",
        "actual_outflow", "predicted_outflow",
    }
    parquet = pq.ParquetFile(predictions_path)
    missing = required_prediction_columns - set(parquet.schema.names)
    if missing:
        raise RuntimeError("Проверка отчёта: в predictions нет колонок {}.".format(sorted(missing)))

    totals: Dict[Tuple[int, str], Dict[str, float]] = {}
    prediction_rows = 0
    negative_predictions = 0
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=sorted(required_prediction_columns)
    ):
        frame = batch.to_pandas()
        prediction_rows += len(frame)
        negative_predictions += int(
            frame[["predicted_inflow", "predicted_outflow"]].lt(-1e-6).sum().sum()
        )
        if not np_isfinite_frame(frame, [
            "actual_inflow", "predicted_inflow", "actual_outflow", "predicted_outflow"
        ]):
            raise RuntimeError("Проверка отчёта: predictions содержит NaN или infinity.")
        grouped = frame.groupby(["fold", "model"], as_index=False).agg(
            actual_inflow=("actual_inflow", "sum"),
            predicted_inflow=("predicted_inflow", "sum"),
            actual_outflow=("actual_outflow", "sum"),
            predicted_outflow=("predicted_outflow", "sum"),
        )
        for _, row in grouped.iterrows():
            key = (int(row["fold"]), str(row["model"]))
            accumulator = totals.setdefault(key, {
                "actual_inflow": 0.0,
                "predicted_inflow": 0.0,
                "actual_outflow": 0.0,
                "predicted_outflow": 0.0,
            })
            for column in accumulator:
                accumulator[column] += float(row[column])

    if prediction_rows < 1:
        raise RuntimeError("Проверка отчёта: predictions пуст.")
    if negative_predictions:
        raise RuntimeError(
            "Проверка отчёта: найдено {} отрицательных прогнозов зачислений/списаний.".format(
                negative_predictions
            )
        )

    errors: List[str] = []
    flow_columns = {
        "inflow_credit": ("actual_inflow", "predicted_inflow"),
        "outflow_debit": ("actual_outflow", "predicted_outflow"),
    }
    duplicate_keys = metrics.duplicated(["fold", "model", "flow"]).sum()
    if duplicate_keys:
        errors.append("дубли строк метрик: {}".format(int(duplicate_keys)))
    for _, row in metrics.iterrows():
        key = (int(row["fold"]), str(row["model"]))
        values = totals.get(key)
        columns = flow_columns.get(str(row["flow"]))
        if values is None or columns is None:
            errors.append("нет predictions для {} / {}".format(key, row["flow"]))
            continue
        for metric_column, prediction_column in zip(
            ("actual_total", "predicted_total"), columns
        ):
            expected = float(row[metric_column])
            calculated = float(values[prediction_column])
            tolerance = max(0.01, abs(expected) * 1e-8)
            if abs(expected - calculated) > tolerance:
                errors.append(
                    "{} {} {}: CSV={} Parquet={}".format(
                        key, row["flow"], metric_column, expected, calculated
                    )
                )
    if errors:
        raise RuntimeError(
            "Проверка отчёта не пройдена; рейтинг не создан. {}".format(" | ".join(errors[:10]))
        )
    return {
        "prediction_rows": prediction_rows,
        "metric_rows": int(len(metrics)),
        "folds": int(metrics["fold"].nunique()),
        "models": int(metrics["model"].nunique()),
        "checked_totals": int(len(metrics) * 2),
    }


def np_isfinite_frame(frame: pd.DataFrame, columns: List[str]) -> bool:
    return bool(np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all())


def _write_validation_report(output: Path, result: Dict[str, object]) -> None:
    folds = int(result["folds"])
    readiness = (
        "Количество тестовых месяцев достаточно для оценки стабильности."
        if folds >= 10 else
        "Проверка расчётов пройдена, но для оценки стабильности нужно минимум 10 месяцев."
    )
    text = "\n".join([
        "# Проверка корректности расчёта",
        "",
        "**Статус: ПРОЙДЕНА.**",
        "",
        "- Строк детальных прогнозов: **{:,}**.".format(int(result["prediction_rows"])),
        "- Строк метрик: **{:,}**.".format(int(result["metric_rows"])),
        "- Тестовых месяцев: **{}**.".format(folds),
        "- Моделей: **{}**.".format(result["models"]),
        "- Фактические и прогнозные суммы CSV заново сверены с Parquet для каждого "
        "месяца, потока и модели.",
        "- NaN, infinity и отрицательных прогнозов зачислений/списаний не найдено.",
        "",
        readiness,
        "",
    ])
    (output / "00_проверка_расчета.md").write_text(text, encoding="utf-8")


def _business_conclusion(summary: pd.DataFrame, periods: int) -> str:
    status = (
        "ОЦЕНКА СТАБИЛЬНОСТИ ВЫПОЛНЕНА"
        if periods >= 10
        else "ТОЛЬКО ДИАГНОСТИКА: НУЖНО МИНИМУМ 10 ТЕСТОВЫХ МЕСЯЦЕВ"
    )
    lines = [
        "БИЗНЕС-ВЫВОД ПО СТАБИЛЬНОСТИ МОДЕЛЕЙ",
        "Статус: {}".format(status),
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
            "  Лидер {}теста: {} ({})".format(
                "предварительного " if periods < 10 else "",
                model_display_ru(best), best["model"],
            ),
            "  Средняя ошибка месячного итога: {:.2f}%".format(
                best["aggregate_mape_mean_percent"]
            ),
            "  Худший месяц: {:.2f}%".format(best["aggregate_mape_worst_percent"]),
            "  Среднее смещение: {:+.2f}%".format(best["bias_mean_percent"]),
            "",
        ])
    lines.extend([
        "Важно: прогноз описывает зачисления и списания. Для настоящего кассового разрыва",
        "нужны начальные остатки на счетах и обязательные будущие платежи.",
    ])
    return "\n".join(lines)


def _number(value: object, digits: int = 2, sign: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "н/д"
    if pd.isna(numeric):
        return "н/д"
    pattern = "{:+." + str(digits) + "f}" if sign else "{:." + str(digits) + "f}"
    return pattern.format(numeric)


def _markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _markdown_summary(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
) -> str:
    periods = int(windows["fold"].nunique()) if not windows.empty else 0
    test_start = pd.to_datetime(windows["test_month"]).min() if not windows.empty else None
    test_end = pd.to_datetime(windows["test_month"]).max() if not windows.empty else None
    train_start = pd.to_datetime(windows["train_start"]).min() if not windows.empty else None
    train_end = pd.to_datetime(windows["train_end"]).max() if not windows.empty else None

    company_summary_column = (
        "company_median_ape_mean_percent"
        if "company_median_ape_mean_percent" in summary.columns
        else "company_mape_mean_percent"
    )
    overall = summary.groupby("model", as_index=False).agg(
        mean_mape=("aggregate_mape_mean_percent", "mean"),
        worst_mape=("aggregate_mape_worst_percent", "max"),
        mean_wape=("wape_mean_percent", "mean"),
        mean_company_mape=(company_summary_column, "mean"),
    ).sort_values("mean_mape")
    overall_winner = overall.iloc[0] if not overall.empty else None

    lines = [
        "# Краткий отчёт по прогнозу денежных потоков",
        "",
        "## Что прогнозирует модель",
        "",
        "Для каждого ИНН модель прогнозирует **общую сумму зачислений** и "
        "**общую сумму списаний** за календарный месяц. Чистый денежный поток "
        "рассчитывается как `зачисления − списания`.",
        "",
        "> Это прогноз денежных потоков, а не кассового разрыва. Чтобы определить "
        "кассовый разрыв, дополнительно нужны остаток денег на начало периода, "
        "обязательные платежи и доступные кредитные лимиты.",
        "",
        "## Как проводилась проверка",
        "",
        "- Тестовых месяцев: **{}**.".format(periods),
    ]
    if test_start is not None and test_end is not None:
        lines.append("- Тестовый диапазон: **{} — {}**.".format(
            test_start.strftime("%Y-%m"), test_end.strftime("%Y-%m")
        ))
    if train_start is not None and train_end is not None:
        lines.append("- История обучения в разных folds: **{} — {}**.".format(
            train_start.strftime("%Y-%m"), train_end.strftime("%Y-%m")
        ))
    lines.extend([
        "- Для каждого тестового месяца модель видела только предыдущие месяцы. "
        "Будущий тестовый месяц в обучение не попадал.",
        "- Итоговая строка — это среднее качество по всем тестовым месяцам, поэтому "
        "она показывает не единичный удачный прогноз, а устойчивость во времени.",
        "",
        "## Главный вывод",
        "",
    ])
    if overall_winner is not None:
        lines.extend([
            "По среднему совокупному MAPE двух потоков лучший общий результат показала "
            "**{}** (`{}`).".format(
                model_name_ru(overall_winner["model"]), overall_winner["model"]
            ),
            "",
            "- Средняя ошибка месячных итогов двух потоков: **{}%**.".format(
                _number(overall_winner["mean_mape"])
            ),
            "- Худшая ошибка среди потоков и тестовых месяцев: **{}%**.".format(
                _number(overall_winner["worst_mape"])
            ),
            "- Средний WAPE: **{}%**.".format(_number(overall_winner["mean_wape"])),
            "",
            "Общий победитель — удобный ориентир, но для бизнеса правильнее отдельно "
            "смотреть зачисления и списания: у них могут победить разные модели.",
            "",
        ])

    lines.extend(["## Лучшие модели отдельно по потокам", ""])
    best_rows: List[List[str]] = []
    for flow_id, flow_name in FLOW_NAMES_RU.items():
        candidates = summary[summary["flow"].eq(flow_id)].sort_values(
            "aggregate_mape_mean_percent"
        )
        if candidates.empty:
            continue
        best = candidates.iloc[0]
        best_rows.append([
            flow_name,
            model_display_ru(best),
            _number(best["aggregate_mape_mean_percent"]) + "%",
            _number(best["aggregate_mape_std_percent"]) + "%",
            _number(best["aggregate_mape_worst_percent"]) + "%",
            _number(best["bias_mean_percent"], sign=True) + "%",
        ])
    lines.extend(_markdown_table(
        [
            "Поток", "Лучшая модель", "Средняя ошибка итога",
            "Скачки ошибки", "Самый плохой месяц", "Завышение/занижение",
        ],
        best_rows,
    ))

    lines.extend(["", "## Полное сравнение моделей", ""])
    comparison_rows: List[List[str]] = []
    ordered = summary.sort_values(["flow", "aggregate_mape_mean_percent"])
    for _, row in ordered.iterrows():
        comparison_rows.append([
            model_display_ru(row),
            FLOW_NAMES_RU.get(str(row["flow"]), str(row["flow"])),
            _number(row["aggregate_mape_mean_percent"]) + "%",
            _number(row["aggregate_mape_std_percent"]) + "%",
            _number(row["aggregate_mape_worst_percent"]) + "%",
            _number(row["wape_mean_percent"]) + "%",
            _number(row[company_summary_column]) + "%",
            _number(row["bias_mean_percent"], sign=True) + "%",
        ])
    lines.extend(_markdown_table(
        [
            "Модель", "Поток", "Ошибка итога", "Скачки ошибки", "Худший месяц",
            "Ошибка по компаниям", "Медианная ошибка ИНН", "Завышение/занижение",
        ],
        comparison_rows,
    ))

    lines.extend([
        "",
        "## Что означают показатели",
        "",
        "- **Ошибка итога** — ошибка месячной суммы по всем ИНН: "
        "`|сумма прогноза − сумма факта| / сумма факта`. Это основной показатель "
        "для задачи совокупных зачислений и списаний. Чем меньше, тем лучше.",
        "- **Скачки ошибки** — стандартное отклонение ошибки между тестовыми месяцами. "
        "Малое значение означает, что качество меньше скачет от месяца к месяцу.",
        "- **Худший месяц** — максимальный общий MAPE среди тестовых периодов. Он "
        "показывает риск плохого месяца, который среднее значение может скрыть.",
        "- **WAPE** — сумма абсолютных ошибок по отдельным ИНН, делённая на общую "
        "фактическую сумму. Ошибки разных компаний здесь не компенсируют друг друга.",
        "- **Медианная ошибка ИНН** — процентная ошибка типичного ИНН с ненулевым "
        "фактом. Медиана не взрывается из-за нескольких клиентов с очень маленькими суммами.",
        "- **Смещение** — систематическое завышение или занижение. Плюс означает, "
        "что модель в среднем завышает поток; минус — занижает; около нуля лучше.",
        "",
        "### Почему итог месяца может быть угадан, а прогнозы отдельных ИНН — нет",
        "",
        "Модель может завысить прогноз одной компании и занизить другой. В общей "
        "сумме эти ошибки взаимно компенсируются, поэтому ошибка итога будет низкой. "
        "Показатели по компаниям покажут, что точность конкретных клиентов хуже.",
        "",
        "## Как принять решение",
        "",
        "1. Для планирования общей ликвидности сначала сравните **ошибку итога**, "
        "**худший месяц** и **смещение**.",
        "2. Для поклиентских решений обязательно смотрите **ошибку по компаниям** "
        "и **медианную ошибку ИНН**.",
        "3. Сравнивайте ML-модели с `Прогнозом по среднему`: сложная модель имеет "
        "смысл только если стабильно лучше простого baseline.",
        "4. Универсальной границы «хорошей точности» нет. Допустимый процент ошибки "
        "нужно связать с денежной суммой риска и бизнес-допуском компании.",
        "",
        "## Какие файлы смотреть",
        "",
        "- `отчет_стабильность_моделей.csv` — компактный рейтинг моделей.",
        "- `отчет_метрики_по_периодам.csv` — детализация каждого тестового месяца.",
        "- `отчет_прогнозы_по_инн.parquet` — факт и прогноз по каждому ИНН.",
        "- `отчет_окна_тестирования.csv` — какие месяцы использовались для обучения и теста.",
        "- `бизнес_вывод.txt` — самый короткий текстовый вывод.",
        "- `краткий_отчет.md` — этот документ с расшифровкой результатов.",
    ])
    return "\n".join(lines) + "\n"


def _number_ru(value: object, digits: int = 2, sign: bool = False) -> str:
    value_text = _number(value, digits=digits, sign=sign)
    return value_text.replace(".", ",")


def _money_ru(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "н/д"
    if pd.isna(numeric):
        return "н/д"
    return "{:,.0f} ₽".format(numeric).replace(",", " ")


def _business_markdown(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
) -> str:
    """Create a decision-oriented report, not a dump of technical metrics."""
    metric_values = _format_month_columns(metrics)
    window_values = _format_month_columns(windows)
    periods = int(window_values["fold"].nunique()) if not window_values.empty else 0
    enough_periods = periods >= 10
    test_months = sorted(metric_values["test_month"].dropna().astype(str).unique())
    models_count = int(summary["model"].nunique()) if not summary.empty else 0

    winner_rows: List[List[str]] = []
    executive_lines: List[str] = []
    monthly_rows: List[List[str]] = []
    for flow_id, flow_name in FLOW_NAMES_RU.items():
        candidates = summary[summary["flow"].eq(flow_id)].sort_values(
            "aggregate_mape_mean_percent"
        )
        if candidates.empty:
            continue
        winner = candidates.iloc[0]
        winner_id = str(winner["model"])
        winner_name = model_display_ru(winner)
        winner_metrics = metric_values[
            metric_values["flow"].eq(flow_id) & metric_values["model"].eq(winner_id)
        ].sort_values("test_month")
        absolute_gap = (
            winner_metrics["predicted_total"] - winner_metrics["actual_total"]
        ).abs()
        mean_gap = float(absolute_gap.mean()) if not absolute_gap.empty else float("nan")
        worst_gap = float(absolute_gap.max()) if not absolute_gap.empty else float("nan")
        worst_month = "н/д"
        if not winner_metrics.empty:
            worst_index = winner_metrics["aggregate_mape_percent"].idxmax()
            worst_month = str(winner_metrics.loc[worst_index, "test_month"])

        baseline = candidates[candidates["model"].eq("trailing_mean")]
        baseline_mape = (
            float(baseline.iloc[0]["aggregate_mape_mean_percent"])
            if not baseline.empty else float("nan")
        )
        winner_mape = float(winner["aggregate_mape_mean_percent"])
        improvement_pp = baseline_mape - winner_mape
        improvement_text = "нет baseline"
        if not pd.isna(baseline_mape):
            relative = improvement_pp / max(abs(baseline_mape), 1e-9) * 100.0
            improvement_text = "{} п.п. ({}%)".format(
                _number_ru(improvement_pp, sign=True), _number_ru(relative, sign=True)
            )

        stability = _stability_label(
            winner["aggregate_mape_mean_percent"], winner["aggregate_mape_std_percent"]
        )
        winner_rows.append([
            flow_name,
            winner_name,
            _number_ru(winner_mape) + "%",
            _number_ru(winner["aggregate_mape_worst_percent"]) + "%",
            stability,
            _number_ru(winner["bias_mean_percent"], sign=True) + "%",
            _money_ru(mean_gap),
            _money_ru(worst_gap),
            improvement_text,
        ])

        if baseline.empty:
            decision = (
                "Простой прогноз по среднему в этом запуске не проверялся, поэтому "
                "преимущество модели над baseline пока не доказано."
            )
        elif winner_id == "trailing_mean":
            decision = (
                "Сложные модели не превзошли простой прогноз по среднему; для этого "
                "потока разумно оставить baseline до улучшения признаков или данных."
            )
        else:
            decision = (
                "Модель лучше простого прогноза по среднему на **{}**. Перед рабочим "
                "использованием нужно сопоставить худшую ошибку и денежное отклонение "
                "с утверждённым бизнес-допуском."
            ).format(improvement_text)
        executive_lines.extend([
            "- **{}:** {} теста — **{}**. В среднем модель "
            "промахнулась в месячном итоге на **{}%**, в самом плохом месяце — "
            "на **{}%** ({}), среднее абсолютное "
            "денежное отклонение **{}**. {}".format(
                flow_name,
                "рекомендуемая модель по результатам" if enough_periods
                else "лидер предварительного",
                winner_name,
                _number_ru(winner_mape),
                _number_ru(winner["aggregate_mape_worst_percent"]),
                worst_month,
                _money_ru(mean_gap),
                decision,
            )
        ])

        for _, row in winner_metrics.iterrows():
            gap = float(row["predicted_total"] - row["actual_total"])
            monthly_rows.append([
                str(row["test_month"]),
                flow_name,
                winner_name,
                _money_ru(row["actual_total"]),
                _money_ru(row["predicted_total"]),
                _money_ru(gap),
                _number_ru(row["aggregate_mape_percent"]) + "%",
                _number_ru(row["wape_percent"]) + "%",
            ])

    lines = [
        "# Бизнес-отчёт по качеству прогноза денежных потоков",
        "",
        "## Статус проверки",
        "",
        (
            "**ПРОВЕРКА СТАБИЛЬНОСТИ ВЫПОЛНЕНА:** использовано не менее 10 "
            "последовательных тестовых месяцев."
            if enough_periods else
            "**ТОЛЬКО ДИАГНОСТИКА — НЕ ДЛЯ ВНЕДРЕНИЯ:** проверено {} из минимально "
            "необходимых 10 тестовых месяцев. Места моделей предварительные."
        ).format(periods),
        "",
        "## Резюме для принятия решения",
        "",
    ]
    lines.extend(executive_lines or ["Результатов для формирования вывода нет."])
    lines.extend([
        "",
        "> **Важно:** «лучшая модель» означает лучшую среди проверенных моделей на "
        "исторических тестовых месяцах. Это не автоматическое разрешение на запуск "
        "в промышленную эксплуатацию: сначала бизнес должен утвердить допустимую "
        "ошибку в процентах и рублях.",
        "",
        "## Как читать рейтинг",
        "",
        "Рейтинг строится отдельно для зачислений и списаний. В каждом из этих "
        "двух разделов строка с местом 1 — лучший результат.",
        "",
        "Главная колонка — **Средняя ошибка месячного итога, %**. Значение 8% "
        "означает: на исторических тестах прогноз общей суммы за месяц в среднем "
        "отличался от факта на 8%. Например, факт 100 млн ₽ и прогноз 108 млн ₽ "
        "дают ошибку 8%. Чем меньше число, тем лучше.",
        "",
        "Колонка **Насколько ошибка скачет по месяцам** показывает стабильность, а "
        "**Ошибка в самом плохом месяце** — риск единичного сильного промаха. Эти "
        "числа не нужно складывать со средней ошибкой.",
        "",
        "## Что было проверено",
        "",
        "- Тестовых месяцев: **{}**.".format(periods),
        "- Проверено моделей: **{}**.".format(models_count),
        "- Тестовый диапазон: **{}**.".format(
            "{} — {}".format(test_months[0], test_months[-1]) if test_months else "н/д"
        ),
        "- Метод проверки: для каждого месяца модель обучалась только на прошлом; "
        "тестовый месяц в обучение не попадал.",
        "- Прогнозируемые показатели: общая сумма зачислений и общая сумма списаний "
        "по каждому ИНН за календарный месяц.",
        "",
        "## {}".format(
            "Рекомендуемые модели" if enough_periods else "Предварительные лидеры"
        ),
        "",
    ])
    lines.extend(_markdown_table(
        [
            "Поток", "Модель", "Средняя ошибка", "Худшая ошибка", "Стабильность",
            "Смещение", "Средняя ошибка в рублях", "Максимальная ошибка в рублях",
            "Улучшение к baseline",
        ],
        winner_rows,
    ))

    lines.extend(["", "## Полный рейтинг моделей", ""])
    for flow_id, flow_name in FLOW_NAMES_RU.items():
        lines.extend(["### {}".format(flow_name), ""])
        ranking_rows: List[List[str]] = []
        candidates = summary[summary["flow"].eq(flow_id)].sort_values(
            "aggregate_mape_mean_percent"
        )
        for place, (_, row) in enumerate(candidates.iterrows(), start=1):
            ranking_rows.append([
                str(place),
                model_display_ru(row),
                _number_ru(row["aggregate_mape_mean_percent"]) + "%",
                _number_ru(row["aggregate_mape_std_percent"]) + " п.п.",
                _number_ru(row["aggregate_mape_worst_percent"]) + "%",
                _number_ru(row["wape_mean_percent"]) + "%",
                _number_ru(row["bias_mean_percent"], sign=True) + "%",
            ])
        lines.extend(_markdown_table(
            [
                "Место", "Модель", "Средняя ошибка итога", "Скачки ошибки",
                "Самый плохой месяц", "Ошибка по компаниям", "Завышение/занижение",
            ],
            ranking_rows,
        ))
        lines.append("")

    lines.extend([
        "## Результат победителей по месяцам",
        "",
    ])
    lines.extend(_markdown_table(
        [
            "Месяц", "Поток", "Модель", "Факт", "Прогноз", "Отклонение",
            "Ошибка итога", "Ошибка по компаниям",
        ],
        monthly_rows,
    ))

    lines.extend([
        "",
        "## Как читать показатели",
        "",
        "- **Средняя ошибка месячного итога** — на сколько процентов модель в среднем "
        "промахнулась в общей сумме за месяц. Например, факт 100 млн руб., а прогноз "
        "110 млн руб. означает ошибку итога 10%. В расчётах эта метрика называется MAPE.",
        "- **Ошибка в самом плохом месяце** — максимальная ошибка итога за один "
        "тестовый месяц. Этот показатель "
        "показывает риск, который скрывается за средним значением.",
        "- **Ошибка в рублях** — абсолютная разница между общей прогнозной и фактической "
        "суммой. Именно её нужно сопоставлять с финансовым допуском бизнеса.",
        "- **Ошибка по компаниям** — ошибка отдельных ИНН без взаимной компенсации "
        "завышений и занижений разных клиентов. В расчётах эта метрика называется WAPE.",
        "- **Завышение/занижение**: плюс означает систематическое завышение, минус — занижение. "
        "Для управления ликвидностью опасно устойчивое занижение будущих списаний.",
        "- **Стабильность** в отчёте является относительной оценкой разброса ошибки: "
        "до 25% от среднего — стабильная, 25–50% — умеренно стабильная, выше 50% — нестабильная.",
        "",
        "## Ограничения и бизнес-риски",
        "",
        "1. Модель прогнозирует денежные потоки, но не сам кассовый разрыв. Для расчёта "
        "кассового разрыва нужны остаток на начало месяца, обязательные платежи и кредитные лимиты.",
        "2. Хорошая ошибка общей суммы не гарантирует точность по конкретному ИНН: "
        "ошибки клиентов могут компенсировать друг друга.",
        "3. Результат относится к указанным историческим месяцам. После изменения "
        "поведения клиентов или структуры портфеля качество нужно проверять повторно.",
        "4. До внедрения следует зафиксировать два SLA: допустимый средний MAPE и "
        "максимально допустимую ошибку в рублях в худшем месяце.",
        "",
        "## Файлы для работы",
        "",
        "- `00_проверка_расчета.md` — автоматическая сверка итогов CSV "
        "с суммами исходных прогнозов Parquet.",
        "- `01_рейтинг_моделей.csv` — компактная таблица для выбора модели.",
        "- `02_качество_по_месяцам.csv` — факт, прогноз и ошибка каждого месяца.",
        "- `03_окна_тестирования.csv` — периоды обучения, валидации и теста.",
        "- `04_чистый_поток_по_месяцам.csv` — факт и прогноз чистого потока и правильность его знака.",
        "- `отчет_прогнозы_по_инн.parquet` — детализация по ИНН без ограничения Excel.",
        "- `monthly_*.csv` и `monthly_predictions.parquet` — технические исходные отчёты "
        "для аналитиков и воспроизводимости.",
    ])
    return "\n".join(lines) + "\n"


def write_russian_reports(
    output: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    predictions_path: Path,
) -> None:
    """Write human-facing Russian artifacts while keeping technical files compatible."""
    output.mkdir(parents=True, exist_ok=True)
    validation = _validate_prediction_totals(metrics, predictions_path)
    _write_validation_report(output, validation)
    ranking = russian_summary(summary)
    monthly_quality = russian_metrics(metrics)
    test_windows = russian_windows(windows)
    net_flow = russian_net_flow(metrics)
    for filename in ("01_рейтинг_моделей.csv", "отчет_стабильность_моделей.csv"):
        _write_csv(ranking, output / filename)
    for filename in ("02_качество_по_месяцам.csv", "отчет_метрики_по_периодам.csv"):
        _write_csv(monthly_quality, output / filename)
    for filename in ("03_окна_тестирования.csv", "отчет_окна_тестирования.csv"):
        _write_csv(test_windows, output / filename)
    if not net_flow.empty:
        _write_csv(net_flow, output / "04_чистый_поток_по_месяцам.csv")
    _write_russian_predictions(predictions_path, output / "отчет_прогнозы_по_инн.parquet")
    periods = int(windows["fold"].nunique()) if not windows.empty else 0
    (output / "бизнес_вывод.txt").write_text(
        _business_conclusion(summary, periods), encoding="utf-8"
    )
    report = _business_markdown(metrics, summary, windows)
    (output / "бизнес_отчет.md").write_text(report, encoding="utf-8")
    # Keep the old filename as a compatibility alias, but its contents are now the full report.
    (output / "краткий_отчет.md").write_text(report, encoding="utf-8")
