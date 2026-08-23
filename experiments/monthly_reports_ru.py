#!/usr/bin/env python3
"""Русскоязычные бизнес-отчёты для месячного cash-flow benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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

    overall = summary.groupby("model", as_index=False).agg(
        mean_mape=("aggregate_mape_mean_percent", "mean"),
        worst_mape=("aggregate_mape_worst_percent", "max"),
        mean_wape=("wape_mean_percent", "mean"),
        mean_company_mape=("company_mape_mean_percent", "mean"),
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
            "- Средний MAPE двух потоков: **{}%**.".format(
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
            model_name_ru(best["model"]),
            _number(best["aggregate_mape_mean_percent"]) + "%",
            _number(best["aggregate_mape_std_percent"]) + "%",
            _number(best["aggregate_mape_worst_percent"]) + "%",
            _number(best["bias_mean_percent"], sign=True) + "%",
        ])
    lines.extend(_markdown_table(
        ["Поток", "Лучшая модель", "Средний MAPE", "Разброс MAPE", "Худший месяц", "Смещение"],
        best_rows,
    ))

    lines.extend(["", "## Полное сравнение моделей", ""])
    comparison_rows: List[List[str]] = []
    ordered = summary.sort_values(["flow", "aggregate_mape_mean_percent"])
    for _, row in ordered.iterrows():
        comparison_rows.append([
            model_name_ru(row["model"]),
            FLOW_NAMES_RU.get(str(row["flow"]), str(row["flow"])),
            _number(row["aggregate_mape_mean_percent"]) + "%",
            _number(row["aggregate_mape_std_percent"]) + "%",
            _number(row["aggregate_mape_worst_percent"]) + "%",
            _number(row["wape_mean_percent"]) + "%",
            _number(row["company_mape_mean_percent"]) + "%",
            _number(row["bias_mean_percent"], sign=True) + "%",
        ])
    lines.extend(_markdown_table(
        [
            "Модель", "Поток", "MAPE общий", "Разброс", "Худший", "WAPE",
            "MAPE по ИНН", "Смещение",
        ],
        comparison_rows,
    ))

    lines.extend([
        "",
        "## Что означают показатели",
        "",
        "- **MAPE общий** — ошибка совокупной месячной суммы по всем ИНН: "
        "`|сумма прогноза − сумма факта| / сумма факта`. Это основной показатель "
        "для задачи совокупных зачислений и списаний. Чем меньше, тем лучше.",
        "- **Разброс MAPE** — стандартное отклонение ошибки между тестовыми месяцами. "
        "Малое значение означает, что качество меньше скачет от месяца к месяцу.",
        "- **Худший месяц** — максимальный общий MAPE среди тестовых периодов. Он "
        "показывает риск плохого месяца, который среднее значение может скрыть.",
        "- **WAPE** — сумма абсолютных ошибок по отдельным ИНН, делённая на общую "
        "фактическую сумму. Ошибки разных компаний здесь не компенсируют друг друга.",
        "- **MAPE по ИНН** — средняя процентная ошибка компаний, у которых факт не "
        "равен нулю. Показатель чувствителен к небольшим суммам.",
        "- **Смещение** — систематическое завышение или занижение. Плюс означает, "
        "что модель в среднем завышает поток; минус — занижает; около нуля лучше.",
        "",
        "### Почему общий MAPE может быть хорошим, а MAPE по ИНН плохим",
        "",
        "Модель может завысить прогноз одной компании и занизить другой. В общей "
        "сумме эти ошибки взаимно компенсируются, поэтому общий MAPE будет низким. "
        "WAPE и MAPE по ИНН покажут, что точность на уровне конкретного клиента хуже.",
        "",
        "## Как принять решение",
        "",
        "1. Для планирования общей ликвидности сначала сравните **MAPE общий**, "
        "**худший месяц** и **смещение**.",
        "2. Для поклиентских решений обязательно смотрите **WAPE** и **MAPE по ИНН**.",
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
    (output / "краткий_отчет.md").write_text(
        _markdown_summary(metrics, summary, windows), encoding="utf-8"
    )
