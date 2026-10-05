#!/usr/bin/env python3
"""Export actual negative daily/monthly cash flow from two Parquet files.

Standalone Python 3.8+ script; the only external dependency is PyArrow.
Amounts are positive ruble amounts in both inputs. No model is required.
"""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile


MAX_CENTS = 2 ** 63 - 1
CENT = Decimal("0.01")
NOTE = (
    "Отрицательный поток = поступления минус списания < 0 за выбранный день "
    "или месяц. Это факт по загруженным операциям, не прогноз, не остаток "
    "на счёте и не подтверждённый кассовый разрыв."
)


@lru_cache(maxsize=8192)
def transaction_date(value):
    """Use the recorded calendar date; numeric Unix timestamps use UTC."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or isinstance(value, bool):
        raise ValueError("пустая или неверная дата")
    text = str(value).strip()
    if re.fullmatch(r"\d{8}(?:\.0+)?", text):
        return datetime.strptime(text.split(".")[0], "%Y%m%d").date()
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("дата NaN/infinity")
        magnitude = abs(number)
        if 10_000 <= magnitude < 1_000_000 and number.is_integer():
            return date(1970, 1, 1) + timedelta(days=int(number))
        for lower, upper, divisor in (
            (1e9, 1e11, 1), (1e11, 1e14, 1000),
            (1e14, 1e17, 1_000_000), (1e17, 1e20, 1_000_000_000),
        ):
            if lower <= magnitude < upper:
                return datetime.fromtimestamp(number / divisor, tz=timezone.utc).date()
        raise ValueError("неизвестный числовой формат даты")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return datetime.strptime(text, "%d.%m.%Y").date()


def normalize_inn(value):
    if value is None or isinstance(value, bool):
        raise ValueError("пустой или неверный ИНН")
    text = str(value).strip()
    # Arrow may return an integer INN as a float or Decimal. Keep text zeros.
    if re.fullmatch(r"[0-9]+\.0+", text):
        text = text.split(".")[0]
    if not re.fullmatch(r"[0-9]{10}|[0-9]{12}", text):
        raise ValueError("ИНН должен содержать 10 или 12 цифр; проверьте тип колонки")
    return text


def money_cents(value):
    if value is None or isinstance(value, bool):
        raise ValueError("пустая или неверная сумма")
    try:
        amount = Decimal(str(value).strip())
        if not amount.is_finite() or amount < 0:
            raise ValueError("tr_sum должна быть конечной неотрицательной суммой")
        if amount > Decimal(MAX_CENTS) / 100:
            raise ValueError("сумма превышает допустимый диапазон")
        rounded = amount.quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise ValueError("неверная денежная сумма") from error
    return int(rounded * 100), rounded != amount


def resolve_columns(names, inn_column):
    def key(name):
        return "".join(char for char in name.casefold() if char.isalnum())
    result = []
    for expected in ("tr_date", inn_column, "tr_sum"):
        matches = [name for name in names if key(name) == key(expected)]
        if len(matches) != 1:
            raise ValueError("Нужна одна колонка {}. Найдены: {}".format(expected, names))
        result.append(matches[0])
    return result


def ingest(connection, path, flow, year, batch_size):
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    columns = resolve_columns(dataset.schema.names, "kt_inn" if flow == "inflow" else "dt_inn")
    stats = {"path": str(path.resolve()), "columns": columns, "rows_read": 0,
             "rows_in_year": 0, "rows_outside_year": 0, "rounded_amounts": 0,
             "first_observed_date": None, "last_observed_date": None}
    statement = (
        "INSERT INTO daily (inn, day, {0}) VALUES (?, ?, ?) "
        "ON CONFLICT (inn, day) DO UPDATE SET {0} = daily.{0} + excluded.{0}"
    ).format(flow)
    for batch_number, batch in enumerate(dataset.to_batches(columns=columns, batch_size=batch_size), 1):
        data = batch.to_pydict()
        grouped = {}
        for values in zip(*(data[column] for column in columns)):
            stats["rows_read"] += 1
            try:
                day = transaction_date(values[0])
                day_text = day.isoformat()
                stats["first_observed_date"] = min(stats["first_observed_date"] or day_text, day_text)
                stats["last_observed_date"] = max(stats["last_observed_date"] or day_text, day_text)
                if day.year != year:
                    stats["rows_outside_year"] += 1
                    continue
                inn = normalize_inn(values[1])
                amount, rounded = money_cents(values[2])
                stats["rounded_amounts"] += int(rounded)
                key = (inn, day_text)
                grouped[key] = grouped.get(key, 0) + amount
                if grouped[key] > MAX_CENTS:
                    raise ValueError("переполнение суммы за день")
                stats["rows_in_year"] += 1
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(
                    "{}: строка {}: {}. Выгрузка остановлена; ошибочные операции "
                    "не заменяются нулями.".format(path, stats["rows_read"], error)
                ) from error
        with connection:
            connection.executemany(statement, ((inn, day, amount) for (inn, day), amount in grouped.items()))
        if batch_number == 1 or batch_number % 10 == 0:
            print("{}: прочитано {:,}, за {} год {:,}".format(
                flow, stats["rows_read"], year, stats["rows_in_year"]), flush=True)
    return stats


def rubles(cents):
    sign = "-" if cents < 0 else ""
    whole, fraction = divmod(abs(cents), 100)
    return "{}{},{:02d}".format(sign, whole, fraction)


def write_csv(connection, path, header, query, money_positions):
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(header)
        for row in connection.execute(query):
            writer.writerow([rubles(value) if index in money_positions else value
                             for index, value in enumerate(row)])
            count += 1
    return count


def export(outflow, inflow, output, year=2025, batch_size=131072, overwrite=False):
    outflow, inflow, output = Path(outflow), Path(inflow), Path(output)
    if not 1900 <= year <= 9998 or batch_size < 1:
        raise ValueError("Нужны год 1900–9998 и положительный batch-size")
    if outflow.resolve() == inflow.resolve():
        raise ValueError("Приходы и расходы должны быть двумя разными файлами/каталогами")
    for source in (outflow, inflow):
        if not source.exists():
            raise ValueError("Не найден входной Parquet: {}".format(source))
    if any(source.is_dir() and (output.resolve() == source.resolve()
           or source.resolve() in output.resolve().parents) for source in (outflow, inflow)):
        raise ValueError("Каталог результатов не должен находиться внутри входного Parquet-каталога")
    names = {key: "{}_{}.{}".format(key, year, "json" if key == "run_summary" else "csv")
             for key in ("negative_days", "inn_summary", "negative_months", "run_summary")}
    if any((output / name).resolve() in (outflow.resolve(), inflow.resolve()) for name in names.values()):
        raise ValueError("Путь результата совпадает с входным файлом")
    existing = [name for name in names.values() if (output / name).exists()]
    if existing and not overwrite:
        raise ValueError("Результаты уже существуют: {}. Укажите новый output-dir или --overwrite".format(existing))
    output.mkdir(parents=True, exist_ok=True)
    # A disk-backed aggregate bounds RAM by the input batch instead of all transactions.
    with tempfile.TemporaryDirectory(prefix=".negative-cashflow-", dir=str(output)) as staging:
        stage = Path(staging)
        connection = sqlite3.connect(str(stage / "daily.sqlite3"))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("""CREATE TABLE daily (
                inn TEXT NOT NULL, day TEXT NOT NULL,
                inflow INTEGER NOT NULL DEFAULT 0 CHECK(typeof(inflow)='integer' AND inflow>=0),
                outflow INTEGER NOT NULL DEFAULT 0 CHECK(typeof(outflow)='integer' AND outflow>=0),
                PRIMARY KEY(inn, day)
            ) WITHOUT ROWID""")
            sources = {
                "outflow": ingest(connection, outflow, "outflow", year, batch_size),
                "inflow": ingest(connection, inflow, "inflow", year, batch_size),
            }
            if not any(source["rows_in_year"] for source in sources.values()):
                raise ValueError("В обоих файлах нет операций за {} год".format(year))
            daily_count = write_csv(connection, stage / names["negative_days"],
                ["ИНН", "Дата", "Поступления, руб.", "Списания, руб.",
                 "Чистый поток, руб.", "Превышение списаний, руб."],
                "SELECT inn, day, inflow, outflow, inflow-outflow, outflow-inflow "
                "FROM daily WHERE outflow>inflow ORDER BY inn, day", {2, 3, 4, 5})
            inn_count = write_csv(connection, stage / names["inn_summary"],
                ["ИНН", "Дней с отрицательным потоком", "Первая дата", "Последняя дата",
                 "Поступления за год, руб.", "Списания за год, руб.", "Чистый поток за год, руб.",
                 "Сумма превышений списаний в отрицательные дни, руб.",
                 "Максимальное превышение за день, руб."],
                """SELECT inn, SUM(outflow>inflow),
                    MIN(CASE WHEN outflow>inflow THEN day END),
                    MAX(CASE WHEN outflow>inflow THEN day END),
                    SUM(inflow), SUM(outflow), SUM(inflow)-SUM(outflow),
                    SUM(CASE WHEN outflow>inflow THEN outflow-inflow ELSE 0 END),
                    MAX(outflow-inflow)
                FROM daily GROUP BY inn HAVING SUM(outflow>inflow)>0 ORDER BY inn""",
                {4, 5, 6, 7, 8})
            monthly_count = write_csv(connection, stage / names["negative_months"],
                ["ИНН", "Месяц", "Поступления, руб.", "Списания, руб.", "Чистый поток, руб."],
                """SELECT inn, substr(day, 1, 7), SUM(inflow), SUM(outflow), SUM(inflow)-SUM(outflow)
                FROM daily GROUP BY inn, substr(day, 1, 7)
                HAVING SUM(outflow)>SUM(inflow) ORDER BY inn, substr(day, 1, 7)""", {2, 3, 4})
            observed = connection.execute("SELECT COUNT(DISTINCT inn), MIN(day), MAX(day) FROM daily").fetchone()
            summary = {"year": year, "definition": NOTE, "sources": sources,
                       "observed_inns": observed[0], "first_date_in_year": observed[1],
                       "last_date_in_year": observed[2], "inns_with_negative_days": inn_count,
                       "negative_inn_days": daily_count, "negative_inn_months": monthly_count,
                       "files": names, "money": "RUB; integer kopecks; ROUND_HALF_UP per transaction",
                       "coverage_note": "Отсутствие операций считается нулевым потоком. Полнота источников не подтверждена автоматически."}
            (stage / names["run_summary"]).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            connection.close()
        # Publish only after both inputs and every report have passed validation.
        for name in names.values():
            os.replace(str(stage / name), str(output / name))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Все ИНН и даты отрицательного фактического денежного потока.")
    parser.add_argument("--outflow", required=True, help="Parquet: tr_date, dt_inn/dtinn, tr_sum — списания")
    parser.add_argument("--inflow", required=True, help="Parquet: tr_date, kt_inn/ktinn, tr_sum — поступления")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output-dir", help="По умолчанию artifacts/negative_cashflow_YEAR")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--overwrite", action="store_true", help="Заменить ранее созданные результаты")
    args = parser.parse_args(argv)
    output = args.output_dir or "artifacts/negative_cashflow_{}".format(args.year)
    try:
        summary = export(args.outflow, args.inflow, output, args.year, args.batch_size, args.overwrite)
    except ImportError:
        print("Установите PyArrow: {} -m pip install pyarrow".format(sys.executable), file=sys.stderr)
        return 2
    except (ValueError, OSError, sqlite3.Error, OverflowError) as error:
        print("ОШИБКА: {}".format(error), file=sys.stderr)
        return 2
    print("\nГотово: {} ИНН, {} отрицательных дней (ИНН × дата), {} отрицательных месяцев.".format(
        summary["inns_with_negative_days"], summary["negative_inn_days"], summary["negative_inn_months"]))
    for name in summary["files"].values():
        print(Path(output).resolve() / name)
    print(NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
