"""Business-result checks using synthetic Parquet, without ML dependencies."""
import contextlib
import csv
from datetime import date, datetime, timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from export_negative_cashflow import export, money_cents, normalize_inn, transaction_date


class NegativeCashflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.outflow = self.path / "out.parquet"
        self.inflow = self.path / "in.parquet"
        self.result = self.path / "results"

    def write(self, path, inn_column, rows):
        data = dict(zip(("tr_date", inn_column, "tr_sum"), zip(*rows))) if rows else {
            "tr_date": pa.array([], type=pa.int64()),
            inn_column: pa.array([], type=pa.string()),
            "tr_sum": pa.array([], type=pa.float64()),
        }
        pq.write_table(pa.table(data), path, row_group_size=2)

    def run_export(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return export(self.outflow, self.inflow, self.result, batch_size=1, **kwargs)

    def read(self, stem):
        with (self.result / (stem + "_2025.csv")).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream, delimiter=";"))

    def test_daily_events_monthly_and_annual_totals_are_distinct(self):
        a, b, c = "0123456789", "7700000002", "770000000003"
        self.write(self.outflow, "DT_INN", [
            (20241231, a, 9999.0), (20250101, a, 60.0), (20250101, a, 40.0),
            (20250102, a, 0.1), (20250102, a, 0.2), (20251231, a, 50.0),
            (20250301, b, 7.0), (20260101, b, 9999.0),
        ])
        self.write(self.inflow, "ktinn", [
            (20250101, a, 20.0), (20250102, a, 0.3), (20250103, a, 500.0),
            (20250301, c, 100.0),
        ])
        summary = self.run_export()
        daily = self.read("negative_days")
        self.assertEqual([(r["ИНН"], r["Дата"], r["Чистый поток, руб."]) for r in daily], [
            (a, "2025-01-01", "-80,00"), (a, "2025-12-31", "-50,00"), (b, "2025-03-01", "-7,00")])
        client = self.read("inn_summary")[0]
        self.assertEqual(client["Чистый поток за год, руб."], "370,00")
        self.assertEqual(client["Дней с отрицательным потоком"], "2")
        self.assertEqual(client["Сумма превышений списаний в отрицательные дни, руб."], "130,00")
        self.assertEqual(client["Максимальное превышение за день, руб."], "80,00")
        self.assertEqual([(r["ИНН"], r["Месяц"]) for r in self.read("negative_months")], [
            (a, "2025-12"), (b, "2025-03")])
        self.assertEqual(summary["observed_inns"], 3)
        self.assertEqual(summary["negative_inn_days"], 3)
        self.assertEqual(summary["inns_with_negative_days"], 2)
        self.assertEqual(summary["sources"]["outflow"]["rows_outside_year"], 2)
        self.assertFalse(list(self.result.glob(".negative-cashflow-*")))

    def test_empty_side_and_no_negative_results(self):
        self.write(self.outflow, "dt_inn", [])
        self.write(self.inflow, "kt_inn", [(20250610, "7700000001", 20.0)])
        summary = self.run_export()
        self.assertEqual(summary["negative_inn_days"], 0)
        self.assertEqual(self.read("negative_days"), [])
        self.assertEqual(self.read("inn_summary"), [])
        self.assertEqual(self.read("negative_months"), [])

    def test_invalid_data_does_not_publish_partial_results(self):
        self.write(self.outflow, "dt_inn", [(20250101, "7700000001", 10.0)])
        for row in [(20250101, "7700000001", -1.0),
                    (20250101, "7700000001", float("nan")),
                    (20250230, "7700000001", 1.0),
                    (20250101, "invalid", 1.0)]:
            with self.subTest(row=row):
                self.write(self.inflow, "kt_inn", [row])
                with self.assertRaises(ValueError):
                    self.run_export()
                self.assertEqual(list(self.result.iterdir()), [])

    def test_existing_results_survive_failed_overwrite(self):
        self.write(self.outflow, "dt_inn", [(20250101, "7700000001", 10.0)])
        self.write(self.inflow, "kt_inn", [])
        self.run_export()
        before = {p.name: p.read_bytes() for p in self.result.iterdir()}
        with self.assertRaisesRegex(ValueError, "уже существуют"):
            self.run_export()
        self.write(self.inflow, "kt_inn", [(20250101, "7700000001", None)])
        with self.assertRaises(ValueError):
            self.run_export(overwrite=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.result.iterdir()})

    def test_directory_input_and_decimal_amounts(self):
        self.outflow = self.path / "parts"
        self.outflow.mkdir()
        for n, amount in enumerate([Decimal("1.235"), Decimal("2.000")]):
            self.write(self.outflow / (str(n) + ".parquet"), "dtinn", [
                (date(2025, 1, 1), "7700000001", amount)])
        self.write(self.inflow, "kt_inn", [(20250101, "7700000001", 1.0)])
        summary = self.run_export()
        self.assertEqual(self.read("negative_days")[0]["Чистый поток, руб."], "-2,24")
        self.assertEqual(summary["sources"]["outflow"]["rounded_amounts"], 1)

    def test_supported_dates_and_inn_types(self):
        expected = date(2025, 1, 1)
        for value in [expected, datetime(2025, 1, 1, 14, 30), 20250101, 20250101.0,
                      "20250101", "2025-01-01", "01.01.2025", "2025-01-01T23:00:00+03:00",
                      "2025-01-01T23:00:00Z", 1735689600, 1735689600000, 20089]:
            with self.subTest(value=value):
                self.assertEqual(transaction_date(value), expected)
        self.assertEqual(normalize_inn(7700000001.0), "7700000001")
        self.assertEqual(normalize_inn(" 0123456789 "), "0123456789")
        self.assertEqual(money_cents(Decimal("0.005"))[0], 1)

    def test_missing_year_fails_instead_of_reporting_no_problem(self):
        self.write(self.outflow, "dt_inn", [(20240101, "7700000001", 10.0)])
        self.write(self.inflow, "kt_inn", [])
        with self.assertRaisesRegex(ValueError, "нет операций"):
            self.run_export()
        self.assertFalse(list(self.result.glob("*.csv")))

    def test_cli_runs_as_a_single_copied_file(self):
        self.write(self.outflow, "dt_inn", [(20251231, "7700000001", 42.0)])
        self.write(self.inflow, "kt_inn", [])
        script = self.path / "standalone.py"
        script.write_bytes(Path(__file__).with_name("export_negative_cashflow.py").read_bytes())
        result = subprocess.run([sys.executable, str(script), "--outflow", str(self.outflow),
                                 "--inflow", str(self.inflow), "--output-dir", str(self.result)],
                                cwd=str(self.path), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.result / "run_summary_2025.json").read_text())
        self.assertEqual(summary["inns_with_negative_days"], 1)
        self.assertEqual(self.read("negative_days")[0]["Дата"], "2025-12-31")


if __name__ == "__main__":
    unittest.main()
