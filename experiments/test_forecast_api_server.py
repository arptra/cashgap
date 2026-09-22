"""API/UI tests with artificial cash-flow records; no customer data used."""
import json
from calendar import monthrange
from datetime import date
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from forecast_api_server import ForecastStore, allocated_cents, create_app


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for client_index, inn in enumerate(("7700000001", "7700000002", "7800000001")):
        for step, period in enumerate(pd.period_range("2026-01", periods=6, freq="M"), start=1):
            if client_index == 1 and step in (2, 4):
                continue
            inflow = 1_850_000 + step * 135_000 + client_index * 250_000
            outflow = 2_470_000 - step * 60_000 + client_index * 200_000
            rows.append({"inn": inn, "period": str(period), "model": "torch_mlp_3_layers",
                         "forecast_step": step, "forecast_type": "direct" if step == 1 else "recursive",
                         "predicted_inflow": float(inflow), "predicted_outflow": float(outflow),
                         "predicted_net_flow": float(inflow - outflow), "negative_net_flow": inflow < outflow})
    pd.DataFrame(rows).to_parquet(directory / "forecasts_api.parquet", index=False)
    (directory / "model_metadata.json").write_text(json.dumps({
        "model_id": "torch_mlp_3_layers", "model_name_ru": "ТЕСТОВЫЕ ДАННЫЕ · MLP, 3 слоя",
        "last_complete_month": "2025-12",
    }, ensure_ascii=False), encoding="utf-8")


class ForecastUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        create_demo(self.path)
        self.store = ForecastStore(self.path)
        self.client = TestClient(create_app(self.store))
        self.addCleanup(self.client.close)

    def test_ui_and_metadata(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="inn-choice"', page.text)
        self.assertIn('id="period"', page.text)
        self.assertIn("Сохранённый прогноз, не онлайн-пересчёт", page.text)
        self.assertIn("no-store", page.headers["cache-control"])
        meta = self.client.get("/ui/meta").json()
        self.assertEqual(meta["inn_count"], 3)
        self.assertEqual(meta["last_complete_month"], "2025-12")

    def test_search_returns_only_actual_inns(self):
        self.assertEqual(self.client.get("/ui/clients?q=780").json()["inns"], ["7800000001"])
        self.assertEqual(self.client.get("/ui/clients?q=999").json()["inns"], [])
        self.assertEqual(len(self.client.get("/ui/clients").json()["inns"]), 3)

    def test_timeline_filters_missing_months(self):
        timeline = self.client.get("/ui/timeline?inn=7700000002").json()
        self.assertEqual([row["период"] for row in timeline["forecasts"]],
                         ["2026-01", "2026-03", "2026-05", "2026-06"])
        self.assertFalse(timeline["truncated"])
        self.assertEqual(self.client.get("/ui/timeline?inn=unknown").status_code, 404)

    def test_existing_api_unchanged(self):
        response = self.client.get("/forecast?inn=7700000001&period=2026-01")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["прогноз_чистого_потока"], -425_000)
        self.assertTrue(data["отрицательный_чистый_поток"])
        self.assertEqual(self.client.post("/forecast", json={"inn": "7700000001", "period": "202601"}).json(), data)

    def test_auth_protects_all_data_routes(self):
        with TestClient(create_app(self.store, api_key="test-secret")) as client:
            self.assertEqual(client.get("/").status_code, 200)
            for path in ("/ui/meta", "/ui/clients", "/ui/example", "/ui/timeline?inn=7700000001", "/health"):
                self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.get(path, headers={"X-API-Key": "test-secret"}).status_code, 200)

    def test_example_changes_client(self):
        for _ in range(10):
            inn = self.client.get("/ui/example?exclude=7700000001").json()["inn"]
            self.assertIn(inn, self.store.inns)
            self.assertNotEqual(inn, "7700000001")

    def test_invalid_data_fails_clearly(self):
        path = self.path / "forecasts_api.parquet"
        original = pd.read_parquet(path)
        for column, value in (("predicted_inflow", float("nan")), ("predicted_outflow", -1),
                              ("predicted_net_flow", 100)):
            frame = original.copy()
            frame.loc[0, column] = value
            frame.to_parquet(path, index=False)
            with self.assertRaises(ValueError):
                ForecastStore(self.path)
        original.iloc[:0].to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "пуста"):
            ForecastStore(self.path)

    def test_daily_flag_off_by_default(self):
        self.assertFalse(self.client.get("/ui/meta").json()["daily_allocation_enabled"])
        for path in ("/ui/daily-dates", "/ui/daily-allocation"):
            self.assertEqual(self.client.get(path).status_code, 404)
        with TestClient(create_app(self.store, daily_allocation=True)) as client:
            self.assertTrue(client.get("/ui/meta").json()["daily_allocation_enabled"])
            query = "?inn=7700000001&period=2026-01"
            self.assertEqual(client.get("/forecast" + query).json(), self.client.get("/forecast" + query).json())

    def test_daily_auth_and_validation(self):
        with TestClient(create_app(self.store, api_key="secret", daily_allocation=True)) as client:
            headers = {"X-API-Key": "secret"}
            for path in ("/ui/daily-dates?inn=7700000001&period=2026-01",
                         "/ui/daily-allocation?inn=7700000001&start_date=2026-01-25"):
                self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.get(path, headers=headers).status_code, 200)
            for start in ("2026-02-30", "2026-1-1", "not-a-date", "9999-12-31"):
                response = client.get("/ui/daily-allocation", params={"inn": "7700000001", "start_date": start}, headers=headers)
                self.assertEqual(response.status_code, 422)
            for inn, start in (("unknown", "2026-01-01"), ("7700000002", "2026-01-25")):
                response = client.get("/ui/daily-allocation", params={"inn": inn, "start_date": start}, headers=headers)
                self.assertEqual(response.status_code, 404)

    def test_daily_dates_exclude_missing_next_month(self):
        complete = self.store.daily_dates("7700000001", "2026-01")["dates"]
        self.assertEqual(len(complete), 31)
        missing_next = self.store.daily_dates("7700000002", "2026-01")["dates"]
        self.assertEqual(missing_next[-1], "2026-01-18")
        self.assertEqual(len(missing_next), 18)
        last_month = self.store.daily_dates("7700000001", "2026-06")["dates"]
        self.assertEqual(last_month[-1], "2026-06-17")
        with self.assertRaises(LookupError):
            self.store.daily_allocation("7700000001", "2026-06-18")

    def test_daily_month_boundary_and_exact_totals(self):
        data = self.store.daily_allocation("7700000001", "2026-01-25")
        self.assertFalse(data["is_daily_model"])
        self.assertEqual(len(data["rows"]), 14)
        self.assertEqual(data["end_date"], "2026-02-07")
        self.assertEqual([row["период"] for row in data["source_months"]], ["2026-01", "2026-02"])
        cumulative = Decimal("0")
        for row in data["rows"]:
            day = date.fromisoformat(row["date"])
            monthly = self.store.forecast("7700000001", row["source_period"])
            self.assertEqual(row["inflow"], allocated_cents(monthly["прогноз_зачислений"], day) / 100)
            net = Decimal(str(row["inflow"])) - Decimal(str(row["outflow"]))
            self.assertEqual(Decimal(str(row["net_flow"])), net)
            cumulative += net
            self.assertEqual(Decimal(str(row["cumulative_net_flow"])), cumulative)
        for column in ("inflow", "outflow", "net_flow"):
            self.assertEqual(sum(Decimal(str(row[column])) for row in data["rows"]), Decimal(str(data["totals"][column])))
        other = self.store.daily_allocation("7700000001", "2026-01-26")
        for first, second in zip(data["rows"][1:], other["rows"]):
            self.assertEqual((first["date"], first["inflow"], first["outflow"]),
                             (second["date"], second["inflow"], second["outflow"]))

    def test_allocation_conserves_every_month_to_kopeck(self):
        for year, month in ((2024, 2), (2026, 2), (2026, 4), (2026, 12)):
            for amount in (0, .01, .03, .29, 1, 1234567.89):
                days = monthrange(year, month)[1]
                allocated = [allocated_cents(amount, date(year, month, day)) for day in range(1, days + 1)]
                self.assertEqual(sum(allocated), int(Decimal(str(amount)) * 100))
                self.assertLessEqual(max(allocated) - min(allocated), 1)

    def test_leap_day_and_new_year(self):
        frame = pd.read_parquet(self.path / "forecasts_api.parquet")
        for periods, start, end in ((["2024-02", "2024-03"], "2024-02-23", "2024-03-07"),
                                    (["2026-12", "2027-01"], "2026-12-25", "2027-01-07")):
            pair = frame.iloc[:2].copy()
            pair["period"] = periods
            pair.to_parquet(self.path / "forecasts_api.parquet", index=False)
            store = ForecastStore(self.path)
            data = store.daily_allocation("7700000001", start)
            self.assertEqual(data["end_date"], end)
            if start.startswith("2024"):
                self.assertIn("2024-02-29", [row["date"] for row in data["rows"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
