from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forecast_2025 import PREDICTIONS, forecast_year, load_history, parse_dates, read_monthly


def source_files(tmp_path, include_future=False):
    months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    # Two transactions per month, no transaction on a month's last calendar day.
    inflow = pd.DataFrame({"TrDate": np.repeat(months, 2), "KtInn": 7700000001.0,
                           "TrSum": np.tile([600.25, 400.25], len(months))})
    outflow = pd.DataFrame({"tr_date": months, "dt_inn": 7700000001, "tr_sum": 800.25})
    inflow_path, outflow_path = tmp_path / "credit.parquet", tmp_path / "debit.parquet"
    if include_future:
        future = pd.DataFrame({"TrDate": [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-01")],
                               "KtInn": [7700000001.0, 7999999999.0], "TrSum": [1e12, 2e12]})
        inflow = pd.concat([inflow, future], ignore_index=True)
        outflow = pd.concat([outflow, pd.DataFrame({"tr_date": [pd.Timestamp("2025-02-01")],
                                                   "dt_inn": [7700000001], "tr_sum": [3e12]})])
    inflow.to_parquet(inflow_path, index=False)
    outflow.to_parquet(outflow_path, index=False)
    return inflow_path, outflow_path


def test_future_operations_and_future_clients_do_not_change_forecast(tmp_path):
    credit, debit = source_files(tmp_path)
    history, _ = load_history(credit, debit, 2025)
    expected, _ = forecast_year(history, 2025)
    source_files(tmp_path, include_future=True)
    changed_history, info = load_history(credit, debit, 2025)
    actual, _ = forecast_year(changed_history, 2025)
    pd.testing.assert_frame_equal(history, changed_history)
    pd.testing.assert_frame_equal(expected, actual)
    assert actual.inn.unique().tolist() == ["7700000001"]
    assert info["inflow"]["ignored_future_transactions"] == 2
    assert info["outflow"]["ignored_future_transactions"] == 1


def test_stationary_cashflow_produces_twelve_months_in_original_units(tmp_path):
    credit, debit = source_files(tmp_path)
    history, _ = load_history(credit, debit, 2025)
    forecast, info = forecast_year(history, 2025)
    assert history.month.max() == pd.Timestamp("2024-12-01")
    assert forecast.month.tolist() == pd.period_range("2025-01", "2025-12", freq="M").astype(str).tolist()
    np.testing.assert_allclose(forecast.predicted_inflow, 1000.50)
    np.testing.assert_allclose(forecast.predicted_outflow, 800.25)
    np.testing.assert_allclose(forecast.predicted_net_flow, 200.25)
    assert info["training_cutoff_exclusive"] == "2025-01-01"


def test_leading_zero_inn_and_dormant_company_survive_through_year_end(tmp_path):
    credit, debit = source_files(tmp_path)
    frame = pd.read_parquet(credit)
    frame["KtInn"] = frame["KtInn"].astype("int64").astype(str)
    frame = pd.concat([frame, pd.DataFrame({"TrDate": [pd.Timestamp("2023-01-04")],
                                           "KtInn": ["007700000001"], "TrSum": [50.0]})])
    frame.to_parquet(credit, index=False)
    history, _ = load_history(credit, debit, 2025)
    dormant = history.loc[history.inn.eq("007700000001")]
    assert len(dormant) == 24
    assert dormant.month.max() == pd.Timestamp("2024-12-01")
    assert (dormant.iloc[1:][["inflow", "outflow"]] == 0).all().all()
    forecast, _ = forecast_year(history, 2025)
    assert forecast.groupby("inn").size().to_dict() == {"007700000001": 12, "7700000001": 12}
    assert np.isfinite(forecast[PREDICTIONS]).all().all()
    assert (forecast[["predicted_inflow", "predicted_outflow"]] >= 0).all().all()


def test_batches_sum_repeated_month_and_yyyymmdd_dates(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "multi_batch.parquet"
    table = pa.table({"tr_date": [20241203] * 5, "kt_inn": [7700000001] * 5,
                      "tr_sum": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pq.write_table(table, path, row_group_size=2)
    monthly, info = read_monthly(path, "kt_inn", "inflow", pd.Timestamp("2025-01-01"))
    assert monthly.loc[("7700000001", pd.Timestamp("2024-12-01"))] == 15.0
    assert info["used_transactions"] == 5


def test_mixed_date_strings_and_offsets_keep_source_calendar_day():
    actual = parse_dates(pd.Series(["2024-10-01", "2024-10-02 12:00:00",
                                    "2024-12-31T23:30:00-03:00", "2025-01-01T00:30:00+03:00"]))
    expected = pd.Series(pd.to_datetime(["2024-10-01", "2024-10-02", "2024-12-31", "2025-01-01"]))
    pd.testing.assert_series_equal(actual, expected)


def test_no_past_and_insufficient_history_are_explicit_errors(tmp_path):
    credit, debit = source_files(tmp_path)
    with pytest.raises(ValueError, match="Нет истории"):
        load_history(credit, debit, 2023)
    history, _ = load_history(credit, debit, 2025)
    with pytest.raises(ValueError, match="минимум 4"):
        forecast_year(history.loc[history.month.ge("2024-10-01")], 2025)


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_corrupt_historical_amount_is_rejected(tmp_path, amount):
    credit, debit = source_files(tmp_path)
    frame = pd.read_parquet(credit)
    frame.loc[0, "TrSum"] = amount
    frame.to_parquet(credit, index=False)
    with pytest.raises(ValueError, match="неотрицательными"):
        load_history(credit, debit, 2025)


def test_cli_is_standalone_and_csv_totals_reconcile(tmp_path):
    import shutil

    credit, debit = source_files(tmp_path)
    # Execute a copy outside the repo: no hidden imports of experimental modules.
    script = tmp_path / "forecast_2025.py"
    shutil.copy2(Path(__file__).resolve().parents[1] / "forecast_2025.py", script)
    output = tmp_path / "results"
    completed = subprocess.run([sys.executable, str(script), "--inflow", str(credit),
                                "--outflow", str(debit), "--output-dir", str(output)],
                               cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    by_inn = pd.read_csv(output / "forecast_by_inn.csv", dtype={"inn": str})
    monthly = pd.read_csv(output / "monthly_totals.csv")
    annual = pd.read_csv(output / "annual_by_inn.csv", dtype={"inn": str})
    pd.testing.assert_frame_equal(by_inn.groupby("month", as_index=False)[PREDICTIONS].sum(), monthly)
    pd.testing.assert_frame_equal(by_inn.groupby("inn", as_index=False)[PREDICTIONS].sum(), annual)
    np.testing.assert_allclose(by_inn.predicted_net_flow,
                               by_inn.predicted_inflow - by_inn.predicted_outflow, atol=1e-8)
    metadata = json.loads((output / "run_info.json").read_text())
    assert metadata["forecast_year"] == 2025
    assert "За 2025 год" in completed.stdout
