from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.adapters.agami_indian_statements import AgamiIndianStatementsAdapter
from app.adapters.banksim import BankSimAdapter
from app.adapters.ibm_aml import IbmAmlAdapter
from app.adapters.mindweave_us import MindweaveUsAdapter
from app.adapters.paysim import PaySimAdapter
from app.adapters.shell_cashflow import ShellCashflowAdapter
from app.adapters.transaction_categorization import TransactionCategorizationAdapter
from app.canonical.schemas import CANONICAL_COLUMNS


FIXTURES = Path(__file__).parent / "fixtures"


def source(adapter: str, provider: str = "local_file") -> dict:
    return {"id": f"fixture_{adapter}", "provider": provider, "adapter": adapter}


def test_paysim_maps_both_sides_and_ignores_merchants(tmp_path: Path) -> None:
    result = PaySimAdapter(source("paysim"), {"base_date": "2023-01-01"}).normalize(
        [FIXTURES / "paysim.csv"], tmp_path
    )
    frame = pd.read_parquet(result.canonical_path)
    assert set(frame.columns) == set(CANONICAL_COLUMNS)
    assert set(frame["client_id"]) == {"C001", "C002", "C003"}
    assert frame["debit_sum"].sum() > 0 and frame["credit_sum"].sum() > 0
    assert result.target_path is None
    assert result.liquidity_path and result.liquidity_path.exists()


def test_banksim_stays_debit_only(tmp_path: Path) -> None:
    result = BankSimAdapter(source("banksim")).normalize([FIXTURES / "banksim.csv"], tmp_path)
    frame = pd.read_parquet(result.canonical_path)
    assert frame["debit_sum"].sum() == pytest.approx(77.5)
    assert frame["credit_sum"].sum() == 0
    assert "debit_only" in set(frame["data_quality_flags"].dropna())


def test_ibm_aml_uses_polars_and_keeps_currency_sides_separate(tmp_path: Path) -> None:
    result = IbmAmlAdapter(source("ibm_aml"), {"chunk_size": 1}).normalize(
        [FIXTURES / "ibm_aml.csv"], tmp_path
    )
    frame = pd.read_parquet(result.canonical_path)
    assert set(frame["currency"]) == {"USD", "EUR"}
    assert frame.loc[frame["client_id"].str.startswith("001:"), "debit_sum"].sum() == 200
    assert frame.loc[frame["client_id"].str.startswith(("002:", "003:")), "credit_sum"].sum() == 190
    assert result.mapping["engine"] == "polars_lazy_streaming"
    assert result.target_path is None


def test_agami_json_excludes_failed_turnover_and_builds_liquidity(tmp_path: Path) -> None:
    result = AgamiIndianStatementsAdapter(source("agami_indian_statements")).normalize(
        [FIXTURES / "agami.json"], tmp_path
    )
    frame = pd.read_parquet(result.canonical_path)
    assert frame["debit_sum"].sum() == 2500
    assert frame["credit_sum"].sum() == 1000
    assert {"UPI", "NEFT", "CASH", "CHARGE"} <= set(frame["transaction_label"])
    assert result.liquidity_path and result.liquidity_path.exists()
    assert "failed_excluded" in result.quality_flags


def test_mindweave_multitable_maps_company_and_disables_client_classification(tmp_path: Path) -> None:
    files = list(FIXTURES.glob("mindweave_*.csv"))
    result = MindweaveUsAdapter(source("mindweave_us")).normalize(files, tmp_path)
    frame = pd.read_parquet(result.canonical_path)
    assert set(frame["client_id"]) == {"CO1"}
    assert frame["debit_sum"].sum() == 650
    assert frame["credit_sum"].sum() == 1200
    assert result.mapping["limitation"].startswith("one_company")
    assert result.liquidity_path and result.liquidity_path.exists()


def test_shell_creates_only_aggregate_forecasting_series(tmp_path: Path) -> None:
    result = ShellCashflowAdapter(source("shell_cashflow")).normalize(
        [FIXTURES / "shell_cashflow.csv"], tmp_path
    )
    series = pd.read_parquet(result.forecast_path)
    assert set(series.columns) == {"series_id", "date", "inflow", "outflow", "net_flow"}
    assert set(series["series_id"]) == {"SHELL_AGGREGATE"}
    assert "not_client_level" in result.quality_flags
    assert result.target_path is None


def test_transaction_categorization_remains_a_separate_table(tmp_path: Path) -> None:
    result = TransactionCategorizationAdapter(source("transaction_categorization")).normalize(
        [FIXTURES / "categorization.csv"], tmp_path
    )
    frame = pd.read_parquet(result.categorization_path)
    assert list(frame.columns) == ["transaction_description", "category", "country", "currency"]
    assert result.canonical_path is None
