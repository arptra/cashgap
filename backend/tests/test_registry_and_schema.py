from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.canonical.compatibility import build_compatibility_report
from app.canonical.normalization import aggregate_canonical, require_fx_for_mixed_currency
from app.canonical.schemas import CANONICAL_COLUMNS, assert_valid_cash_gap_target, validate_canonical_frame
from app.services.registry import _load_yaml


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["C1", "2023-01", "PAYMENT", 10.0, 0.0, 1, 0, "fixture", "local", "USD", None],
            ["C1", "2023-01", "PAYMENT", 5.0, 2.0, 1, 1, "fixture", "local", "USD", "fallback"],
        ],
        columns=CANONICAL_COLUMNS,
    )


def test_registry_contains_all_required_external_sources() -> None:
    sources = {source["id"]: source for source in _load_yaml()}
    assert set(sources) >= {
        "kaggle_paysim", "kaggle_banksim", "kaggle_ibm_aml", "kaggle_shell_cashflow",
        "hf_paysim_banks", "hf_indian_bank_statements", "hf_us_bank_transactions",
        "hf_transaction_categorization",
    }
    assert all(source["cash_gap_target"] is False for source in sources.values())


def test_canonical_validation_and_debit_credit_aggregation() -> None:
    aggregated = aggregate_canonical(_canonical())
    assert list(aggregated.columns) == CANONICAL_COLUMNS
    assert len(aggregated) == 1
    assert aggregated.iloc[0]["debit_sum"] == 15
    assert aggregated.iloc[0]["credit_sum"] == 2
    assert aggregated.iloc[0]["debit_nonzero_count"] == 2


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_canonical_rejects_nan_and_infinity(bad_value: float) -> None:
    frame = _canonical()
    frame.loc[0, "debit_sum"] = bad_value
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_canonical_frame(frame)


@pytest.mark.parametrize("column", ["isFraud", "fraud", "isFlaggedFraud", "Is Laundering", "AML labels", "transaction_anomaly_label", "balance_breach_proxy"])
def test_fraud_and_aml_fields_are_forbidden_cash_gap_targets(column: str) -> None:
    with pytest.raises(ValueError, match="cannot be used"):
        assert_valid_cash_gap_target(column)


def test_currency_isolation_requires_explicit_fx_for_combining() -> None:
    frame = _canonical()
    frame = pd.concat([frame, frame.assign(currency="EUR")], ignore_index=True)
    require_fx_for_mixed_currency(frame, combine_currencies=False, fx_rates=None)
    with pytest.raises(ValueError, match="FX rates"):
        require_fx_for_mixed_currency(frame, combine_currencies=True, fx_rates=None)


def test_compatibility_explains_why_external_proxy_data_cannot_classify() -> None:
    report = build_compatibility_report(
        {"clients": 1, "months": 12, "has_debit": True, "has_credit": True, "has_balance": True, "has_cash_gap_target": False},
        {"id": "external", "supported_tasks": ["flow_forecasting", "proxy_risk"], "limitations": ["one_company"]},
    )
    assert report["classification_eligible"] is False
    assert report["forecasting_eligible"] is True
    assert report["proxy_eligible"] is True
    assert any("cash-gap target" in reason for reason in report["reasons"])
