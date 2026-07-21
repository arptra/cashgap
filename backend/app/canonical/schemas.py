from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


CANONICAL_COLUMNS = [
    "client_id",
    "month",
    "transaction_label",
    "debit_sum",
    "credit_sum",
    "debit_nonzero_count",
    "credit_nonzero_count",
    "source_dataset",
    "source_provider",
    "currency",
    "data_quality_flags",
]

LIQUIDITY_COLUMNS = [
    "client_id",
    "month",
    "opening_balance",
    "closing_balance",
    "minimum_observed_balance",
    "maximum_observed_balance",
    "balance_observations_count",
    "balance_breach_proxy",
]

FORBIDDEN_CASH_GAP_TARGETS = {
    "isfraud",
    "fraud",
    "isflaggedfraud",
    "islaundering",
    "amllabel",
    "amllabels",
    "transactionanomaly",
    "transactionanomalylabel",
    "balancebreachproxy",
}


def normalized_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def assert_valid_cash_gap_target(target_column: str) -> None:
    normalized = normalized_column_name(target_column)
    if normalized in FORBIDDEN_CASH_GAP_TARGETS or "fraud" in normalized or "launder" in normalized or "aml" in normalized:
        raise ValueError(
            f"'{target_column}' is a fraud/AML/anomaly label and cannot be used as a cash-gap target"
        )


class CanonicalMonthlyRow(BaseModel):
    client_id: str
    month: str
    transaction_label: str
    debit_sum: float = Field(ge=0)
    credit_sum: float = Field(ge=0)
    debit_nonzero_count: int = Field(ge=0)
    credit_nonzero_count: int = Field(ge=0)
    source_dataset: str
    source_provider: str
    currency: str | None = None
    data_quality_flags: str | None = None

    @field_validator("month")
    @classmethod
    def valid_month(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}", value):
            raise ValueError("month must be YYYY-MM")
        return value


class LiquidityMonthlyRow(BaseModel):
    client_id: str
    month: str
    opening_balance: float | None = None
    closing_balance: float | None = None
    minimum_observed_balance: float | None = None
    maximum_observed_balance: float | None = None
    balance_observations_count: int = Field(default=0, ge=0)
    balance_breach_proxy: int = Field(default=0, ge=0, le=1)


def validate_canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical frame is missing columns: {sorted(missing)}")
    result = frame[CANONICAL_COLUMNS].copy()
    result["client_id"] = result["client_id"].astype(str)
    result["month"] = result["month"].astype(str)
    result["transaction_label"] = result["transaction_label"].fillna("OTHER").astype(str)
    if not result["month"].str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("Every month must use YYYY-MM format")
    numeric = ["debit_sum", "credit_sum", "debit_nonzero_count", "credit_nonzero_count"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    values = result[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Canonical numeric fields contain NaN or infinity")
    if (result[["debit_sum", "credit_sum"]] < 0).any().any():
        raise ValueError("Debit and credit amounts must be non-negative")
    if (result[["debit_nonzero_count", "credit_nonzero_count"]] < 0).any().any():
        raise ValueError("Transaction counts must be non-negative")
    result["debit_nonzero_count"] = result["debit_nonzero_count"].astype("int64")
    result["credit_nonzero_count"] = result["credit_nonzero_count"].astype("int64")
    result["source_dataset"] = result["source_dataset"].astype(str)
    result["source_provider"] = result["source_provider"].astype(str)
    result["currency"] = result["currency"].where(result["currency"].notna(), None)
    result["data_quality_flags"] = result["data_quality_flags"].where(result["data_quality_flags"].notna(), None)
    return result
