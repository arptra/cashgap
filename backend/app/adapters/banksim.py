from __future__ import annotations

import pandas as pd

from app.adapters.base import BaseAdapter


class BankSimAdapter(BaseAdapter):
    required_columns = {"step", "customer", "category", "amount", "fraud"}

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"BankSim columns missing: {sorted(missing)}")
        base = pd.Timestamp(self.options.get("base_date", "2023-01-01"))
        dates = base + pd.to_timedelta(pd.to_numeric(frame["step"], errors="coerce").fillna(0), unit="D")
        amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0).abs()
        return pd.DataFrame(
            {
                "client_id": frame["customer"].astype(str),
                "month": dates.dt.strftime("%Y-%m"),
                "transaction_label": frame["category"].fillna("OTHER").astype(str),
                "debit_sum": amount,
                "credit_sum": 0.0,
                "debit_nonzero_count": (amount > 0).astype(int),
                "credit_nonzero_count": 0,
                "source_dataset": self.source_id,
                "source_provider": self.provider,
                "currency": None,
                "data_quality_flags": "debit_only",
            }
        )

    def mapping_report(self) -> dict:
        return {
            "step": "month = base_date + step days",
            "customer": "client_id",
            "category": "transaction_label",
            "amount": "debit_sum",
            "fraud": "ignored; forbidden as cash-gap target",
            "limitations": ["debit_only", "no_balance", "no_credit_flow"],
        }

