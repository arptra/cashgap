from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.adapters.base import BaseAdapter, NormalizationResult, iter_tabular_chunks
from app.canonical.normalization import combine_canonical_parts, save_canonical


class PaySimAdapter(BaseAdapter):
    required_columns = {
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
    }

    def _dates(self, frame: pd.DataFrame) -> pd.Series:
        base = pd.Timestamp(self.options.get("base_date", "2023-01-01"))
        return base + pd.to_timedelta(pd.to_numeric(frame["step"], errors="coerce").fillna(0), unit="h")

    @staticmethod
    def _side(
        frame: pd.DataFrame,
        client_column: str,
        old_column: str,
        new_column: str,
        side: str,
        months: pd.Series,
        include_merchants: bool,
        source_id: str,
        provider: str,
    ) -> pd.DataFrame:
        old = pd.to_numeric(frame[old_column], errors="coerce")
        new = pd.to_numeric(frame[new_column], errors="coerce")
        amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0).abs()
        delta = new - old
        valid_delta = old.notna() & new.notna() & np.isfinite(delta)
        debit = np.where(valid_delta & (delta < 0), -delta, 0.0)
        credit = np.where(valid_delta & (delta > 0), delta, 0.0)

        transaction_type = frame["type"].fillna("OTHER").astype(str).str.upper()
        fallback = ~valid_delta | ((delta == 0) & (amount > 0))
        origin_debit = transaction_type.isin(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
        origin_credit = transaction_type.isin(["CASH_IN", "CREDIT"])
        if side == "origin":
            debit = np.where(fallback & origin_debit, amount, debit)
            credit = np.where(fallback & origin_credit, amount, credit)
        else:
            credit = np.where(fallback & origin_debit, amount, credit)
            debit = np.where(fallback & origin_credit, amount, debit)

        client = frame[client_column].astype(str)
        allowed = client.str.startswith("C") | (include_merchants & client.str.startswith("M"))
        return pd.DataFrame(
            {
                "client_id": client[allowed].to_numpy(),
                "month": months[allowed].dt.strftime("%Y-%m").to_numpy(),
                "transaction_label": transaction_type[allowed].to_numpy(),
                "debit_sum": np.asarray(debit)[allowed.to_numpy()],
                "credit_sum": np.asarray(credit)[allowed.to_numpy()],
                "debit_nonzero_count": (np.asarray(debit)[allowed.to_numpy()] > 0).astype(int),
                "credit_nonzero_count": (np.asarray(credit)[allowed.to_numpy()] > 0).astype(int),
                "source_dataset": source_id,
                "source_provider": provider,
                "currency": None,
                "data_quality_flags": np.where(fallback[allowed], "balance_delta_fallback", None),
            }
        )

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"PaySim columns missing: {sorted(missing)}")
        months = self._dates(frame)
        include_merchants = bool(self.options.get("include_merchants_as_clients", False))
        origin = self._side(
            frame, "nameOrig", "oldbalanceOrg", "newbalanceOrig", "origin", months,
            include_merchants, self.source_id, self.provider,
        )
        destination = self._side(
            frame, "nameDest", "oldbalanceDest", "newbalanceDest", "destination", months,
            include_merchants, self.source_id, self.provider,
        )
        return pd.concat([origin, destination], ignore_index=True)

    def liquidity_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        months = self._dates(frame).dt.strftime("%Y-%m")
        threshold = float(self.options.get("balance_breach_threshold", 0.0))
        sides = []
        for client_col, old_col, new_col in (
            ("nameOrig", "oldbalanceOrg", "newbalanceOrig"),
            ("nameDest", "oldbalanceDest", "newbalanceDest"),
        ):
            clients = frame[client_col].astype(str)
            allowed = clients.str.startswith("C") | (
                bool(self.options.get("include_merchants_as_clients", False)) & clients.str.startswith("M")
            )
            sides.append(
                pd.DataFrame(
                    {
                        "client_id": clients[allowed],
                        "month": months[allowed],
                        "old": pd.to_numeric(frame.loc[allowed, old_col], errors="coerce"),
                        "new": pd.to_numeric(frame.loc[allowed, new_col], errors="coerce"),
                    }
                )
            )
        observations = pd.concat(sides, ignore_index=True).dropna(subset=["old", "new"])
        if observations.empty:
            return pd.DataFrame()
        grouped = observations.groupby(["client_id", "month"], sort=False)
        liquidity = grouped.agg(
            opening_balance=("old", "first"),
            closing_balance=("new", "last"),
            old_min=("old", "min"),
            new_min=("new", "min"),
            old_max=("old", "max"),
            new_max=("new", "max"),
            balance_observations_count=("new", "count"),
        ).reset_index()
        liquidity["minimum_observed_balance"] = liquidity[["old_min", "new_min"]].min(axis=1)
        liquidity["maximum_observed_balance"] = liquidity[["old_max", "new_max"]].max(axis=1)
        liquidity["balance_breach_proxy"] = (liquidity["minimum_observed_balance"] < threshold).astype(int)
        return liquidity.drop(columns=["old_min", "new_min", "old_max", "new_max"])

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        compatible = self.detect(files)
        if not compatible:
            raise ValueError("No PaySim-compatible file found")
        canonical_parts, liquidity_parts = [], []
        for path in compatible:
            for chunk in iter_tabular_chunks(path):
                canonical_parts.append(self.normalize_frame(chunk))
                liquidity = self.liquidity_frame(chunk)
                if not liquidity.empty:
                    liquidity_parts.append(liquidity)
        canonical = combine_canonical_parts(canonical_parts)
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")
        liquidity_path = None
        if liquidity_parts:
            liquidity = pd.concat(liquidity_parts, ignore_index=True)
            liquidity = liquidity.groupby(["client_id", "month"], as_index=False).agg(
                opening_balance=("opening_balance", "first"),
                closing_balance=("closing_balance", "last"),
                minimum_observed_balance=("minimum_observed_balance", "min"),
                maximum_observed_balance=("maximum_observed_balance", "max"),
                balance_observations_count=("balance_observations_count", "sum"),
                balance_breach_proxy=("balance_breach_proxy", "max"),
            )
            liquidity_path = output_dir / "monthly_liquidity.parquet"
            liquidity.to_parquet(liquidity_path, index=False)
        return NormalizationResult(
            canonical_path=canonical_path,
            liquidity_path=liquidity_path,
            mapping=self.mapping_report(),
            quality_flags=["balance_delta_fallback"] if canonical["data_quality_flags"].notna().any() else [],
        )

    def mapping_report(self) -> dict:
        return {
            "step": "month = base_date + step hours",
            "nameOrig/nameDest": "client_id (C* by default; M* optional)",
            "old/new balances": "signed debit/credit delta",
            "type": "transaction_label",
            "isFraud/isFlaggedFraud": "ignored; forbidden as cash-gap target",
        }

