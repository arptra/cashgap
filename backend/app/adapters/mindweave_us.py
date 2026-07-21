from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from app.adapters.base import AdapterError, BaseAdapter, NormalizationResult, iter_tabular_chunks, sample_columns
from app.canonical.normalization import save_canonical


def _first(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


class MindweaveUsAdapter(BaseAdapter):
    required_columns: set[str] = set()

    def detect(self, files: list[Path]) -> list[Path]:
        return [path for path in files if path.suffix.lower() in {".csv", ".parquet", ".jsonl", ".json"}]

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise AdapterError("Mindweave is multi-table; use normalize(files, output_dir)")

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        compatible = self.detect(files)
        tables: dict[str, pd.DataFrame] = {}
        for path in compatible:
            name = path.stem.lower()
            chunks = list(iter_tabular_chunks(path))
            if chunks:
                tables[name] = pd.concat(chunks, ignore_index=True)
        transaction_key = next((key for key in tables if "transaction" in key), None)
        if not transaction_key:
            raise AdapterError("Cannot find bank_transactions table")
        transactions = tables[transaction_key]
        columns = list(transactions.columns)
        date_col = _first(columns, ["date", "transaction_date", "posted_at", "timestamp"])
        amount_col = _first(columns, ["amount", "transaction_amount", "value"])
        account_col = _first(columns, ["bank_account_id", "account_id"])
        company_col = _first(columns, ["company_id", "business_id"])
        direction_col = _first(columns, ["direction", "transaction_type", "type"])
        description_col = _first(columns, ["description", "memo", "category"])
        if not date_col or not amount_col:
            raise AdapterError("bank_transactions needs date and amount columns")

        account_mapping: dict[str, str] = {}
        account_key = next((key for key in tables if "account" in key and "transaction" not in key), None)
        if account_key:
            accounts = tables[account_key]
            aid = _first(accounts.columns, ["id", "bank_account_id", "account_id"])
            cid = _first(accounts.columns, ["company_id", "business_id"])
            if aid and cid:
                account_mapping = dict(zip(accounts[aid].astype(str), accounts[cid].astype(str)))

        amount = pd.to_numeric(transactions[amount_col], errors="coerce").fillna(0)
        if direction_col:
            direction = transactions[direction_col].astype(str).str.lower()
            debit_mask = direction.str.contains("debit|out|withdraw|expense")
            credit_mask = direction.str.contains("credit|in|deposit|income")
            unresolved = ~(debit_mask | credit_mask)
            debit_mask = debit_mask | (unresolved & (amount < 0))
            credit_mask = credit_mask | (unresolved & (amount >= 0))
        else:
            debit_mask = amount < 0
            credit_mask = amount >= 0
        if company_col:
            clients = transactions[company_col].astype(str)
        elif account_col and account_mapping:
            clients = transactions[account_col].astype(str).map(account_mapping).fillna("UNKNOWN_COMPANY")
        else:
            clients = pd.Series("SINGLE_COMPANY", index=transactions.index)
        dates = pd.to_datetime(transactions[date_col], errors="coerce")
        canonical = pd.DataFrame(
            {
                "client_id": clients,
                "month": dates.dt.strftime("%Y-%m"),
                "transaction_label": transactions[description_col].fillna("OTHER").astype(str) if description_col else "OTHER",
                "debit_sum": amount.abs().where(debit_mask, 0.0),
                "credit_sum": amount.abs().where(credit_mask, 0.0),
                "debit_nonzero_count": debit_mask.astype(int),
                "credit_nonzero_count": credit_mask.astype(int),
                "source_dataset": self.source_id,
                "source_provider": self.provider,
                "currency": None,
                "data_quality_flags": "one_company",
            }
        ).dropna(subset=["month"])
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")

        liquidity_path = None
        statement_key = next((key for key in tables if "statement" in key), None)
        if statement_key:
            statements = tables[statement_key]
            sdate = _first(statements.columns, ["date", "statement_date", "month"])
            balance = _first(statements.columns, ["closing_balance", "balance", "ending_balance"])
            scid = _first(statements.columns, ["company_id", "business_id"])
            said = _first(statements.columns, ["bank_account_id", "account_id"])
            if sdate and balance:
                dates = pd.to_datetime(statements[sdate], errors="coerce")
                clients = (
                    statements[scid].astype(str) if scid
                    else statements[said].astype(str).map(account_mapping).fillna("SINGLE_COMPANY") if said
                    else pd.Series("SINGLE_COMPANY", index=statements.index)
                )
                balances = pd.DataFrame({"client_id": clients, "month": dates.dt.strftime("%Y-%m"), "date": dates, "balance": pd.to_numeric(statements[balance], errors="coerce")}).dropna()
                if not balances.empty:
                    liquidity = balances.sort_values("date").groupby(["client_id", "month"], as_index=False).agg(
                        opening_balance=("balance", "first"), closing_balance=("balance", "last"),
                        minimum_observed_balance=("balance", "min"), maximum_observed_balance=("balance", "max"),
                        balance_observations_count=("balance", "count"),
                    )
                    threshold = float(self.options.get("balance_breach_threshold", 0.0))
                    liquidity["balance_breach_proxy"] = (liquidity["minimum_observed_balance"] < threshold).astype(int)
                    liquidity_path = output_dir / "monthly_liquidity.parquet"
                    liquidity.to_parquet(liquidity_path, index=False)
        return NormalizationResult(
            canonical_path=canonical_path,
            liquidity_path=liquidity_path,
            mapping={
                "detected_tables": list(tables),
                "transaction_columns": {
                    "date": date_col, "amount": amount_col, "account_id": account_col,
                    "company_id": company_col, "direction": direction_col, "description": description_col,
                },
                "client_level": "company_id",
                "limitation": "one_company; classification disabled",
            },
            quality_flags=["one_company"],
        )

