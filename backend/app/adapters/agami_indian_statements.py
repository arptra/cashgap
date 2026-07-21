from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.adapters.base import AdapterError, BaseAdapter, NormalizationResult
from app.canonical.normalization import save_canonical


CATEGORY_PATTERNS = [
    ("UPI", r"\bUPI\b"),
    ("NEFT", r"\bNEFT\b"),
    ("RTGS", r"\bRTGS\b"),
    ("IMPS", r"\bIMPS\b"),
    ("CHEQUE", r"CHEQUE|\bCHQ\b"),
    ("CASH", r"\bCASH\b"),
    ("ATM", r"\bATM\b"),
    ("CHARGE", r"CHARGE|FEE"),
    ("REVERSAL", r"REVERSAL|REVERSED"),
]


def categorize_description(value: Any) -> str:
    text = str(value or "").upper()
    for label, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, text):
            return label
    return "OTHER"


def _number(value: Any) -> float:
    if value in (None, "", "-"):
        return 0.0
    cleaned = re.sub(r"[^0-9.\-]", "", str(value).replace(",", ""))
    try:
        return float(cleaned or 0)
    except ValueError:
        return 0.0


def _statement_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for key in ("statements", "accounts", "data"):
            if isinstance(payload.get(key), list):
                yield from _statement_records(payload[key])
                return
        yield payload


class AgamiIndianStatementsAdapter(BaseAdapter):
    required_columns: set[str] = set()

    def detect(self, files: list[Path]) -> list[Path]:
        return [path for path in files if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}]

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise AdapterError("Agami statements require structured JSON; use normalize(files, output_dir)")

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        compatible = self.detect(files)
        if not compatible:
            raise AdapterError("No structured JSON files found; PDF/OCR is intentionally unsupported")
        transaction_rows: list[dict[str, Any]] = []
        balance_rows: list[dict[str, Any]] = []
        failed_by_client_month: dict[tuple[str, str], int] = {}
        threshold = float(self.options.get("balance_breach_threshold", 0.0))
        for path in compatible:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                payloads = [json.loads(path.read_text(encoding="utf-8"))]
            for payload in payloads:
                for statement in _statement_records(payload):
                    client_id = str(
                        statement.get("customer_id")
                        or statement.get("account_number")
                        or statement.get("accountNumber")
                        or "UNKNOWN_ACCOUNT"
                    )
                    currency = statement.get("currency")
                    transactions = statement.get("transactions") or statement.get("transaction_details") or []
                    for transaction in transactions:
                        if not isinstance(transaction, dict):
                            continue
                        date = pd.to_datetime(transaction.get("date") or transaction.get("transaction_date"), errors="coerce")
                        if pd.isna(date):
                            continue
                        month = date.strftime("%Y-%m")
                        failed = bool(transaction.get("failed", False))
                        debit = _number(transaction.get("debit"))
                        credit = _number(transaction.get("credit"))
                        if failed:
                            failed_by_client_month[(client_id, month)] = failed_by_client_month.get((client_id, month), 0) + 1
                            debit = credit = 0.0
                        transaction_rows.append(
                            {
                                "client_id": client_id,
                                "month": month,
                                "transaction_label": categorize_description(transaction.get("description")),
                                "debit_sum": abs(debit),
                                "credit_sum": abs(credit),
                                "debit_nonzero_count": int(debit != 0),
                                "credit_nonzero_count": int(credit != 0),
                                "source_dataset": self.source_id,
                                "source_provider": self.provider,
                                "currency": currency,
                                "data_quality_flags": "failed_excluded" if failed else None,
                            }
                        )
                        balance = transaction.get("balance")
                        if balance not in (None, ""):
                            balance_rows.append(
                                {"client_id": client_id, "month": month, "date": date, "balance": _number(balance)}
                            )
                    start_date = pd.to_datetime(statement.get("start_date"), errors="coerce")
                    end_date = pd.to_datetime(statement.get("end_date"), errors="coerce")
                    if not pd.isna(start_date) and statement.get("opening_balance") is not None:
                        balance_rows.append(
                            {"client_id": client_id, "month": start_date.strftime("%Y-%m"), "date": start_date, "balance": _number(statement.get("opening_balance"))}
                        )
                    if not pd.isna(end_date) and statement.get("closing_balance") is not None:
                        balance_rows.append(
                            {"client_id": client_id, "month": end_date.strftime("%Y-%m"), "date": end_date, "balance": _number(statement.get("closing_balance"))}
                        )
        if not transaction_rows:
            raise AdapterError("Structured JSON contains no recognizable transactions")
        canonical = pd.DataFrame(transaction_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")
        liquidity_path = None
        if balance_rows:
            balances = pd.DataFrame(balance_rows).sort_values(["client_id", "date"])
            liquidity = balances.groupby(["client_id", "month"], as_index=False).agg(
                opening_balance=("balance", "first"),
                closing_balance=("balance", "last"),
                minimum_observed_balance=("balance", "min"),
                maximum_observed_balance=("balance", "max"),
                balance_observations_count=("balance", "count"),
            )
            liquidity["balance_breach_proxy"] = (liquidity["minimum_observed_balance"] < threshold).astype(int)
            liquidity_path = output_dir / "monthly_liquidity.parquet"
            liquidity.to_parquet(liquidity_path, index=False)
        return NormalizationResult(
            canonical_path=canonical_path,
            liquidity_path=liquidity_path,
            mapping=self.mapping_report(),
            quality_flags=["failed_excluded"] if failed_by_client_month else [],
        )

    def mapping_report(self) -> dict:
        return {
            "customer_id/account_number": "client_id",
            "transactions.date": "month",
            "transactions.debit/credit": "debit_sum/credit_sum",
            "transactions.description": "rule-based transaction_label",
            "transactions.balance": "monthly liquidity min/max/open/close",
            "transactions.failed": "excluded from turnover; retained as quality signal",
            "balance_breach_proxy": "balance below configured threshold; not a cash-gap target",
        }

