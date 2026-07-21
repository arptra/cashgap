from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.adapters.base import AdapterError, BaseAdapter, NormalizationResult, iter_tabular_chunks


class TransactionCategorizationAdapter(BaseAdapter):
    required_columns = {"transaction_description", "category", "country", "currency"}

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise AdapterError(f"Categorization columns missing: {sorted(missing)}")
        return frame[["transaction_description", "category", "country", "currency"]].dropna(
            subset=["transaction_description", "category"]
        )

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        compatible = self.detect(files)
        if not compatible:
            raise AdapterError("No transaction-categorization file found")
        parts = []
        for path in compatible:
            for chunk in iter_tabular_chunks(path):
                parts.append(self.normalize_frame(chunk))
        data = pd.concat(parts, ignore_index=True).drop_duplicates()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "categorization.parquet"
        data.to_parquet(path, index=False)
        return NormalizationResult(
            canonical_path=None,
            categorization_path=path,
            mapping={
                "transaction_description": "text feature",
                "category": "categorization target (not cash-gap target)",
                "country/currency": "metadata",
                "monthly_cashflow": "unsupported: no client_id/date/amount",
            },
            quality_flags=["categorization_only"],
        )

