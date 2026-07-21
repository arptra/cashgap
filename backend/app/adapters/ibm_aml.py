from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from app.adapters.base import BaseAdapter, NormalizationResult
from app.canonical.normalization import combine_canonical_parts, save_canonical
from app.canonical.normalization import require_fx_for_mixed_currency


class IbmAmlAdapter(BaseAdapter):
    required_columns = {
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering",
    }

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"IBM AML columns missing: {sorted(missing)}")
        month = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True).dt.strftime("%Y-%m")
        paid = pd.to_numeric(frame["Amount Paid"], errors="coerce").fillna(0).abs()
        received = pd.to_numeric(frame["Amount Received"], errors="coerce").fillna(0).abs()
        label = frame["Payment Format"].fillna("OTHER").astype(str)
        sender = pd.DataFrame(
            {
                "client_id": frame["From Bank"].astype(str) + ":" + frame["Account"].astype(str),
                "month": month,
                "transaction_label": label,
                "debit_sum": paid,
                "credit_sum": 0.0,
                "debit_nonzero_count": (paid > 0).astype(int),
                "credit_nonzero_count": 0,
                "source_dataset": self.source_id,
                "source_provider": self.provider,
                "currency": frame["Payment Currency"].astype(str),
                "data_quality_flags": None,
            }
        )
        receiver = pd.DataFrame(
            {
                "client_id": frame["To Bank"].astype(str) + ":" + frame["Account.1"].astype(str),
                "month": month,
                "transaction_label": label,
                "debit_sum": 0.0,
                "credit_sum": received,
                "debit_nonzero_count": 0,
                "credit_nonzero_count": (received > 0).astype(int),
                "source_dataset": self.source_id,
                "source_provider": self.provider,
                "currency": frame["Receiving Currency"].astype(str),
                "data_quality_flags": None,
            }
        )
        result = pd.concat([sender, receiver], ignore_index=True).dropna(subset=["month"])
        require_fx_for_mixed_currency(
            result,
            bool(self.options.get("combine_currencies", False)),
            self.options.get("fx_rates"),
        )
        return result

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        """Normalize one schema-compatible file with Polars streaming batches.

        IBM publishes several very large CSV variants.  We intentionally choose
        the smallest compatible file unless the user explicitly selects one.
        """

        compatible = sorted(self.detect(files), key=lambda path: path.stat().st_size)
        if not compatible:
            raise ValueError("No IBM AML-compatible CSV was found")
        selected_name = self.options.get("selected_file")
        if selected_name:
            selected = next(
                (
                    path
                    for path in compatible
                    if path.name == selected_name or path.as_posix().endswith(f"/{selected_name}")
                ),
                None,
            )
            if selected is None:
                raise ValueError(f"Selected IBM AML file is not compatible: {selected_name}")
        else:
            selected = compatible[0]
        if selected.suffix.lower() != ".csv":
            return super().normalize([selected], output_dir)

        parts: list[pd.DataFrame] = []
        lazy = pl.scan_csv(
            selected,
            infer_schema_length=10_000,
            low_memory=True,
            schema_overrides={
                "From Bank": pl.String,
                "Account": pl.String,
                "To Bank": pl.String,
                "Account.1": pl.String,
                "Receiving Currency": pl.String,
                "Payment Currency": pl.String,
                "Payment Format": pl.String,
            },
        )
        for batch in lazy.collect_batches(
            chunk_size=int(self.options.get("chunk_size", 200_000)),
            engine="streaming",
        ):
            normalized = self.normalize_frame(batch.to_pandas())
            if not normalized.empty:
                parts.append(normalized)
        canonical = combine_canonical_parts(parts)
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")
        return NormalizationResult(
            canonical_path=canonical_path,
            mapping={
                **self.mapping_report(),
                "compatible_files": [
                    {"name": path.name, "size_bytes": path.stat().st_size} for path in compatible
                ],
                "selected_file": selected.name,
                "engine": "polars_lazy_streaming",
            },
        )

    def mapping_report(self) -> dict:
        return {
            "From Bank + Account": "sender client_id",
            "Amount Paid / Payment Currency": "sender debit, isolated by currency",
            "To Bank + Account.1": "receiver client_id",
            "Amount Received / Receiving Currency": "receiver credit, isolated by currency",
            "Payment Format": "transaction_label",
            "Is Laundering": "ignored; forbidden as cash-gap target",
            "file_detection": "schema-based",
        }
