from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.adapters.base import AdapterError, BaseAdapter, NormalizationResult, iter_tabular_chunks
from app.canonical.normalization import save_canonical


class ShellCashflowAdapter(BaseAdapter):
    required_columns: set[str] = set()

    def detect(self, files: list[Path]) -> list[Path]:
        return [path for path in files if path.suffix.lower() == ".csv"]

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise AdapterError("Shell competition files require schema discovery; use normalize")

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        reports = []
        selected: tuple[Path, pd.DataFrame, str, str | None, str | None, str | None] | None = None
        for path in self.detect(files):
            chunks = list(iter_tabular_chunks(path))
            if not chunks:
                continue
            frame = pd.concat(chunks, ignore_index=True)
            lower = {column.lower(): column for column in frame.columns}
            date_col = next((column for key, column in lower.items() if "date" in key or "month" in key), None)
            inflow_col = next((column for key, column in lower.items() if "inflow" in key or "receipt" in key), None)
            outflow_col = next((column for key, column in lower.items() if "outflow" in key or "payment" in key), None)
            net_col = next((column for key, column in lower.items() if "net" in key and "flow" in key), None)
            reports.append({"file": path.name, "date": date_col, "inflow": inflow_col, "outflow": outflow_col, "net_flow": net_col, "size_bytes": path.stat().st_size})
            if date_col and (inflow_col or outflow_col or net_col) and selected is None:
                selected = (path, frame, date_col, inflow_col, outflow_col, net_col)
        if selected is None:
            raise AdapterError("No CSV with a date and cash-flow value columns was found")
        path, frame, date_col, inflow_col, outflow_col, net_col = selected
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        inflow = pd.to_numeric(frame[inflow_col], errors="coerce").fillna(0) if inflow_col else pd.Series(0.0, index=frame.index)
        outflow = pd.to_numeric(frame[outflow_col], errors="coerce").fillna(0).abs() if outflow_col else pd.Series(0.0, index=frame.index)
        if net_col and not inflow_col and not outflow_col:
            net = pd.to_numeric(frame[net_col], errors="coerce").fillna(0)
            inflow, outflow = net.clip(lower=0), (-net).clip(lower=0)
        series = pd.DataFrame({"series_id": "SHELL_AGGREGATE", "date": dates, "inflow": inflow, "outflow": outflow}).dropna(subset=["date"])
        series = series.groupby(["series_id", "date"], as_index=False)[["inflow", "outflow"]].sum()
        series["net_flow"] = series["inflow"] - series["outflow"]
        output_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = output_dir / "forecast_series.parquet"
        series.to_parquet(forecast_path, index=False)
        canonical = pd.concat(
            [
                pd.DataFrame({
                    "client_id": "SHELL_AGGREGATE", "month": series["date"].dt.strftime("%Y-%m"),
                    "transaction_label": "INFLOW", "debit_sum": 0.0, "credit_sum": series["inflow"],
                    "debit_nonzero_count": 0, "credit_nonzero_count": (series["inflow"] != 0).astype(int),
                    "source_dataset": self.source_id, "source_provider": self.provider,
                    "currency": None, "data_quality_flags": "aggregate_series|not_client_level",
                }),
                pd.DataFrame({
                    "client_id": "SHELL_AGGREGATE", "month": series["date"].dt.strftime("%Y-%m"),
                    "transaction_label": "OUTFLOW", "debit_sum": series["outflow"], "credit_sum": 0.0,
                    "debit_nonzero_count": (series["outflow"] != 0).astype(int), "credit_nonzero_count": 0,
                    "source_dataset": self.source_id, "source_provider": self.provider,
                    "currency": None, "data_quality_flags": "aggregate_series|not_client_level",
                }),
            ], ignore_index=True,
        )
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")
        return NormalizationResult(
            canonical_path=canonical_path,
            forecast_path=forecast_path,
            mapping={"files": reports, "selected_file": path.name, "series_id": "SHELL_AGGREGATE"},
            quality_flags=["aggregate_series", "not_client_level"],
        )

