from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow.parquet as pq

from app.canonical.normalization import combine_canonical_parts, save_canonical


@dataclass
class NormalizationResult:
    canonical_path: Path | None
    liquidity_path: Path | None = None
    target_path: Path | None = None
    categorization_path: Path | None = None
    forecast_path: Path | None = None
    mapping: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)


class AdapterError(ValueError):
    pass


def sample_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        if suffix in {".parquet", ".pq"}:
            return pq.ParquetFile(path).schema_arrow.names
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True, nrows=1).columns.astype(str).tolist()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return list(payload[0]) if isinstance(payload[0], dict) else []
            return list(payload) if isinstance(payload, dict) else []
    except Exception:
        return []
    return []


def iter_tabular_chunks(path: Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)
    elif suffix in {".jsonl", ".ndjson"}:
        yield from pd.read_json(path, lines=True, chunksize=chunksize)
    elif suffix in {".parquet", ".pq"}:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for start in range(0, len(payload), chunksize):
                yield pd.DataFrame(payload[start : start + chunksize])
        elif isinstance(payload, dict):
            yield pd.DataFrame([payload])
    else:
        raise AdapterError(f"Unsupported tabular file: {path.name}")


class BaseAdapter(ABC):
    required_columns: set[str] = set()

    def __init__(self, source: dict[str, Any], options: dict[str, Any] | None = None):
        self.source = source
        self.options = options or {}
        self.source_id = source["id"]
        self.provider = source["provider"]

    def detect(self, files: list[Path]) -> list[Path]:
        matches = []
        for path in files:
            columns = set(sample_columns(path))
            if self.required_columns <= columns:
                matches.append(path)
        return matches

    @abstractmethod
    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def normalize(self, files: list[Path], output_dir: Path) -> NormalizationResult:
        compatible = self.detect(files)
        if not compatible:
            raise AdapterError(
                f"No compatible files for adapter {self.source.get('adapter')}; "
                f"required columns: {sorted(self.required_columns)}"
            )
        parts: list[pd.DataFrame] = []
        for path in compatible:
            for chunk in iter_tabular_chunks(path):
                normalized = self.normalize_frame(chunk)
                if not normalized.empty:
                    parts.append(normalized)
        canonical = combine_canonical_parts(parts)
        canonical_path = save_canonical(canonical, output_dir / "monthly_canonical.parquet")
        return NormalizationResult(
            canonical_path=canonical_path,
            mapping=self.mapping_report(),
            quality_flags=sorted(
                set(
                    flag
                    for raw in canonical["data_quality_flags"].dropna().astype(str)
                    for flag in raw.split("|")
                    if flag
                )
            ),
        )

    def mapping_report(self) -> dict[str, Any]:
        return {"adapter": self.source.get("adapter"), "source": self.source_id}

