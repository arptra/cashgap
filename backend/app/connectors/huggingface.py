from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from app.connectors.base import AccessResult, BaseConnector, ConnectorError


class HuggingFaceConnector(BaseConnector):
    @property
    def token(self) -> str | None:
        return os.getenv("HF_TOKEN") or None

    def fetch_metadata(self) -> dict[str, Any]:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self.token).dataset_info(self.source["remote_id"])
            return {
                "id": info.id,
                "revision": info.sha,
                "last_modified": info.last_modified.isoformat() if info.last_modified else None,
                "private": info.private,
                "downloads": getattr(info, "downloads", None),
            }
        except Exception as exc:
            raise ConnectorError(f"Cannot access Hugging Face dataset {self.source['remote_id']}: {exc}") from exc

    def check_access(self) -> AccessResult:
        try:
            metadata = self.fetch_metadata()
            return AccessResult(True, "Hugging Face dataset is accessible", bool(self.source.get("requires_auth")), metadata)
        except ConnectorError as exc:
            return AccessResult(False, str(exc), bool(self.source.get("requires_auth")))

    def _stream_dataset(self, cancel_event: threading.Event | None) -> Path:
        try:
            from datasets import IterableDatasetDict, load_dataset
        except ImportError as exc:
            raise ConnectorError("Hugging Face datasets is not installed. Run make setup.") from exc

        split = self.options.get("split")
        loaded = load_dataset(
            self.source["remote_id"],
            split=split,
            streaming=True,
            token=self.token,
            trust_remote_code=False,
            revision=self.options.get("revision"),
        )
        streams = loaded.items() if hasattr(loaded, "items") else [(split or "train", loaded)]
        max_rows = self.options.get("max_rows")
        for split_name, stream in streams:
            target = self.output_dir / f"{split_name}.jsonl"
            with target.open("w", encoding="utf-8") as output:
                for index, row in enumerate(stream):
                    if index % 1000 == 0:
                        self.ensure_not_cancelled(cancel_event)
                    if max_rows is not None and index >= int(max_rows):
                        break
                    output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return self.output_dir

    def _snapshot_dataset(self, cancel_event: threading.Event | None) -> Path:
        self.ensure_not_cancelled(cancel_event)
        try:
            from huggingface_hub import snapshot_download

            patterns = self.options.get("allow_patterns") or self.source.get("download_patterns")
            snapshot_download(
                repo_id=self.source["remote_id"],
                repo_type="dataset",
                revision=self.options.get("revision"),
                token=self.token,
                local_dir=self.output_dir,
                allow_patterns=patterns,
                ignore_patterns=["*.pdf", "*.png", "*.jpg", "*.jpeg"],
            )
            self.ensure_not_cancelled(cancel_event)
            return self.output_dir
        except Exception as exc:
            raise ConnectorError(f"Hugging Face snapshot download failed: {exc}") from exc

    def download(self, cancel_event: threading.Event | None = None) -> Path:
        complex_adapters = {"agami_indian_statements", "mindweave_us"}
        if self.source.get("adapter") in complex_adapters or self.source.get("download_patterns"):
            return self._snapshot_dataset(cancel_event)
        return self._stream_dataset(cancel_event)

