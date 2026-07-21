from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from app.connectors.base import AccessResult, BaseConnector, ConnectorError


class KaggleDatasetConnector(BaseConnector):
    def _api(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise ConnectorError("Kaggle client is not installed. Run make setup.") from exc
        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            raise ConnectorError(
                "Kaggle authentication failed. Set KAGGLE_USERNAME/KAGGLE_KEY or "
                "place kaggle.json in ~/.kaggle/. Credentials are never stored by CashGap Lab."
            ) from exc
        return api

    def check_access(self) -> AccessResult:
        try:
            metadata = self.fetch_metadata()
            return AccessResult(True, "Kaggle dataset is accessible", True, metadata)
        except ConnectorError as exc:
            return AccessResult(False, str(exc), True)

    def fetch_metadata(self) -> dict[str, Any]:
        try:
            dataset = self._api().dataset_view(self.source["remote_id"])
            return {
                "ref": getattr(dataset, "ref", self.source["remote_id"]),
                "title": getattr(dataset, "title", self.source.get("title")),
                "size_bytes": getattr(dataset, "total_bytes", None),
                "last_updated": str(getattr(dataset, "last_updated", "")) or None,
            }
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"Cannot access Kaggle dataset {self.source['remote_id']}: {exc}") from exc

    def download(self, cancel_event: threading.Event | None = None) -> Path:
        self.ensure_not_cancelled(cancel_event)
        try:
            self._api().dataset_download_files(
                self.source["remote_id"],
                path=str(self.output_dir),
                unzip=True,
                quiet=False,
                force=bool(self.options.get("force", False)),
            )
        except Exception as exc:
            raise ConnectorError(f"Kaggle dataset download failed: {exc}") from exc
        self.ensure_not_cancelled(cancel_event)
        return self.output_dir

