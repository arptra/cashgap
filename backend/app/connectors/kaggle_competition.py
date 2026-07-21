from __future__ import annotations

import threading
import zipfile
from pathlib import Path
from typing import Any

from app.connectors.base import AccessResult, ConnectorError
from app.connectors.kaggle_dataset import KaggleDatasetConnector


class KaggleCompetitionConnector(KaggleDatasetConnector):
    def check_access(self) -> AccessResult:
        try:
            metadata = self.fetch_metadata()
            return AccessResult(True, "Competition is accessible and rules appear accepted", True, metadata)
        except ConnectorError as exc:
            return AccessResult(False, str(exc), True)

    def fetch_metadata(self) -> dict[str, Any]:
        try:
            files = self._api().competition_list_files(self.source["remote_id"])
            return {
                "competition": self.source["remote_id"],
                "files": [
                    {"name": getattr(item, "name", None), "size_bytes": getattr(item, "total_bytes", None)}
                    for item in files
                ],
            }
        except ConnectorError:
            raise
        except Exception as exc:
            message = str(exc)
            if "403" in message or "rules" in message.lower() or "forbidden" in message.lower():
                message = "Competition access denied. Open Kaggle and accept the competition rules first."
            raise ConnectorError(message) from exc

    def download(self, cancel_event: threading.Event | None = None) -> Path:
        self.ensure_not_cancelled(cancel_event)
        try:
            self._api().competition_download_files(
                self.source["remote_id"], path=str(self.output_dir), quiet=False, force=False
            )
            for archive in self.output_dir.glob("*.zip"):
                self.ensure_not_cancelled(cancel_event)
                with zipfile.ZipFile(archive) as zipped:
                    zipped.extractall(self.output_dir)
        except Exception as exc:
            message = str(exc)
            if "403" in message or "rules" in message.lower():
                message = "Download denied: accept the competition rules on Kaggle first."
            raise ConnectorError(message) from exc
        return self.output_dir

