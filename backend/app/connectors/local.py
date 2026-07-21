from __future__ import annotations

import shutil
import threading
from pathlib import Path

from app.connectors.base import AccessResult, BaseConnector, ConnectorError


class LocalFileConnector(BaseConnector):
    @property
    def source_path(self) -> Path:
        raw = self.options.get("path")
        if not raw:
            raise ConnectorError("A local file or directory path is required")
        return Path(str(raw)).expanduser().resolve()

    def check_access(self) -> AccessResult:
        try:
            path = self.source_path
            if not path.exists():
                return AccessResult(False, f"Local path does not exist: {path}")
            return AccessResult(True, "Local path is readable", False, self.fetch_metadata())
        except ConnectorError as exc:
            return AccessResult(False, str(exc))

    def fetch_metadata(self) -> dict:
        path = self.source_path
        if not path.exists():
            raise ConnectorError(f"Local path does not exist: {path}")
        files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        return {"path": str(path), "files": len(files), "size_bytes": sum(item.stat().st_size for item in files)}

    def download(self, cancel_event: threading.Event | None = None) -> Path:
        path = self.source_path
        self.ensure_not_cancelled(cancel_event)
        if path.is_file():
            shutil.copy2(path, self.output_dir / path.name)
        else:
            for item in path.rglob("*"):
                self.ensure_not_cancelled(cancel_event)
                if item.is_file():
                    target = self.output_dir / item.relative_to(path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
        return self.output_dir

