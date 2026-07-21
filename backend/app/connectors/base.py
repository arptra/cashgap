from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AccessResult:
    accessible: bool
    message: str
    requires_auth: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ConnectorError(RuntimeError):
    pass


class DownloadCancelled(ConnectorError):
    pass


class BaseConnector(ABC):
    def __init__(self, source: dict[str, Any], output_dir: Path, options: dict[str, Any] | None = None):
        self.source = source
        self.output_dir = output_dir
        self.options = options or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def check_access(self) -> AccessResult:
        raise NotImplementedError

    @abstractmethod
    def fetch_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def download(self, cancel_event: threading.Event | None = None) -> Path:
        raise NotImplementedError

    def list_files(self) -> list[dict[str, Any]]:
        return [
            {"path": str(path.relative_to(self.output_dir)), "size_bytes": path.stat().st_size}
            for path in sorted(self.output_dir.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ]

    def get_local_path(self) -> Path:
        return self.output_dir

    @staticmethod
    def ensure_not_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled("Download cancelled by user")

