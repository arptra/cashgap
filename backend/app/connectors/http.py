from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.connectors.base import AccessResult, BaseConnector, ConnectorError


class HttpConnector(BaseConnector):
    @property
    def url(self) -> str:
        url = str(self.options.get("url") or self.source.get("remote_id") or "")
        if urlparse(url).scheme not in {"http", "https"}:
            raise ConnectorError("A valid http:// or https:// URL is required")
        return url

    def check_access(self) -> AccessResult:
        try:
            response = requests.head(self.url, allow_redirects=True, timeout=15)
            response.raise_for_status()
            return AccessResult(True, "HTTP source is accessible", False, self.fetch_metadata())
        except Exception as exc:
            return AccessResult(False, f"HTTP access failed: {exc}")

    def fetch_metadata(self) -> dict:
        response = requests.head(self.url, allow_redirects=True, timeout=15)
        response.raise_for_status()
        return {
            "url": response.url,
            "size_bytes": int(response.headers["content-length"]) if response.headers.get("content-length") else None,
            "content_type": response.headers.get("content-type"),
            "etag": response.headers.get("etag"),
        }

    def download(self, cancel_event: threading.Event | None = None) -> Path:
        filename = self.options.get("filename") or Path(urlparse(self.url).path).name or "downloaded_data"
        filename = Path(str(filename)).name
        target = self.output_dir / filename
        try:
            with requests.get(self.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with target.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        self.ensure_not_cancelled(cancel_event)
                        if chunk:
                            output.write(chunk)
        except Exception as exc:
            raise ConnectorError(f"HTTP download failed: {exc}") from exc
        return self.output_dir

