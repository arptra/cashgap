from __future__ import annotations

from typing import Any


SECRET_KEYS = {
    "authorization", "credential", "credentials", "hf_token", "kaggle_key",
    "kaggle_username", "password", "secret", "token",
}


def sanitize_for_storage(value: Any) -> Any:
    """Remove credentials before a request configuration reaches SQLite."""

    if isinstance(value, dict):
        return {
            key: sanitize_for_storage(item)
            for key, item in value.items()
            if str(key).lower() not in SECRET_KEYS
        }
    if isinstance(value, list):
        return [sanitize_for_storage(item) for item in value]
    return value
