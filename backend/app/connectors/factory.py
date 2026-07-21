from __future__ import annotations

from pathlib import Path

from app.connectors.base import BaseConnector, ConnectorError
from app.connectors.http import HttpConnector
from app.connectors.huggingface import HuggingFaceConnector
from app.connectors.kaggle_competition import KaggleCompetitionConnector
from app.connectors.kaggle_dataset import KaggleDatasetConnector
from app.connectors.local import LocalFileConnector


CONNECTORS = {
    "kaggle_dataset": KaggleDatasetConnector,
    "kaggle_competition": KaggleCompetitionConnector,
    "huggingface": HuggingFaceConnector,
    "http": HttpConnector,
    "local": LocalFileConnector,
}


def create_connector(source: dict, output_dir: Path, options: dict | None = None) -> BaseConnector:
    connector_class = CONNECTORS.get(source["provider"])
    if connector_class is None:
        raise ConnectorError(f"Unsupported provider: {source['provider']}")
    return connector_class(source, output_dir, options)

