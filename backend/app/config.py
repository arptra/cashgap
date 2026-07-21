from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
CODE_ROOT = BACKEND_DIR.parent
REPO_ROOT = Path(os.getenv("CASHGAP_ROOT", BACKEND_DIR.parent)).resolve()
DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
UPLOADED_DIR = DATA_DIR / "uploaded"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = REPO_ROOT / "cache"
LOGS_DIR = REPO_ROOT / "logs"
CONFIG_DIR = Path(os.getenv("CASHGAP_CONFIG_DIR", CODE_ROOT / "config")).resolve()
DATASET_REGISTRY_PATH = CONFIG_DIR / "datasets.yaml"
MODEL_REGISTRY_PATH = CONFIG_DIR / "models.yaml"
MODEL_CACHE_DIR = CACHE_DIR / "models"
EXTERNAL_DIR = REPO_ROOT / "external"
COMPETITION_SOURCES_DIR = EXTERNAL_DIR / "competition_sources"
DB_PATH = Path(os.getenv("CASHGAP_DB_PATH", BACKEND_DIR / "cashgap.db")).resolve()


def ensure_directories() -> None:
    for directory in (
        SYNTHETIC_DIR, UPLOADED_DIR, RAW_DIR, NORMALIZED_DIR, MODELS_DIR,
        RUNS_DIR, CACHE_DIR, LOGS_DIR, MODEL_CACHE_DIR, COMPETITION_SOURCES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
