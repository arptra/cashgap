from __future__ import annotations

import json
import traceback
from pathlib import Path

from app.canonical.compatibility import build_compatibility_report
from app.canonical.profiling import profile_canonical
from app.config import NORMALIZED_DIR
from app.ml.synthetic import SyntheticConfig, generate_synthetic_dataset
from app.storage.database import update_dataset


def generate_dataset_job(dataset_id: str, config_values: dict) -> None:
    output_dir = NORMALIZED_DIR / dataset_id
    try:
        update_dataset(dataset_id, status="running", error=None)
        config = SyntheticConfig.from_dict(config_values)
        summary = generate_synthetic_dataset(config, output_dir, source_dataset=dataset_id)
        profile = profile_canonical(
            Path(summary["files"]["monthly_aggregates"]),
            liquidity_path=Path(summary["files"]["liquidity"]),
            target_path=Path(summary["files"]["target"]),
            extra_paths=[Path(summary["files"]["daily_hidden"])],
        )
        synthetic_source = {
            "id": "synthetic",
            "supported_tasks": ["cash_gap_classification", "flow_forecasting", "proxy_risk"],
            "cash_gap_target": True,
        }
        summary.update(
            {
                **profile,
                "stage": "normalized",
                "source_id": "synthetic",
                "source_provider": "synthetic",
                "mapping": {"generator": "daily trajectories -> canonical monthly categories"},
                "quality_flags": [],
                "compatibility": build_compatibility_report(profile, synthetic_source),
            }
        )
        (output_dir / "config.json").write_text(
            json.dumps(config_values, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        update_dataset(
            dataset_id,
            status="completed",
            summary_json=summary,
            paths_json=summary["files"],
            error=None,
        )
    except Exception:
        update_dataset(dataset_id, status="failed", error=traceback.format_exc())
