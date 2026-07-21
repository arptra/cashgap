from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub

from app.model_plugins import create_model_plugin
from app.models_registry.installer import cache_metadata, install_pretrained, uninstall_model
from app.models_registry.registry import get_model_spec, load_model_specs
from app.models_registry.schemas import ModelSpec, ModelStatus


def test_model_registry_has_all_three_model_types_and_required_adapters() -> None:
    specs = load_model_specs()
    assert {spec.type for spec in specs} == {
        "competition_recipe",
        "pretrained_model",
        "local_trainable_model",
    }
    by_id = {spec.id: spec for spec in specs}
    assert {
        "shell_catboost_darts_3rd",
        "shell_prophet_8th",
        "chronos_bolt_tiny",
        "chronos_bolt_small",
        "chronos_2",
        "timesfm_2_5",
        "seasonal_naive_local",
        "logistic_cashgap",
        "catboost_cashgap",
        "lightgbm_cashgap",
        "random_forest_cashgap",
    } <= set(by_id)
    assert by_id["chronos_2"].plugin != by_id["chronos_bolt_tiny"].plugin


def test_optional_dependencies_return_explicit_environment_status() -> None:
    chronos = get_model_spec("chronos_bolt_tiny")
    timesfm = get_model_spec("timesfm_2_5")
    prophet = get_model_spec("shell_prophet_8th")
    assert chronos and timesfm and prophet
    assert create_model_plugin(chronos).check_environment().status in {
        ModelStatus.AVAILABLE,
        ModelStatus.INSTALLED,
        ModelStatus.NOT_INSTALLED,
    }
    assert create_model_plugin(timesfm).check_environment().status in {
        ModelStatus.AVAILABLE,
        ModelStatus.INSTALLED,
        ModelStatus.NOT_INSTALLED,
    }
    prophet_report = create_model_plugin(prophet).check_environment()
    if not prophet_report.dependency_installed:
        assert prophet_report.status == ModelStatus.NOT_INSTALLED
        assert "pip install prophet" in (prophet_report.install_command or "")


def test_cash_gap_model_explains_missing_historical_target() -> None:
    spec = get_model_spec("logistic_cashgap")
    assert spec
    report = create_model_plugin(spec).check_compatibility(
        {
            "status": "completed",
            "summary": {"n_clients": 12, "n_months": 12, "has_cash_gap_target": False},
            "paths": {"monthly_aggregates": "/tmp/monthly.parquet"},
        },
        target="cash_gap_next_month",
        series_level="client",
        horizon=1,
        min_history=6,
    )
    assert report.compatible is False
    assert "Для обучения классификатора необходим исторический признак кассового разрыва" in report.reasons


def test_pretrained_install_is_revision_pinned_and_marks_only_complete_snapshots(monkeypatch) -> None:
    spec = ModelSpec(
        id="fake_safe_model",
        name="Fake safe model",
        type="pretrained_model",
        provider="huggingface",
        task="cash_flow_forecasting",
        plugin="chronos_bolt",
        model_id="example/safe-model",
        requires_training=False,
        cpu_supported=True,
        compatible_targets=["net_flow"],
        license="Apache-2.0",
    )

    class Info:
        sha = "fixed-revision-123"

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def model_info(self, model_id):
            assert model_id == "example/safe-model"
            return Info()

    def fake_snapshot_download(**kwargs):
        assert kwargs["revision"] == "fixed-revision-123"
        assert "*.safetensors" in kwargs["allow_patterns"]
        assert "*.bin" in kwargs["ignore_patterns"]
        root = Path(kwargs["local_dir"])
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"safe")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = install_pretrained(spec)
    assert result["installed"] is True
    assert result["revision"] == "fixed-revision-123"
    assert not list(Path(result["path"]).rglob("*.bin"))
    marker = json.loads((Path(result["path"]) / ".cashgap-model.json").read_text(encoding="utf-8"))
    assert marker["safe_formats_only"] is True
    uninstall_model(spec)
    assert cache_metadata(spec)["installed"] is False
