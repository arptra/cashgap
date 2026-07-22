from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import huggingface_hub

from app.api import models as models_api
from app.model_plugins import create_model_plugin
from app.models_registry import installer as model_installer
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
    assert by_id["chronos_bolt_tiny"].bundled is True


def test_missing_lightgbm_is_reported_instead_of_claiming_installed(monkeypatch) -> None:
    spec = get_model_spec("lightgbm_cashgap")
    assert spec
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package=None):
        return None if name == "lightgbm" else original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    report = create_model_plugin(spec).check_environment()
    assert report.status == ModelStatus.NOT_INSTALLED
    assert report.installed is False
    assert "requirements-lightgbm.txt" in (report.install_command or "")


def test_duplicate_model_install_reuses_active_job(monkeypatch) -> None:
    monkeypatch.setattr(
        models_api,
        "list_jobs",
        lambda: [
            {
                "id": "job_existing",
                "job_type": "model_install",
                "status": "installing",
                "options": {"model_id": "chronos_bolt_small"},
            }
        ],
    )

    def fail_if_created(*args, **kwargs):
        raise AssertionError("a duplicate installation job must not be created")

    monkeypatch.setattr(models_api, "create_job", fail_if_created)
    result = models_api.install("chronos_bolt_small")
    assert result == {"job_id": "job_existing", "model_id": "chronos_bolt_small", "status": "installing"}


def test_bundled_model_cannot_be_uninstalled() -> None:
    spec = get_model_spec("chronos_bolt_tiny")
    assert spec
    try:
        uninstall_model(spec)
    except RuntimeError as error:
        assert "Bundled offline model" in str(error)
    else:
        raise AssertionError("bundled model uninstall must be rejected")


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

    attempts = 0

    def fake_snapshot_download(**kwargs):
        nonlocal attempts
        attempts += 1
        assert kwargs["revision"] == "fixed-revision-123"
        assert "*.safetensors" in kwargs["allow_patterns"]
        assert "*.bin" in kwargs["ignore_patterns"]
        if attempts == 1:
            raise ConnectionError("transient proxy failure")
        root = Path(kwargs["local_dir"])
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"safe")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(model_installer.time, "sleep", lambda _: None)
    result = install_pretrained(spec)
    assert attempts == 2
    assert result["installed"] is True
    assert result["revision"] == "fixed-revision-123"
    assert not list(Path(result["path"]).rglob("*.bin"))
    marker = json.loads((Path(result["path"]) / ".cashgap-model.json").read_text(encoding="utf-8"))
    assert marker["safe_formats_only"] is True
    uninstall_model(spec)
    assert cache_metadata(spec)["installed"] is False
