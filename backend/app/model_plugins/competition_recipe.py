from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from app.model_plugins.base import BaseModelPlugin
from app.models_registry.installer import cache_metadata
from app.models_registry.schemas import EnvironmentReport, ModelStatus


def kaggle_authenticated() -> bool:
    return bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".kaggle" / "access_token").exists()
    )


class CompetitionRecipePlugin(BaseModelPlugin):
    optional_dependency: str | None = None

    def check_environment(self) -> EnvironmentReport:
        info = cache_metadata(self.spec)
        dependency_ok = not self.optional_dependency or importlib.util.find_spec(self.optional_dependency) is not None
        if info["installed"] and dependency_ok:
            return EnvironmentReport(status=ModelStatus.INSTALLED, message="Notebook подключён как источник; внутренний adapter готов", installed=True, size_bytes=info["size_bytes"], revision=info.get("revision"))
        if not dependency_ok:
            return EnvironmentReport(status=ModelStatus.NOT_INSTALLED, message=f"Не установлена библиотека {self.optional_dependency}", installed=bool(info["installed"]), dependency_installed=False, size_bytes=info["size_bytes"], install_command=self.spec.install_command)
        if not kaggle_authenticated():
            return EnvironmentReport(status=ModelStatus.AUTH_REQUIRED, message="Нужна авторизация Kaggle для получения исходного notebook", installed=False, dependency_installed=dependency_ok, install_command=self.spec.install_command)
        return EnvironmentReport(status=ModelStatus.AVAILABLE, message="Можно подключить notebook через официальный Kaggle CLI", installed=False)

    def run(self, dataset, options):
        if self.spec.type == "competition_recipe" and not cache_metadata(self.spec)["installed"]:
            raise RuntimeError("Сначала подключите исходный Kaggle notebook; скачанный код не будет исполняться")
        return super().run(dataset, options)
