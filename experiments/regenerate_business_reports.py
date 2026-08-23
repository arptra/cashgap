#!/usr/bin/env python3
"""Rebuild Russian business reports from completed benchmark artifacts.

No model is trained and Parquet source transactions are not read again.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path


def _check_dependencies() -> None:
    required = {"pandas": "pandas", "pyarrow": "pyarrow"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if not missing:
        print("Проверка зависимостей: библиотеки бизнес-отчёта установлены.")
        return
    executable = shlex.quote(sys.executable)
    print("ОШИБКА: отсутствуют библиотеки: {}".format(", ".join(missing)))
    print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
    print("Jupyter: %pip install {}".format(" ".join(missing)))
    print("После установки перезапустите kernel Jupyter.")
    raise SystemExit(2)


if __name__ == "__main__":
    _check_dependencies()

import pandas as pd

try:
    from experiments.monthly_reports_ru import write_russian_reports
except ImportError:
    from monthly_reports_ru import write_russian_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True,
        help="Каталог завершённого benchmark с monthly_fold_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    paths = {
        "metrics": output / "monthly_fold_metrics.csv",
        "summary": output / "monthly_stability_summary.csv",
        "windows": output / "monthly_fold_windows.csv",
        "predictions": output / "monthly_predictions.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Нельзя пересобрать отчёт: отсутствуют технические результаты:\n- {}".format(
                "\n- ".join(missing)
            )
        )
    metrics = pd.read_csv(paths["metrics"])
    summary = pd.read_csv(paths["summary"])
    windows = pd.read_csv(paths["windows"])
    write_russian_reports(output, metrics, summary, windows, paths["predictions"])
    print("\n=== БИЗНЕС-ОТЧЁТ ПЕРЕСОБРАН БЕЗ ОБУЧЕНИЯ ===")
    for filename in (
        "бизнес_отчет.md",
        "01_рейтинг_моделей.csv",
        "02_качество_по_месяцам.csv",
        "03_окна_тестирования.csv",
        "04_чистый_поток_по_месяцам.csv",
        "отчет_прогнозы_по_инн.parquet",
    ):
        print(output / filename)


if __name__ == "__main__":
    main()
