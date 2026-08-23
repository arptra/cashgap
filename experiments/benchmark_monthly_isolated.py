#!/usr/bin/env python3
"""Run monthly folds in isolated processes and combine their artifacts.

Each child process still starts all selected models in parallel. Process-level
isolation guarantees that CUDA contexts, native thread pools and allocator
caches are released by the operating system after every test month.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _check_dependencies() -> None:
    required = {"pandas": "pandas", "pyarrow": "pyarrow"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if not missing:
        print("Проверка зависимостей: библиотеки изолированного runner установлены.")
        return
    executable = shlex.quote(sys.executable)
    print("ОШИБКА: отсутствуют библиотеки runner: {}".format(", ".join(missing)))
    print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
    print("Jupyter: %pip install {}".format(" ".join(missing)))
    print("После установки перезапустите kernel Jupyter.")
    raise SystemExit(2)


if __name__ == "__main__":
    _check_dependencies()

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from experiments.monthly_reports_ru import write_russian_reports
except ImportError:
    from monthly_reports_ru import write_russian_reports


DEFAULT_MODELS = ",".join([
    "trailing_mean",
    "linear_regression",
    "gradient_boosting",
    "torch_mlp_2_layers",
    "torch_mlp_3_layers",
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outflow", required=True)
    parser.add_argument("--inflow", required=True)
    parser.add_argument("--output-dir", default="artifacts/monthly_benchmark_isolated")
    parser.add_argument("--test-periods", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--mlp2-device", default="cuda:0")
    parser.add_argument("--mlp3-device", default="cuda:1")
    parser.add_argument("--mlp2-layers", default="512,256")
    parser.add_argument("--mlp3-layers", default="768,512,256")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--boosting-iterations", type=int, default=250)
    parser.add_argument("--mlp-params", default=None)
    parser.add_argument("--boosting-params", default=None)
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    parser.add_argument(
        "--save-model", default=None,
        help="После benchmark обучить на всех данных и сохранить одну выбранную модель",
    )
    parser.add_argument(
        "--model-output-dir", default=None,
        help="Каталог модели; по умолчанию OUTPUT_DIR/saved_model",
    )
    parser.add_argument("--forecast-months", type=int, default=12)
    parser.add_argument(
        "--save-model-device", default=None,
        help="Устройство финального обучения; по умолчанию GPU выбранной MLP",
    )
    return parser.parse_args()


def child_command(args: argparse.Namespace, fold_output: Path, offset: int) -> List[str]:
    benchmark = Path(__file__).with_name("benchmark_monthly_cashflow.py").resolve()
    command = [
        sys.executable,
        "-u",
        str(benchmark),
        "--outflow", args.outflow,
        "--inflow", args.inflow,
        "--output-dir", str(fold_output),
        "--test-periods", "1",
        "--test-offset", str(offset),
        "--min-train-months", str(args.min_train_months),
        "--models", args.models,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--mlp2-device", args.mlp2_device,
        "--mlp3-device", args.mlp3_device,
        "--mlp2-layers", args.mlp2_layers,
        "--mlp3-layers", args.mlp3_layers,
        "--cpu-threads", str(args.cpu_threads),
        "--boosting-iterations", str(args.boosting_iterations),
        "--mape-zero-floor", str(args.mape_zero_floor),
        "--technical-reports-only",
    ]
    if args.parallel:
        command.append("--parallel")
    optional = {
        "--max-inns": args.max_inns,
        "--mlp-params": args.mlp_params,
        "--boosting-params": args.boosting_params,
    }
    for option, value in optional.items():
        if value is not None:
            command.extend([option, str(value)])
    return command


def run_child(command: List[str], log_path: Path) -> None:
    print("Команда дочернего процесса: {}".format(
        " ".join(shlex.quote(part) for part in command)
    ), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        hint = ""
        if return_code == -11:
            hint = (
                " Это нативный SIGSEGV. Для одной Torch MLP используйте "
                "experiments/benchmark_torch_sequential.py: он не смешивает PyArrow и CUDA."
            )
        raise RuntimeError(
            "Дочерний процесс периода завершился с кодом {}. Постоянный лог: {}.{}".format(
                return_code, log_path.resolve(), hint
            )
        )


def export_command(args: argparse.Namespace, model_output: Path) -> List[str]:
    exporter = Path(__file__).with_name("export_monthly_model.py").resolve()
    if args.save_model == "torch_mlp_3_layers":
        default_device = args.mlp3_device
    else:
        default_device = args.mlp2_device
    command = [
        sys.executable,
        "-u",
        str(exporter),
        "--outflow", args.outflow,
        "--inflow", args.inflow,
        "--output-dir", str(model_output),
        "--model", args.save_model,
        "--forecast-months", str(args.forecast_months),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.save_model_device or default_device,
        "--mlp2-layers", args.mlp2_layers,
        "--mlp3-layers", args.mlp3_layers,
        "--cpu-threads", str(args.cpu_threads),
        "--boosting-iterations", str(args.boosting_iterations),
    ]
    optional = {
        "--max-inns": args.max_inns,
        "--mlp-params": args.mlp_params,
        "--boosting-params": args.boosting_params,
    }
    for option, value in optional.items():
        if value is not None:
            command.extend([option, str(value)])
    return command


def append_prediction_batches(
    source: Path, fold_number: int, writer: Optional[pq.ParquetWriter], destination: Path,
) -> pq.ParquetWriter:
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        fold_index = table.schema.get_field_index("fold")
        fold_type = table.schema.field(fold_index).type
        table = table.set_column(
            fold_index, "fold", pa.array([fold_number] * len(table), type=fold_type)
        )
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
        writer.write_table(table)
    if writer is None:
        raise RuntimeError("No prediction batches found in {}".format(source))
    return writer


def stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["model", "flow"], as_index=False).agg(
        aggregate_mape_mean_percent=("aggregate_mape_percent", "mean"),
        aggregate_mape_std_percent=("aggregate_mape_percent", "std"),
        aggregate_mape_worst_percent=("aggregate_mape_percent", "max"),
        wape_mean_percent=("wape_percent", "mean"),
        company_mape_mean_percent=("company_mape_nonzero_percent", "mean"),
        bias_mean_percent=("bias_percent", "mean"),
        folds=("fold", "nunique"),
    ).sort_values(["flow", "aggregate_mape_mean_percent"])


def main() -> None:
    args = parse_args()
    if args.test_periods < 1:
        raise ValueError("--test-periods должен быть положительным.")
    if args.save_model and args.forecast_months < 1:
        raise ValueError("--forecast-months должен быть положительным.")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.save_model and args.save_model not in selected_models:
        raise ValueError(
            "--save-model={} отсутствует в --models. Сначала честно протестируйте сохраняемую модель.".format(
                args.save_model
            )
        )
    output = Path(args.output_dir)
    children = output / "isolated_folds"
    children.mkdir(parents=True, exist_ok=True)
    in_progress = output / "monthly_predictions.inprogress.parquet"
    final_predictions = output / "monthly_predictions.parquet"
    if in_progress.exists():
        in_progress.unlink()

    metric_frames: List[pd.DataFrame] = []
    window_frames: List[pd.DataFrame] = []
    writer: Optional[pq.ParquetWriter] = None
    completed = False
    try:
        offsets = list(range(args.test_periods - 1, -1, -1))
        for fold_number, offset in enumerate(offsets, start=1):
            fold_output = children / "fold_{:02d}_offset_{:02d}".format(fold_number, offset)
            fold_output.mkdir(parents=True, exist_ok=True)
            log_path = fold_output / "child.log"
            print("\n=== ИЗОЛИРОВАННЫЙ ПЕРИОД {}/{} | смещение {} ===".format(
                fold_number, args.test_periods, offset
            ), flush=True)
            run_child(child_command(args, fold_output, offset), log_path)

            metrics = pd.read_csv(fold_output / "monthly_fold_metrics.csv")
            windows = pd.read_csv(fold_output / "monthly_fold_windows.csv")
            metrics["fold"] = fold_number
            windows["fold"] = fold_number
            metric_frames.append(metrics)
            window_frames.append(windows)
            writer = append_prediction_batches(
                fold_output / "monthly_predictions.parquet", fold_number, writer, in_progress
            )
            print("Артефакты периода {} объединены; память дочернего процесса полностью освобождена.".format(
                fold_number
            ), flush=True)
        completed = True
    finally:
        if writer is not None:
            writer.close()

    if not completed:
        raise RuntimeError("Изолированный benchmark не завершил все тестовые периоды.")
    os.replace(str(in_progress), str(final_predictions))
    metrics = pd.concat(metric_frames, ignore_index=True)
    windows = pd.concat(window_frames, ignore_index=True)
    summary = stability_summary(metrics)
    metrics.to_csv(output / "monthly_fold_metrics.csv", index=False)
    windows.to_csv(output / "monthly_fold_windows.csv", index=False)
    summary.to_csv(output / "monthly_stability_summary.csv", index=False)
    write_russian_reports(output, metrics, summary, windows, final_predictions)
    config: Dict[str, object] = vars(args).copy()
    config["execution"] = "one isolated child process per test month"
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== СТАБИЛЬНОСТЬ ЗА {} ИЗОЛИРОВАННЫХ ТЕСТОВЫХ МЕСЯЦЕВ ===".format(
        args.test_periods
    ))
    print((output / "бизнес_вывод.txt").read_text(encoding="utf-8"))
    print("\nРусские отчёты и технические файлы сохранены: {}".format(output.resolve()))
    if args.save_model:
        model_output = Path(args.model_output_dir) if args.model_output_dir else output / "saved_model"
        model_output.mkdir(parents=True, exist_ok=True)
        print("\n=== ФИНАЛЬНОЕ ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛИ {} ===".format(
            args.save_model
        ))
        run_child(export_command(args, model_output), output / "model_export.log")
        print("Модель и таблица прогнозов для API сохранены: {}".format(
            model_output.resolve()
        ))


if __name__ == "__main__":
    main()
