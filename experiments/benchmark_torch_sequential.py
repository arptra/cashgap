#!/usr/bin/env python3
"""Stable sequential monthly benchmark for one Torch MLP.

Architecture:
  1. pandas/PyArrow prepare features once in the parent process;
  2. each test month runs in a fresh NumPy/PyTorch-only GPU worker;
  3. workers are strictly sequential and alternate between requested GPUs;
  4. completed fold artifacts are resumable and merged into Russian reports.

This mode is designed for EL8 hosts where mixed Arrow/CUDA processes terminate
with native SIGSEGV (exit code -11).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


if __name__ == "__main__" and os.environ.get("CASHGAP_EXTERNAL_DRIVER") != "1":
    try:
        from experiments.launch_training import detach_current_script
    except ImportError:
        from launch_training import detach_current_script
    if detach_current_script("benchmark", sys.argv[1:]):
        raise SystemExit(0)


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: отсутствуют библиотеки: {}".format(", ".join(missing)))
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
        print("Jupyter: %pip install {}".format(" ".join(missing)))
        print("После установки перезапустите kernel Jupyter.")
        raise SystemExit(2)
    print("Проверка зависимостей: библиотеки последовательного Torch benchmark установлены.")


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from experiments.benchmark_monthly_cashflow import build_monthly_dataset, target_matrix
    from experiments.monthly_reports_ru import write_russian_reports
    from experiments.train_cashflow_proxy import build_observed_daily
except ImportError:
    from benchmark_monthly_cashflow import build_monthly_dataset, target_matrix
    from monthly_reports_ru import write_russian_reports
    from train_cashflow_proxy import build_observed_daily


PREPARED_FORMAT_VERSION = 1
MODEL_CHOICES = ("torch_mlp_2_layers", "torch_mlp_3_layers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outflow", required=True)
    parser.add_argument("--inflow", required=True)
    parser.add_argument("--output-dir", default="artifacts/torch_sequential")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="torch_mlp_2_layers")
    parser.add_argument("--test-periods", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--layers", default="4096,2048",
        help="Ширины скрытых слоёв; high-load профиль по умолчанию: 4096,2048",
    )
    parser.add_argument(
        "--devices", default="cuda:0,cuda:1",
        help="GPU по очереди, например cuda:0,cuda:1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--activation", choices=("relu", "gelu", "silu"), default="relu")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Включить mixed precision на GPU")
    parser.add_argument(
        "--resume", action="store_true",
        help="Не пересчитывать подготовленные данные и уже завершённые периоды",
    )
    parser.add_argument(
        "--rebuild-prepared", action="store_true",
        help="Принудительно заново подготовить NumPy-данные",
    )
    return parser.parse_args()


def month_code(value: pd.Timestamp) -> int:
    return int(value.year * 100 + value.month)


def month_timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(value) // 100, month=int(value) % 100, day=1)


def input_fingerprint(path_value: str) -> Dict[str, object]:
    path = Path(path_value)
    payload: Dict[str, object] = {"path": str(path.resolve())}
    if path.is_file():
        stat = path.stat()
        payload.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return payload
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.parquet") if item.is_file())
        payload.update({
            "files": len(files),
            "total_size": sum(item.stat().st_size for item in files),
            "latest_mtime_ns": max((item.stat().st_mtime_ns for item in files), default=0),
        })
        return payload
    payload["missing"] = True
    return payload


def expected_prepared_key(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "format_version": PREPARED_FORMAT_VERSION,
        "outflow": input_fingerprint(args.outflow),
        "inflow": input_fingerprint(args.inflow),
        "max_inns": args.max_inns,
    }


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def prepared_is_valid(prepared: Path, expected_key: Dict[str, object]) -> bool:
    manifest_path = prepared / "manifest.json"
    required = ["features.npy", "targets.npy", "months.npy", "inns.npy"]
    if not manifest_path.exists() or any(not (prepared / name).exists() for name in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("prepared_key") != expected_key:
            return False
        rows = int(manifest["rows"])
        return all(len(np.load(str(prepared / name), mmap_mode="r", allow_pickle=False)) == rows for name in required)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def prepare_numpy_data(args: argparse.Namespace, prepared: Path) -> Dict[str, object]:
    prepared.mkdir(parents=True, exist_ok=True)
    expected_key = expected_prepared_key(args)
    if args.resume and not args.rebuild_prepared and prepared_is_valid(prepared, expected_key):
        print("Подготовленные NumPy-данные найдены, повторная обработка Parquet не нужна.")
        return json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))

    print("\n=== ЭТАП 1/2: ОДНОКРАТНАЯ ПОДГОТОВКА ДАННЫХ БЕЗ CUDA ===")
    observed_daily = build_observed_daily(args)
    monthly, features = build_monthly_dataset(observed_daily)
    del observed_daily
    feature_matrix = monthly[features].fillna(0.0).to_numpy(dtype=np.float32)
    targets = target_matrix(monthly).astype(np.float64, copy=False)
    months = np.asarray([month_code(pd.Timestamp(value)) for value in monthly["month"]], dtype=np.int32)
    inns = monthly["inn"].astype(str).to_numpy(dtype=str)
    if not np.isfinite(feature_matrix).all() or not np.isfinite(targets).all():
        raise ValueError("В подготовленных признаках или targets есть NaN/infinity.")
    atomic_npy(prepared / "features.npy", feature_matrix)
    atomic_npy(prepared / "targets.npy", targets)
    atomic_npy(prepared / "months.npy", months)
    atomic_npy(prepared / "inns.npy", inns)
    manifest: Dict[str, object] = {
        "prepared_key": expected_key,
        "rows": int(len(monthly)),
        "feature_count": int(len(features)),
        "features": list(features),
        "months": [int(value) for value in sorted(np.unique(months))],
        "inns": int(monthly["inn"].nunique()),
    }
    temporary = prepared / "manifest.json.inprogress"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(prepared / "manifest.json"))
    print("NumPy dataset | строк {:,} | ИНН {:,} | месяцев {} | признаков {}".format(
        manifest["rows"], manifest["inns"], len(manifest["months"]), manifest["feature_count"]
    ))
    del monthly, feature_matrix, targets, months, inns
    return manifest


def fold_specs(manifest: Dict[str, object], test_periods: int, min_train_months: int) -> List[Dict[str, int]]:
    months = [int(value) for value in manifest["months"]]
    required = min_train_months + test_periods + 1
    if len(months) < required:
        raise ValueError(
            "Нужно минимум {} месяцев для {} тестовых периодов; найдено {}.".format(
                required, test_periods, len(months)
            )
        )
    selected = months[-test_periods:]
    folds: List[Dict[str, int]] = []
    for fold_number, test_month in enumerate(selected, start=1):
        position = months.index(test_month)
        validation_month = months[position - 1]
        train_months = months[:position - 1]
        if len(train_months) < min_train_months:
            raise ValueError("Период {} имеет только {} месяцев обучения.".format(
                test_month, len(train_months)
            ))
        folds.append({
            "fold": fold_number,
            "train_start": train_months[0],
            "train_end": train_months[-1],
            "validation_month": validation_month,
            "test_month": test_month,
        })
    return folds


def worker_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, cpu_threads))
    env.update({
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "MALLOC_ARENA_MAX": "2",
        "CUDA_MODULE_LOADING": "LAZY",
    })
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512,garbage_collection_threshold:0.8",
    )
    return env


def nvidia_snapshot() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "nvidia-smi не найден."
    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "nvidia-smi завершился ошибкой: {}".format(error)


def worker_command(
    args: argparse.Namespace,
    prepared: Path,
    fold_output: Path,
    fold: Dict[str, int],
    device: str,
) -> List[str]:
    worker = Path(__file__).with_name("torch_fold_worker.py").resolve()
    command = [
        sys.executable,
        "-X", "faulthandler",
        "-u",
        str(worker),
        "--prepared-dir", str(prepared),
        "--output-dir", str(fold_output),
        "--model", args.model,
        "--fold", str(fold["fold"]),
        "--validation-month", str(fold["validation_month"]),
        "--test-month", str(fold["test_month"]),
        "--device", device,
        "--layers", args.layers,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed + fold["fold"]),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--activation", args.activation,
        "--patience", str(args.patience),
        "--cpu-threads", str(args.cpu_threads),
        "--mape-zero-floor", str(args.mape_zero_floor),
    ]
    if args.amp:
        command.append("--amp")
    if getattr(args, "score_only", False):
        command.append("--score-only")
    return command


def validate_fold_artifacts(
    fold_output: Path, require_predictions: bool = True,
) -> Tuple[bool, str]:
    marker = fold_output / "SUCCESS"
    metrics_path = fold_output / "metrics.json"
    window_path = fold_output / "window.json"
    predictions_path = fold_output / "predictions.npz"
    required_paths = [marker, metrics_path, window_path]
    if require_predictions:
        required_paths.append(predictions_path)
    if not all(path.exists() for path in required_paths):
        return False, "не хватает обязательных артефактов периода"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        window = json.loads(window_path.read_text(encoding="utf-8"))
        if require_predictions:
            with np.load(str(predictions_path), allow_pickle=False) as predictions:
                required = {
                    "inn", "actual_inflow", "predicted_inflow", "actual_outflow", "predicted_outflow"
                }
                if not required.issubset(predictions.files):
                    return False, "в predictions.npz отсутствуют массивы"
                lengths = {len(predictions[name]) for name in required}
                if len(lengths) != 1 or next(iter(lengths)) < 1:
                    return False, "массивы predictions пустые или разной длины"
        if len(metrics) != 2 or int(window["test_rows"]) < 1:
            return False, "metrics/window неполные"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, "OK"


def run_worker(
    command: Sequence[str], log_path: Path, cpu_threads: int,
    timeout_seconds: Optional[int] = None,
) -> int:
    snapshot_before = nvidia_snapshot()
    print("GPU до запуска: {}".format(snapshot_before), flush=True)
    print("Команда worker: {}".format(" ".join(shlex.quote(part) for part in command)), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("GPU ДО ЗАПУСКА\n{}\n\n".format(snapshot_before))
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=worker_env(cpu_threads),
        )
        assert process.stdout is not None

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)

        reader = threading.Thread(target=copy_output, name="torch-worker-log", daemon=True)
        reader.start()
        timeout_message: Optional[str] = None
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = 124
            timeout_message = "WORKER TIMEOUT после {} секунд; процесс остановлен.\n".format(
                timeout_seconds
            )
        reader.join(timeout=30)
        if timeout_message is not None:
            print(timeout_message, end="", flush=True)
            log_file.write(timeout_message)
        snapshot_after = nvidia_snapshot()
        log_file.write("\nGPU ПОСЛЕ ЗАПУСКА\n{}\n".format(snapshot_after))
    print("GPU после запуска: {}".format(snapshot_after), flush=True)
    return return_code


def signal_description(return_code: int) -> str:
    if return_code >= 0:
        return "exit code {}".format(return_code)
    number = -return_code
    try:
        return "signal {} ({})".format(number, signal.Signals(number).name)
    except ValueError:
        return "signal {}".format(number)


def failure_report(fold_output: Path, return_code: int, validation_error: str) -> Path:
    log_path = fold_output / "worker.log"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    report = fold_output / "failure_diagnostics.txt"
    report.write_text(
        "WORKER FAILURE\n"
        "{}\n"
        "artifact validation: {}\n"
        "\nLAST 120 LOG LINES\n{}\n".format(
            signal_description(return_code), validation_error, "\n".join(lines[-120:])
        ),
        encoding="utf-8",
    )
    return report


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


def merge_artifacts(
    output: Path,
    fold_outputs: Sequence[Path],
    folds: Sequence[Dict[str, int]],
    model: str,
) -> None:
    metric_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    predictions_path = output / "monthly_predictions.parquet"
    in_progress = output / "monthly_predictions.inprogress.parquet"
    writer: Optional[pq.ParquetWriter] = None
    try:
        for fold, fold_output in zip(folds, fold_outputs):
            metrics = json.loads((fold_output / "metrics.json").read_text(encoding="utf-8"))
            window = json.loads((fold_output / "window.json").read_text(encoding="utf-8"))
            test_month = month_timestamp(fold["test_month"])
            for row in metrics:
                row["test_month"] = test_month
                metric_rows.append(row)
            window_rows.append({
                "fold": fold["fold"],
                "train_start": month_timestamp(fold["train_start"]),
                "train_end": month_timestamp(fold["train_end"]),
                "validation_month": month_timestamp(fold["validation_month"]),
                "test_month": test_month,
                "train_rows": int(window["train_rows"]),
                "validation_rows": int(window["validation_rows"]),
                "test_rows": int(window["test_rows"]),
            })
            with np.load(str(fold_output / "predictions.npz"), allow_pickle=False) as values:
                prediction_frame = pd.DataFrame({
                    "fold": fold["fold"],
                    "test_month": test_month,
                    "inn": values["inn"].astype(str),
                    "model": model,
                    "actual_inflow": values["actual_inflow"],
                    "predicted_inflow": values["predicted_inflow"],
                    "actual_outflow": values["actual_outflow"],
                    "predicted_outflow": values["predicted_outflow"],
                })
            table = pa.Table.from_pandas(prediction_frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(in_progress, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("Нет прогнозов для объединения.")
    os.replace(str(in_progress), str(predictions_path))
    metrics_frame = pd.DataFrame(metric_rows)
    windows_frame = pd.DataFrame(window_rows)
    summary = stability_summary(metrics_frame)
    metrics_frame.to_csv(output / "monthly_fold_metrics.csv", index=False)
    windows_frame.to_csv(output / "monthly_fold_windows.csv", index=False)
    summary.to_csv(output / "monthly_stability_summary.csv", index=False)
    write_russian_reports(output, metrics_frame, summary, windows_frame, predictions_path)
    print("\n=== {} ПЕРИОДОВ ЗАВЕРШЕНЫ ПОСЛЕДОВАТЕЛЬНО ===".format(len(fold_outputs)))
    print((output / "бизнес_вывод.txt").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.test_periods < 1:
        raise ValueError("--test-periods должен быть положительным.")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or any(not item.startswith("cuda:") for item in devices):
        raise ValueError("--devices должен содержать CUDA-устройства, например cuda:0,cuda:1.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "prepared_numpy"
    manifest = prepare_numpy_data(args, prepared)
    folds = fold_specs(manifest, args.test_periods, args.min_train_months)
    workers_root = output / "sequential_folds"
    workers_root.mkdir(parents=True, exist_ok=True)
    print("\n=== ЭТАП 2/2: {} ПЕРИОДОВ СТРОГО ПО ОЧЕРЕДИ ===".format(len(folds)))
    print("Модель: {} | слои {} | batch {:,} | GPU по очереди: {}".format(
        args.model, args.layers, args.batch_size, ", ".join(devices)
    ))
    fold_outputs: List[Path] = []
    for index, fold in enumerate(folds):
        device = devices[index % len(devices)]
        fold_output = workers_root / "fold_{:02d}_{}".format(fold["fold"], fold["test_month"])
        fold_output.mkdir(parents=True, exist_ok=True)
        fold_outputs.append(fold_output)
        valid, reason = validate_fold_artifacts(fold_output)
        if args.resume and valid:
            print("\nПериод {}/{} {} уже готов — пропускаю.".format(
                fold["fold"], len(folds), fold["test_month"]
            ))
            continue
        print("\n--- ПЕРИОД {}/{} | тест {} | {} ---".format(
            fold["fold"], len(folds), fold["test_month"], device
        ))
        command = worker_command(args, prepared, fold_output, fold, device)
        return_code = run_worker(command, fold_output / "worker.log", args.cpu_threads)
        valid, reason = validate_fold_artifacts(fold_output)
        if return_code != 0 and valid:
            print(
                "ВНИМАНИЕ: worker завершился через {}, но полностью записанные артефакты "
                "проверены — период принят.".format(signal_description(return_code))
            )
            continue
        if return_code != 0 or not valid:
            report = failure_report(fold_output, return_code, reason)
            raise RuntimeError(
                "Период {} упал: {}. Диагностика: {}. Повторный запуск с --resume "
                "продолжит с этого периода.".format(
                    fold["test_month"], signal_description(return_code), report.resolve()
                )
            )
        print("Период {} завершён и проверен; CUDA-worker полностью остановлен.".format(
            fold["test_month"]
        ))

    merge_artifacts(output, fold_outputs, folds, args.model)
    config = vars(args).copy()
    config["execution"] = "strictly sequential pure NumPy/PyTorch workers"
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Результаты сохранены: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
