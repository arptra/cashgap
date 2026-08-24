#!/usr/bin/env python3
"""Crash-resistant Optuna tuning for monthly Torch MLP models.

The dispatcher never imports CUDA. It prepares NumPy arrays once, asks Optuna
for a wave of parameters, then assigns at most one pure Torch subprocess to
each GPU. Every tuning fold gets a fresh process. Failed native workers mark
only their trial as failed; the study continues and remains resumable in SQLite.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


if __name__ == "__main__" and os.environ.get("CASHGAP_EXTERNAL_DRIVER") != "1":
    try:
        from experiments.launch_training import detach_current_script
    except ImportError:
        from launch_training import detach_current_script
    if detach_current_script("autotune", sys.argv[1:]):
        raise SystemExit(0)


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sklearn": "scikit-learn",
        "optuna": "optuna==3.6.2" if sys.version_info[:2] == (3, 8) else "optuna",
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: устойчивому автотюнингу не хватает: {}".format(", ".join(missing)))
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
        print("Jupyter: %pip install {}".format(" ".join(missing)))
        print("После установки перезапустите kernel Jupyter.")
        raise SystemExit(2)
    print("Проверка зависимостей: библиотеки устойчивого автотюнинга установлены.")


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

try:
    from experiments.benchmark_torch_sequential import (
        failure_report,
        fold_specs,
        prepare_numpy_data,
        run_worker,
        signal_description,
        validate_fold_artifacts,
        worker_command,
    )
    from experiments.monthly_objective import (
        MONTHLY_OBJECTIVE_NAME,
        MONTHLY_OBJECTIVE_VERSION,
    )
except ImportError:
    from benchmark_torch_sequential import (
        failure_report,
        fold_specs,
        prepare_numpy_data,
        run_worker,
        signal_description,
        validate_fold_artifacts,
        worker_command,
    )
    from monthly_objective import MONTHLY_OBJECTIVE_NAME, MONTHLY_OBJECTIVE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outflow", required=True)
    parser.add_argument("--inflow", required=True)
    parser.add_argument("--output-dir", default="artifacts/cashflow_tuning_stable")
    parser.add_argument("--trials", type=int, default=30, help="Новых trials в этом запуске")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--tuning-periods", type=int, default=3)
    parser.add_argument("--holdout-test-periods", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inns", type=int, default=None)
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, choices=(512, 1024, 2048, 4096), default=2048)
    parser.add_argument("--max-layers", type=int, choices=(2, 3, 4, 5, 6), default=6)
    parser.add_argument(
        "--worker-timeout-minutes", type=int, default=180,
        help="Остановить зависший fold и продолжить подбор (по умолчанию 180 минут)",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--sequential-trials", action="store_true",
        help="Не запускать два CUDA-worker одновременно; GPU всё равно чередуются",
    )
    parser.add_argument("--rebuild-prepared", action="store_true")
    return parser.parse_args()


def preflight_devices(devices: Sequence[str]) -> None:
    """Validate CUDA in short-lived children so the dispatcher never imports it."""
    check = (
        "import os, sys, torch; "
        "d=torch.device(sys.argv[1]); "
        "assert torch.cuda.is_available(), 'torch.cuda.is_available() == False'; "
        "assert d.index is not None and d.index < torch.cuda.device_count(), "
        "'requested GPU is not visible'; "
        "torch.cuda.set_device(d.index); "
        "x=torch.ones(1024, device=d); torch.cuda.synchronize(d); "
        "p=torch.cuda.get_device_properties(d.index); "
        "print('{} | {} | {:.1f} GiB'.format(d, p.name, p.total_memory/1024**3), flush=True); "
        "os._exit(0)"
    )
    print("\nПроверка CUDA в изолированных процессах:", flush=True)
    for device in devices:
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-c", check, device],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("CUDA preflight для {} завис более чем на 60 секунд.".format(
                device
            )) from error
        details = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "CUDA preflight для {} завершился с {}. {}".format(
                    device, signal_description(completed.returncode), details
                )
            )
        print("- {}".format(details), flush=True)


def tuning_folds(
    manifest: Dict[str, object], tuning_periods: int, holdout_periods: int,
    min_train_months: int,
) -> Tuple[List[Dict[str, int]], List[int]]:
    all_months = [int(value) for value in manifest["months"]]
    if holdout_periods < 1 or len(all_months) <= holdout_periods:
        raise ValueError("Недостаточно месяцев, чтобы защитить {} holdout-периодов.".format(
            holdout_periods
        ))
    tuning_months = all_months[:-holdout_periods]
    protected = all_months[-holdout_periods:]
    tuning_manifest = dict(manifest)
    tuning_manifest["months"] = tuning_months
    return fold_specs(tuning_manifest, tuning_periods, min_train_months), protected


def width_choices(max_width: int) -> List[int]:
    return [value for value in (128, 256, 512, 1024, 2048, 4096) if value <= max_width]


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, object]:
    layer_count = trial.suggest_int("n_layers", 1, args.max_layers)
    first_width = trial.suggest_categorical("first_width", width_choices(args.max_width))
    shrink = trial.suggest_categorical("width_shrink", [0.5, 0.75, 1.0])
    layers = [max(32, int(first_width * (shrink ** index))) for index in range(layer_count)]
    params: Dict[str, object] = {
        "layers": layers,
        "batch_size": trial.suggest_categorical(
            "batch_size", [4096, 8192, 16384, 32768, 65536]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        "patience": trial.suggest_int("patience", 8, 18),
    }
    trial.set_user_attr("layers", layers)
    return params


def trial_worker_args(
    base: argparse.Namespace, params: Dict[str, object], trial_number: int,
) -> Namespace:
    return Namespace(
        model="torch_mlp_tuned",
        layers=",".join(str(value) for value in params["layers"]),
        epochs=base.epochs,
        batch_size=int(params["batch_size"]),
        seed=base.seed + trial_number * 1000,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        dropout=float(params["dropout"]),
        activation=str(params["activation"]),
        patience=int(params["patience"]),
        cpu_threads=base.cpu_threads,
        mape_zero_floor=base.mape_zero_floor,
        amp=base.amp,
        score_only=True,
    )


def _preflight_one_worker(
    base: argparse.Namespace,
    prepared: Path,
    fold: Dict[str, int],
    device: str,
    output: Path,
    label: str,
) -> Tuple[bool, str]:
    params: Dict[str, object] = {
        "layers": [128, 64],
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.10,
        "activation": "relu",
        "patience": 2,
    }
    worker_args = trial_worker_args(base, params, -1)
    worker_args.epochs = 2
    worker_args.seed = base.seed
    fold_output = output / label / device.replace(":", "_")
    fold_output.mkdir(parents=True, exist_ok=True)
    command = worker_command(worker_args, prepared, fold_output, fold, device)
    return_code = run_worker(
        command, fold_output / "worker.log", base.cpu_threads,
        timeout_seconds=min(base.worker_timeout_minutes * 60, 20 * 60),
    )
    valid, reason = validate_fold_artifacts(fold_output, require_predictions=False)
    if return_code == 0 and valid:
        return True, "OK"
    if return_code != 0 and valid:
        return True, "{} после полной записи".format(signal_description(return_code))
    report = failure_report(fold_output, return_code, reason)
    return False, "{} | {}".format(signal_description(return_code), report.resolve())


def training_preflight(
    base: argparse.Namespace,
    prepared: Path,
    fold: Dict[str, int],
    devices: Sequence[str],
    output: Path,
) -> bool:
    """Prove that a real worker trains before Optuna is allowed to create trials."""
    root = output / "_training_preflight" / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    print("\n=== PREFLIGHT: 2 ЭПОХИ РЕАЛЬНОГО ОБУЧЕНИЯ НА КАЖДОЙ GPU ===", flush=True)

    def sequential_check(label: str) -> Tuple[bool, List[str]]:
        failures: List[str] = []
        for device in devices:
            ok, detail = _preflight_one_worker(
                base, prepared, fold, device, root, "{}_{}".format(label, device.replace(":", "_"))
            )
            print("Preflight {} | {} | {}".format(device, "OK" if ok else "FAILED", detail), flush=True)
            if not ok:
                failures.append("{}: {}".format(device, detail))
        return not failures, failures

    sequential_ok, sequential_failures = sequential_check("sequential")
    if not sequential_ok and base.amp:
        print(
            "AMP дал сбой в реальном preflight. Автоматически повторяю без AMP.",
            flush=True,
        )
        base.amp = False
        sequential_ok, sequential_failures = sequential_check("sequential_no_amp")
    if not sequential_ok:
        raise RuntimeError(
            "Реальный Torch-worker не проходит даже безопасный последовательный preflight; "
            "Optuna не запущена и новые failed trials не созданы. {}".format(
                " | ".join(sequential_failures)
            )
        )

    if base.sequential_trials or len(devices) < 2:
        print("Autotune будет выполнять trials по одному, чередуя GPU.", flush=True)
        return False

    print("\nПроверка двух одновременных CUDA-worker...", flush=True)
    concurrent_results: List[Tuple[str, bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(
                _preflight_one_worker,
                base,
                prepared,
                fold,
                device,
                root,
                "concurrent_{}".format(device.replace(":", "_")),
            ): device
            for device in devices
        }
        for future in concurrent.futures.as_completed(futures):
            device = futures[future]
            try:
                ok, detail = future.result()
            except Exception as error:
                ok, detail = False, repr(error)
            concurrent_results.append((device, ok, detail))
            print("Concurrent preflight {} | {} | {}".format(
                device, "OK" if ok else "FAILED", detail
            ), flush=True)
    if all(ok for _, ok, _ in concurrent_results):
        print("Два одновременных CUDA-worker проверены: параллельный режим разрешён.", flush=True)
        return True
    print(
        "Одновременный CUDA-запуск нестабилен. Trials будут идти по одному с "
        "чередованием GPU; это предотвращает повторение SIGSEGV.",
        flush=True,
    )
    return False


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_trial(
    base: argparse.Namespace,
    prepared: Path,
    folds: Sequence[Dict[str, int]],
    trial_number: int,
    params: Dict[str, object],
    device: str,
    trials_root: Path,
) -> Dict[str, object]:
    trial_output = trials_root / "trial_{:05d}".format(trial_number)
    trial_output.mkdir(parents=True, exist_ok=True)
    atomic_json(trial_output / "params.json", {
        "trial": trial_number,
        "device": device,
        "params": params,
    })
    worker_args = trial_worker_args(base, params, trial_number)
    fold_scores: List[float] = []
    fold_details: List[Dict[str, object]] = []
    for fold in folds:
        fold_output = trial_output / "fold_{:02d}_{}".format(fold["fold"], fold["test_month"])
        fold_output.mkdir(parents=True, exist_ok=True)
        command = worker_command(worker_args, prepared, fold_output, fold, device)
        print("\n[TUNING] trial {} | fold {}/{} | {} | layers {} | batch {}".format(
            trial_number, fold["fold"], len(folds), device, params["layers"], params["batch_size"]
        ), flush=True)
        return_code = run_worker(
            command,
            fold_output / "worker.log",
            base.cpu_threads,
            timeout_seconds=getattr(base, "worker_timeout_minutes", 180) * 60,
        )
        valid, reason = validate_fold_artifacts(fold_output, require_predictions=False)
        if return_code != 0 and valid:
            print("Trial {} fold {}: {} после полной записи — результат принят.".format(
                trial_number, fold["fold"], signal_description(return_code)
            ), flush=True)
        elif return_code != 0 or not valid:
            report = failure_report(fold_output, return_code, reason)
            result = {
                "status": "failed",
                "trial": trial_number,
                "device": device,
                "failed_fold": fold["fold"],
                "return_code": return_code,
                "signal": signal_description(return_code),
                "reason": reason,
                "diagnostics": str(report.resolve()),
                "params": params,
                "completed_fold_scores": fold_scores,
            }
            atomic_json(trial_output / "trial_result.json", result)
            return result
        metrics = json.loads((fold_output / "metrics.json").read_text(encoding="utf-8"))
        score = float(np.mean([float(row["aggregate_mape"]) for row in metrics]))
        fold_scores.append(score)
        fold_details.append({
            "fold": fold["fold"],
            "test_month": fold["test_month"],
            "score": score,
        })
    result = {
        "status": "complete",
        "trial": trial_number,
        "device": device,
        "objective": float(np.mean(fold_scores)),
        "objective_percent": float(np.mean(fold_scores) * 100),
        "folds": fold_details,
        "params": params,
    }
    atomic_json(trial_output / "trial_result.json", result)
    return result


def completed_trials(study: optuna.Study) -> List[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def display_month(value: object) -> str:
    text_value = str(value)
    return "{}-{}".format(text_value[:4], text_value[4:6]) if len(text_value) == 6 else text_value


def write_reports(
    study: optuna.Study,
    output: Path,
    folds: Sequence[Dict[str, int]],
    protected: Sequence[int],
    devices: Sequence[str],
) -> None:
    trials_frame = study.trials_dataframe()
    trials_frame.to_csv(output / "trials.csv", index=False)
    russian = trials_frame.rename(columns={
        "number": "Номер варианта",
        "value": "Целевая ошибка",
        "datetime_start": "Время начала",
        "datetime_complete": "Время завершения",
        "duration": "Длительность",
        "state": "Состояние",
        "params_n_layers": "Количество слоёв",
        "params_first_width": "Ширина первого слоя",
        "params_width_shrink": "Коэффициент сужения слоёв",
        "params_batch_size": "Размер пакета",
        "params_learning_rate": "Скорость обучения",
        "params_weight_decay": "Регуляризация весов",
        "params_dropout": "Доля отключения нейронов",
        "params_activation": "Функция активации",
        "params_patience": "Ожидание ранней остановки",
        "user_attrs_layers": "Структура слоёв",
    })
    if "Состояние" in russian.columns:
        russian["Состояние"] = russian["Состояние"].replace({
            "COMPLETE": "ЗАВЕРШЕН",
            "FAIL": "ОШИБКА",
            "RUNNING": "ВЫПОЛНЯЕТСЯ",
            "PRUNED": "ОСТАНОВЛЕН_ДОСРОЧНО",
        })
    if "Целевая ошибка" in russian.columns:
        russian["Целевая ошибка, %"] = russian.pop("Целевая ошибка") * 100.0
    russian.to_csv(
        output / "отчет_автотюнинг.csv", index=False, sep=";", decimal=",",
        encoding="utf-8-sig", float_format="%.4f",
    )
    complete = completed_trials(study)
    failed = [trial for trial in study.trials if trial.state == TrialState.FAIL]
    running = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
    lines = [
        "# Отчёт по автоматическому подбору MLP",
        "",
        "## Состояние",
        "",
        "- Успешных trials: **{}**.".format(len(complete)),
        "- Неудачных trials: **{}**.".format(len(failed)),
        "- Незавершённых trials из предыдущих запусков: **{}**.".format(len(running)),
        "- GPU workers: **{}**.".format(", ".join(devices)),
        "- Месяцы подбора: **{}**.".format(
            ", ".join(display_month(fold["test_month"]) for fold in folds)
        ),
        "- Защищённые финальные месяцы: **{}**.".format(
            ", ".join(display_month(value) for value in protected)
        ),
        "",
        "Каждый trial оценивается по среднему совокупному MAPE зачислений и списаний "
        "на нескольких временных folds. Чем меньше objective, тем лучше.",
        "Модель учит поправку в рублях к прогнозу по среднему; логарифмический target "
        "из старой версии больше не используется.",
        "",
        "> Этот результат выбирает настройки модели, но ещё не является финальной "
        "оценкой для бизнеса. Качество выбранных настроек нужно отдельно проверить "
        "на защищённых {} месяцах, которые автотюнинг не видел.".format(len(protected)),
        "",
    ]
    if complete:
        best = study.best_trial
        best_params = dict(best.params)
        best_params["layers"] = list(best.user_attrs.get("layers", []))
        payload = {
            "model": "mlp",
            "objective_version": MONTHLY_OBJECTIVE_VERSION,
            "objective_name": MONTHLY_OBJECTIVE_NAME,
            "objective": "mean aggregate monthly MAPE for credit and debit",
            "best_value": float(best.value),
            "best_value_percent": float(best.value * 100),
            "best_params": best_params,
            "tuning_periods": [str(fold["test_month"]) for fold in folds],
            "protected_holdout_periods": len(protected),
            "trials_total": len(study.trials),
            "trials_complete": len(complete),
            "trials_failed": len(failed),
        }
        (output / "best_params.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines.extend([
            "## Лучший результат",
            "",
            "- Номер варианта: **{}**.".format(best.number),
            "- Средняя целевая ошибка: **{}%**.".format(
                "{:.3f}".format(best.value * 100).replace(".", ",")
            ),
            "- Слои: **{}**.".format(best_params.get("layers")),
            "- Размер пакета: **{}**.".format(best_params.get("batch_size")),
            "- Скорость обучения: **{}**.".format(best_params.get("learning_rate")),
            "- Доля отключения нейронов: **{}**.".format(best_params.get("dropout")),
            "- Функция активации: **{}**.".format(best_params.get("activation")),
            "",
            "Файл `best_params.json` можно передать финальному benchmark через "
            "`--mlp-params`.",
            "",
        ])
    else:
        lines.extend([
            "## Лучший результат",
            "",
            "Пока нет ни одного успешного trial. Смотрите диагностику внутри "
            "`trials/trial_XXXXX/fold_XX_YYYYMM/failure_diagnostics.txt`.",
            "",
        ])
    lines.extend([
        "## Почему отдельный failed trial не останавливает подбор",
        "",
        "CUDA-worker каждого fold является отдельным процессом. Если конкретная "
        "комбинация слоёв или batch приводит к OOM/SIGSEGV, Optuna получает состояние "
        "FAIL и сохраняет диагностику. Две полностью неудачные волны подряд аварийно "
        "останавливают подбор, чтобы не создавать десятки одинаковых FAILED trials.",
    ])
    report = "\n".join(lines) + "\n"
    (output / "отчет_автотюнинг.md").write_text(report, encoding="utf-8")
    (output / "краткий_отчет_автотюнинг.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials должен быть положительным.")
    if args.worker_timeout_minutes < 1:
        raise ValueError("--worker-timeout-minutes должен быть положительным.")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or any(not item.startswith("cuda:") for item in devices):
        raise ValueError("--devices должен содержать CUDA-устройства, например cuda:0,cuda:1.")
    if len(set(devices)) != len(devices):
        raise ValueError("В --devices одно и то же устройство указано несколько раз.")
    preflight_devices(devices)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials_root = output / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    # Autotuning always resumes compatible prepared arrays automatically.
    args.resume = True
    prepared = output / "prepared_numpy"
    manifest = prepare_numpy_data(args, prepared)
    folds, protected = tuning_folds(
        manifest, args.tuning_periods, args.holdout_test_periods, args.min_train_months
    )
    print("\n=== УСТОЙЧИВЫЙ AUTOTUNE MLP ===")
    print("Новых trials: {} | одновременно: {} | устройства: {}".format(
        args.trials, len(devices), ", ".join(devices)
    ))
    print("Tuning folds: {} | holdout не используется: {}".format(
        ", ".join(str(fold["test_month"]) for fold in folds),
        ", ".join(str(value) for value in protected),
    ))
    storage = "sqlite:///{}".format((output / "study.sqlite3").resolve())
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name="cashflow_mlp_stable",
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    study_key = {
        "prepared_key": manifest["prepared_key"],
        "tuning_folds": list(folds),
        "protected_months": list(protected),
        "epochs": args.epochs,
        "mape_zero_floor": args.mape_zero_floor,
        "objective_version": MONTHLY_OBJECTIVE_VERSION,
        "objective_name": MONTHLY_OBJECTIVE_NAME,
    }
    previous_key = study.user_attrs.get("study_key")
    if previous_key is not None and previous_key != study_key:
        raise ValueError(
            "Этот --output-dir уже содержит study с другими данными, периодами или "
            "настройками objective. Укажите новый --output-dir, чтобы не смешивать результаты."
        )
    if previous_key is None:
        study.set_user_attr("study_key", study_key)

    parallel_safe = training_preflight(args, prepared, folds[0], devices, output)
    simultaneous = len(devices) if parallel_safe else 1
    print(
        "Фактический режим Optuna: одновременно {} trial(s); GPU {}.".format(
            simultaneous,
            "работают параллельно" if parallel_safe else "чередуются после проверки SIGSEGV",
        ),
        flush=True,
    )

    remaining = args.trials
    device_cursor = 0
    failed_waves = 0
    while remaining > 0:
        wave_size = min(simultaneous, remaining)
        wave = []
        for index in range(wave_size):
            trial = study.ask()
            params = suggest_params(trial, args)
            device = devices[(device_cursor + index) % len(devices)]
            wave.append((trial, params, device))
        device_cursor = (device_cursor + wave_size) % len(devices)
        wave_successes = 0
        wave_failures: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=wave_size) as executor:
            futures = {
                executor.submit(
                    run_trial, args, prepared, folds, trial.number, params, device, trials_root
                ): trial
                for trial, params, device in wave
            }
            for future in concurrent.futures.as_completed(futures):
                trial = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    study.tell(trial, state=TrialState.FAIL)
                    wave_failures.append("trial {}: {!r}".format(trial.number, error))
                    print("Trial {} аварийно завершился: {!r}; подбор продолжается.".format(
                        trial.number, error
                    ), flush=True)
                    continue
                if result["status"] == "complete":
                    study.tell(trial, float(result["objective"]))
                    wave_successes += 1
                    print("\n>>> Trial {} готов | objective {:.3f}% | {}".format(
                        trial.number, float(result["objective_percent"]), result["device"]
                    ), flush=True)
                else:
                    study.tell(trial, state=TrialState.FAIL)
                    wave_failures.append(
                        "trial {}: {}".format(
                            trial.number, result.get("signal", result.get("reason"))
                        )
                    )
                    print("\n>>> Trial {} FAILED: {} | подбор продолжается".format(
                        trial.number, result.get("signal", result.get("reason"))
                    ), flush=True)
        remaining -= wave_size
        write_reports(study, output, folds, protected, devices)
        print("Состояние study сохранено: успешных {} / всего {}.".format(
            len(completed_trials(study)), len(study.trials)
        ), flush=True)
        if wave_successes == 0:
            failed_waves += 1
        else:
            failed_waves = 0
        if failed_waves >= 2:
            raise RuntimeError(
                "Две последовательные волны autotune полностью упали. Подбор остановлен, "
                "чтобы не создавать десятки одинаковых FAILED trials. Последние причины: {}".format(
                    " | ".join(wave_failures)
                )
            )

    complete = completed_trials(study)
    if complete:
        print("\n=== AUTOTUNE ЗАВЕРШЁН ===")
        print("Лучший objective: {:.3f}% | trial {}".format(
            study.best_value * 100, study.best_trial.number
        ))
        print("Параметры: {}".format((output / "best_params.json").resolve()))
    else:
        print("\nAutotune завершил запуск без успешных trials. Откройте краткий отчёт и diagnostics.")
    print("Отчёт: {}".format((output / "отчет_автотюнинг.md").resolve()))


if __name__ == "__main__":
    main()
