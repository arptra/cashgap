#!/usr/bin/env python3
"""Pure NumPy/PyTorch worker for one monthly fold.

The worker intentionally does not import pandas, PyArrow, sklearn or scipy.
This keeps CUDA in a small native runtime and avoids cross-library teardown
segfaults. Successful workers terminate with os._exit(0) after all artifacts
are flushed, bypassing unsafe CUDA/native destructors at interpreter shutdown.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: GPU-worker не нашёл: {}".format(", ".join(missing)), flush=True)
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--validation-month", type=int, required=True)
    parser.add_argument("--test-month", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--activation", choices=("relu", "gelu", "silu"), default="relu")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--mape-zero-floor", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> Tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--layers должен содержать размеры через запятую.") from error
    if not layers or any(width < 1 for width in layers):
        raise ValueError("--layers должен содержать положительные размеры.")
    return layers


def activation_class(name: str):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]


def build_model(input_width: int, layers: Sequence[int], activation: str, dropout: float) -> nn.Module:
    modules: List[nn.Module] = []
    width = input_width
    kind = activation_class(activation)
    for hidden in layers:
        modules.extend([nn.Linear(width, int(hidden)), kind(), nn.Dropout(dropout)])
        width = int(hidden)
    modules.append(nn.Linear(width, 2))
    return nn.Sequential(*modules)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def metrics(actual: np.ndarray, predicted: np.ndarray, zero_floor: float) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for index, flow in enumerate(("inflow_credit", "outflow_debit")):
        truth = actual[:, index].astype(np.float64)
        forecast = predicted[:, index].astype(np.float64)
        nonzero = np.abs(truth) > zero_floor
        total_truth = float(truth.sum())
        total_forecast = float(forecast.sum())
        aggregate_mape = abs(total_forecast - total_truth) / max(abs(total_truth), zero_floor)
        wape = np.abs(forecast - truth).sum() / max(np.abs(truth).sum(), zero_floor)
        company_mape = (
            float(np.mean(np.abs(forecast[nonzero] - truth[nonzero]) / np.abs(truth[nonzero])))
            if nonzero.any() else float("nan")
        )
        rows.append({
            "flow": flow,
            "aggregate_mape": float(aggregate_mape),
            "aggregate_mape_percent": float(aggregate_mape * 100),
            "company_mape_nonzero": company_mape,
            "company_mape_nonzero_percent": float(company_mape * 100),
            "wape": float(wape),
            "wape_percent": float(wape * 100),
            "mae": float(np.mean(np.abs(forecast - truth))),
            "bias_percent": float(
                (total_forecast - total_truth) / max(abs(total_truth), zero_floor) * 100
            ),
            "actual_total": total_truth,
            "predicted_total": total_forecast,
            "nonzero_companies": int(nonzero.sum()),
            "companies": int(len(truth)),
        })
    return rows


def gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return "VRAM allocated {:.2f} | reserved {:.2f} | peak {:.2f} GiB".format(
        allocated, reserved, peak
    )


def live_gpu_status(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "nvidia-smi не найден"
    index = 0 if device.index is None else int(device.index)
    command = [
        executable,
        "--id={}".format(index),
        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "nvidia-smi: {}".format(error)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    layers = parse_layers(args.layers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    success_marker = output / "SUCCESS"
    if success_marker.exists():
        success_marker.unlink()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    device_index = 0 if device.index is None else int(device.index)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() вернул False.")
        if device_index >= torch.cuda.device_count():
            raise ValueError("Запрошена {}, PyTorch видит GPU: {}.".format(device, torch.cuda.device_count()))
        torch.cuda.set_device(device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    properties = torch.cuda.get_device_properties(device_index) if device.type == "cuda" else None
    print("\n=== ЧИСТЫЙ GPU-WORKER | ПЕРИОД {} ===".format(args.fold), flush=True)
    print("PID: {} | Python: {} | ОС: {}".format(
        os.getpid(), sys.version.split()[0], platform.platform()
    ), flush=True)
    print("PyTorch: {} | CUDA wheel: {} | cuDNN: {}".format(
        torch.__version__, torch.version.cuda, torch.backends.cudnn.version()
    ), flush=True)
    if properties is not None:
        print("Устройство: {} | {} | {:.1f} GiB".format(
            device, properties.name, properties.total_memory / (1024 ** 3)
        ), flush=True)
    else:
        print("Устройство: CPU (только локальная проверка worker).", flush=True)
    print("Native env | OMP={} | MKL={} | OpenBLAS={} | allocator={}".format(
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    ), flush=True)

    prepared = Path(args.prepared_dir)
    X_all = np.load(str(prepared / "features.npy"), mmap_mode="r", allow_pickle=False)
    y_all = np.load(str(prepared / "targets.npy"), mmap_mode="r", allow_pickle=False)
    months = np.load(str(prepared / "months.npy"), mmap_mode="r", allow_pickle=False)
    inns = np.load(str(prepared / "inns.npy"), mmap_mode="r", allow_pickle=False)
    train_indices = np.flatnonzero(months < args.validation_month)
    valid_indices = np.flatnonzero(months == args.validation_month)
    test_indices = np.flatnonzero(months == args.test_month)
    if min(len(train_indices), len(valid_indices), len(test_indices)) == 0:
        raise ValueError("Пустой train/validation/test: {}/{}/{}.".format(
            len(train_indices), len(valid_indices), len(test_indices)
        ))
    print("Строки | обучение {:,} | валидация {:,} | тест {:,} | признаков {}".format(
        len(train_indices), len(valid_indices), len(test_indices), X_all.shape[1]
    ), flush=True)

    X_train_raw = np.asarray(X_all[train_indices], dtype=np.float32)
    X_valid_raw = np.asarray(X_all[valid_indices], dtype=np.float32)
    X_test_raw = np.asarray(X_all[test_indices], dtype=np.float32)
    actual_test = np.asarray(y_all[test_indices], dtype=np.float64)
    feature_mean = X_train_raw.mean(axis=0, dtype=np.float64)
    feature_scale = X_train_raw.std(axis=0, dtype=np.float64)
    feature_scale = np.maximum(feature_scale, 1e-6)
    X_train_np = ((X_train_raw - feature_mean) / feature_scale).astype(np.float32)
    X_valid_np = ((X_valid_raw - feature_mean) / feature_scale).astype(np.float32)
    X_test_np = ((X_test_raw - feature_mean) / feature_scale).astype(np.float32)
    y_train_log = np.log1p(np.asarray(y_all[train_indices], dtype=np.float32))
    y_valid_log = np.log1p(np.asarray(y_all[valid_indices], dtype=np.float32))
    target_mean = y_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(
        y_train_log.std(axis=0, dtype=np.float64).astype(np.float32), 1e-3
    )
    y_train_np = ((y_train_log - target_mean) / target_scale).astype(np.float32)
    y_valid_np = ((y_valid_log - target_mean) / target_scale).astype(np.float32)
    if not all(np.isfinite(item).all() for item in (
        X_train_np, X_valid_np, X_test_np, y_train_np, y_valid_np
    )):
        raise ValueError("После масштабирования обнаружены NaN или infinity.")

    x_train = torch.as_tensor(X_train_np, device=device)
    x_valid = torch.as_tensor(X_valid_np, device=device)
    x_test = torch.as_tensor(X_test_np, device=device)
    y_train = torch.as_tensor(y_train_np, device=device)
    y_valid = torch.as_tensor(y_valid_np, device=device)
    model = build_model(X_all.shape[1], layers, args.activation, args.dropout).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Модель {} | слои {} | параметров {:,} | batch {:,} | AMP {}".format(
        args.model, list(layers), parameters, args.batch_size, args.amp
    ), flush=True)
    print("После загрузки: {}".format(gpu_memory(device)), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    remaining = args.patience
    batch_size = min(args.batch_size, len(x_train))
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + epoch)
        order = torch.randperm(len(x_train), generator=generator, device=device)
        for start in range(0, len(x_train), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_function(model(x_train[indices]), y_train[indices])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            validation_loss = float(loss_function(model(x_valid), y_valid))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            remaining = args.patience
        else:
            remaining -= 1
        if epoch == 1 or epoch % 5 == 0 or remaining == 0:
            elapsed = time.perf_counter() - started
            print("Эпоха {:3d} | validation {:.6f} | {:.1f} сек | {}".format(
                epoch, validation_loss, elapsed, gpu_memory(device)
            ), flush=True)
            print("  GPU live | utilization, memory, power, temperature: {}".format(
                live_gpu_status(device)
            ), flush=True)
        if remaining == 0:
            print("Ранняя остановка: эпоха {}, лучшая {}.".format(epoch, best_epoch), flush=True)
            break
    if best_state is None:
        raise RuntimeError("Не сохранено ни одного состояния MLP.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        prediction_scaled = model(x_test).float().cpu().numpy()
    prediction_log = prediction_scaled * target_scale + target_mean
    prediction = np.maximum(np.expm1(prediction_log), 0.0).astype(np.float64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metric_rows = metrics(actual_test, prediction, args.mape_zero_floor)
    for row in metric_rows:
        row.update({
            "fold": args.fold,
            "test_month": str(args.test_month),
            "model": args.model,
            "training_seconds": round(training_seconds, 3),
        })

    atomic_npz(
        output / "predictions.npz",
        inn=np.asarray(inns[test_indices]),
        actual_inflow=actual_test[:, 0],
        predicted_inflow=prediction[:, 0],
        actual_outflow=actual_test[:, 1],
        predicted_outflow=prediction[:, 1],
    )
    atomic_json(output / "metrics.json", metric_rows)
    atomic_json(output / "window.json", {
        "fold": args.fold,
        "validation_month": args.validation_month,
        "test_month": args.test_month,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(valid_indices)),
        "test_rows": int(len(test_indices)),
        "device": args.device,
        "best_epoch": best_epoch,
        "training_seconds": round(training_seconds, 3),
    })
    success_marker.write_text("OK\n", encoding="utf-8")
    print("SUCCESS | период {} | {:.1f} сек | {}".format(
        args.fold, training_seconds, gpu_memory(device)
    ), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # CUDA/native destructors have caused exit -11 on the target EL8 host.
    # Artifacts are already durable, so skip interpreter-level native cleanup.
    os._exit(0)


if __name__ == "__main__":
    main()
