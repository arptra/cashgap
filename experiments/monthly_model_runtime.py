#!/usr/bin/env python3
"""Standalone runtime for a monthly PyTorch cash-flow checkpoint.

The runtime accepts the already engineered raw feature matrix in the exact
column order stored inside ``model.pt`` and returns two ruble amounts per row:
``predicted_inflow`` and ``predicted_outflow``. It intentionally has no pandas,
PyArrow, FastAPI or project-module dependency, so it can be handed to another
team together with the checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _check_dependencies() -> None:
    required = {
        "numpy": "numpy==1.24.4" if sys.version_info[:2] == (3, 8) else "numpy",
        "torch": "torch==2.3.1" if sys.version_info[:2] == (3, 8) else "torch",
    }
    missing = [
        package for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: runtime модели не хватает: {}".format(", ".join(missing)))
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
        raise SystemExit(2)


if __name__ == "__main__":
    _check_dependencies()

import numpy as np
import torch
from torch import nn


RUNTIME_FORMAT_VERSION = 1
SUPPORTED_CHECKPOINT_FORMAT = 2
SUPPORTED_OBJECTIVE_VERSION = 2
OUTPUT_NAMES: Tuple[str, str] = ("predicted_inflow", "predicted_outflow")
BASELINE_FEATURES: Tuple[str, str] = (
    "target_inflow_mean_3",
    "target_outflow_mean_3",
)
SOURCE_NAMES: Tuple[str, str, str, str, str, str] = (
    "target_inflow",
    "target_outflow",
    "target_net_flow",
    "active_inflow_days",
    "active_outflow_days",
    "negative_days",
)


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[str(name).lower()]()
    except KeyError as error:
        raise ValueError("Неизвестная функция активации {!r}.".format(name)) from error


def _load_checkpoint(path: Path) -> Dict[str, object]:
    # weights_only prevents execution of arbitrary pickle objects. The exported
    # checkpoint contains only tensors and primitive Python containers.
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older Torch builds.
        value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("model.pt должен содержать словарь checkpoint.")
    return value


def _require_sequence(checkpoint: Mapping[str, object], key: str) -> Sequence[object]:
    value = checkpoint.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("В checkpoint отсутствует непустое поле {!r}.".format(key))
    return value


class MonthlyCashflowRuntime:
    """Load one checkpoint once and execute efficient batch inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        cpu_threads: Optional[int] = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("Checkpoint не найден: {}".format(path))
        checkpoint = _load_checkpoint(path)
        if int(checkpoint.get("format_version", -1)) != SUPPORTED_CHECKPOINT_FORMAT:
            raise ValueError(
                "Неподдерживаемый format_version={!r}; ожидается {}.".format(
                    checkpoint.get("format_version"), SUPPORTED_CHECKPOINT_FORMAT
                )
            )
        if int(checkpoint.get("objective_version", -1)) != SUPPORTED_OBJECTIVE_VERSION:
            raise ValueError(
                "Checkpoint использует старую objective_version={!r}; ожидается {}.".format(
                    checkpoint.get("objective_version"), SUPPORTED_OBJECTIVE_VERSION
                )
            )
        if cpu_threads is not None:
            if int(cpu_threads) < 1:
                raise ValueError("cpu_threads должен быть положительным.")
            torch.set_num_threads(int(cpu_threads))

        self.model_path = path
        self.model_id = str(checkpoint.get("model_id", "unknown"))
        self.objective_name = str(checkpoint.get("objective_name", "unknown"))
        self.feature_names = tuple(str(value) for value in _require_sequence(checkpoint, "features"))
        self.layers = tuple(int(value) for value in _require_sequence(checkpoint, "layers"))
        self.activation = str(checkpoint.get("activation", "relu"))
        self.dropout = float(checkpoint.get("dropout", 0.0))
        missing_baseline = [name for name in BASELINE_FEATURES if name not in self.feature_names]
        if missing_baseline:
            raise ValueError("Checkpoint не содержит baseline-признаки: {}.".format(missing_baseline))
        self.baseline_indices = tuple(self.feature_names.index(name) for name in BASELINE_FEATURES)

        width = len(self.feature_names)
        modules: List[nn.Module] = []
        for hidden in self.layers:
            modules.extend([
                nn.Linear(width, hidden),
                _activation(self.activation),
                nn.Dropout(self.dropout),
            ])
            width = hidden
        modules.append(nn.Linear(width, 2))
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Запрошена CUDA, но torch.cuda.is_available() == False.")
            index = 0 if self.device.index is None else int(self.device.index)
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    "Запрошена {}, но PyTorch видит GPU: {}.".format(
                        self.device, torch.cuda.device_count()
                    )
                )
        self.model = nn.Sequential(*modules)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("В checkpoint отсутствует state_dict.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        feature_count = len(self.feature_names)
        self.feature_mean = self._vector(checkpoint, "feature_mean", feature_count, np.float64)
        self.feature_scale = self._vector(checkpoint, "feature_scale", feature_count, np.float64)
        self.active_features = self._vector(
            checkpoint, "active_features", feature_count, np.bool_
        ).astype(bool, copy=False)
        self.residual_scale = self._vector(checkpoint, "residual_scale", 2, np.float64)
        if (self.feature_scale <= 0).any() or (self.residual_scale <= 0).any():
            raise ValueError("Checkpoint содержит неположительный масштаб.")

    @staticmethod
    def _vector(
        checkpoint: Mapping[str, object], key: str, length: int, dtype: object,
    ) -> np.ndarray:
        value = checkpoint.get(key)
        array = np.asarray(value, dtype=dtype)
        if array.shape != (length,):
            raise ValueError(
                "Поле {} имеет форму {}, ожидается ({},).".format(key, array.shape, length)
            )
        if array.dtype != np.bool_ and not np.isfinite(array).all():
            raise ValueError("Поле {} содержит NaN или infinity.".format(key))
        return array

    @property
    def input_width(self) -> int:
        return len(self.feature_names)

    @property
    def output_names(self) -> Tuple[str, str]:
        return OUTPUT_NAMES

    def describe(self) -> Dict[str, object]:
        return {
            "runtime_format_version": RUNTIME_FORMAT_VERSION,
            "model_id": self.model_id,
            "objective_name": self.objective_name,
            "device": str(self.device),
            "layers": list(self.layers),
            "input_width": self.input_width,
            "input_features": list(self.feature_names),
            "outputs": list(self.output_names),
        }

    def predict_matrix(self, raw_features: np.ndarray, batch_size: int = 65_536) -> np.ndarray:
        """Predict from an ``(N, input_width)`` raw feature matrix.

        Columns must follow ``feature_names`` exactly. The method performs the
        saved normalization and converts residual network outputs back to rubles.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size должен быть положительным.")
        values = np.asarray(raw_features)
        if values.ndim != 2 or values.shape[1] != self.input_width:
            raise ValueError(
                "Ожидается матрица (N, {}), получено {}.".format(
                    self.input_width, values.shape
                )
            )
        if len(values) == 0:
            return np.empty((0, 2), dtype=np.float64)

        parts: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), int(batch_size)):
                raw = np.asarray(values[start:start + int(batch_size)], dtype=np.float64)
                if not np.isfinite(raw).all():
                    raise ValueError("В признаках есть NaN или infinity.")
                baseline = np.maximum(raw[:, self.baseline_indices], 0.0)
                normalized = (raw - self.feature_mean) / self.feature_scale
                normalized[:, ~self.active_features] = 0.0
                normalized = np.clip(normalized, -10.0, 10.0).astype(np.float32)
                tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(self.device)
                correction = self.model(tensor).float().cpu().numpy().astype(np.float64)
                prediction = np.maximum(
                    baseline + correction * self.residual_scale.reshape(1, 2), 0.0
                )
                parts.append(prediction)
        return np.concatenate(parts, axis=0)

    def predict_one(self, raw_features: Mapping[str, float]) -> Dict[str, float]:
        missing = [name for name in self.feature_names if name not in raw_features]
        if missing:
            raise ValueError(
                "Не хватает {} признаков; первые отсутствующие: {}.".format(
                    len(missing), missing[:10]
                )
            )
        row = np.asarray(
            [[float(raw_features[name]) for name in self.feature_names]], dtype=np.float64
        )
        prediction = self.predict_matrix(row, batch_size=1)[0]
        return {
            OUTPUT_NAMES[0]: float(prediction[0]),
            OUTPUT_NAMES[1]: float(prediction[1]),
        }

    @staticmethod
    def _nan_mean(values: np.ndarray) -> np.ndarray:
        count = np.sum(~np.isnan(values), axis=1)
        total = np.nansum(values, axis=1)
        return np.divide(
            total, count, out=np.full(len(values), np.nan), where=count > 0
        )

    @classmethod
    def _nan_std(cls, values: np.ndarray) -> np.ndarray:
        count = np.sum(~np.isnan(values), axis=1)
        mean = cls._nan_mean(values)
        centered = np.where(np.isnan(values), 0.0, values - mean[:, None])
        variance = np.divide(
            np.sum(centered ** 2, axis=1),
            count - 1,
            out=np.full(len(values), np.nan),
            where=count > 1,
        )
        return np.sqrt(variance)

    def build_features_from_history(
        self,
        monthly_history: np.ndarray,
        history_months: Sequence[int],
        forecast_month: Sequence[int],
    ) -> np.ndarray:
        """Build the model matrix from the last 12 calendar months.

        ``monthly_history`` has shape ``(N, 12, 6)`` and is ordered from the
        oldest to the newest month. The six columns follow ``SOURCE_NAMES``.
        Calendar months without operations after a company first appears are
        zeros; unavailable months before its first appearance are NaN.
        ``history_months`` is the total number of observed calendar months before
        the forecast, and ``forecast_month`` contains month numbers 1..12.
        """
        history = np.asarray(monthly_history, dtype=np.float64)
        if history.ndim != 3 or history.shape[1:] != (12, len(SOURCE_NAMES)):
            raise ValueError(
                "Ожидается history формы (N, 12, {}), получено {}.".format(
                    len(SOURCE_NAMES), history.shape
                )
            )
        counts = np.asarray(history_months, dtype=np.float64)
        months = np.asarray(forecast_month, dtype=np.int64)
        if counts.shape != (len(history),) or months.shape != (len(history),):
            raise ValueError("history_months и forecast_month должны иметь форму (N,).")
        if (
            not np.isfinite(counts).all()
            or (counts < 1).any()
            or not np.allclose(counts, np.rint(counts))
        ):
            raise ValueError("history_months должен содержать положительные целые числа.")
        if ((months < 1) | (months > 12)).any():
            raise ValueError("forecast_month должен содержать номера месяцев 1..12.")
        if np.isinf(history).any():
            raise ValueError("История содержит infinity.")
        nonnegative_columns = history[:, :, [0, 1, 3, 4, 5]]
        if np.nanmin(nonnegative_columns, initial=0.0) < 0:
            raise ValueError("Суммы и счётчики активных дней не могут быть отрицательными.")
        if np.nanmax(history[:, :, [3, 4, 5]], initial=0.0) > 31:
            raise ValueError("Счётчик активных дней не может превышать 31.")
        present = ~np.isnan(history[:, :, 0]) & ~np.isnan(history[:, :, 1])
        inconsistent = present & ~np.isclose(
            history[:, :, 2], history[:, :, 0] - history[:, :, 1],
            rtol=1e-6, atol=0.01, equal_nan=True,
        )
        if inconsistent.any():
            raise ValueError(
                "target_net_flow в истории не равен target_inflow - target_outflow."
            )

        values: Dict[str, np.ndarray] = {}
        for source_index, source in enumerate(SOURCE_NAMES):
            source_history = history[:, :, source_index]
            for lag in (1, 2, 3, 6, 12):
                values["{}_lag_{}".format(source, lag)] = source_history[:, -lag]
            for window in (3, 6, 12):
                window_values = source_history[:, -window:]
                values["{}_mean_{}".format(source, window)] = self._nan_mean(window_values)
                values["{}_std_{}".format(source, window)] = self._nan_std(window_values)
        values["inflow_change_1"] = (
            values["target_inflow_lag_1"] - values["target_inflow_lag_2"]
        )
        values["outflow_change_1"] = (
            values["target_outflow_lag_1"] - values["target_outflow_lag_2"]
        )
        values["inflow_ratio_to_mean_6"] = values["target_inflow_lag_1"] / np.maximum(
            values["target_inflow_mean_6"], 1.0
        )
        values["outflow_ratio_to_mean_6"] = values["target_outflow_lag_1"] / np.maximum(
            values["target_outflow_mean_6"], 1.0
        )
        values["month_sin"] = np.sin(2 * math.pi * months / 12.0)
        values["month_cos"] = np.cos(2 * math.pi * months / 12.0)
        values["history_months"] = counts
        missing = [name for name in self.feature_names if name not in values]
        if missing:
            raise ValueError("Runtime не умеет построить признаки: {}.".format(missing))
        matrix = np.column_stack([values[name] for name in self.feature_names])
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def predict_history(
        self,
        monthly_history: np.ndarray,
        history_months: Sequence[int],
        forecast_month: Sequence[int],
        batch_size: int = 65_536,
    ) -> np.ndarray:
        features = self.build_features_from_history(
            monthly_history, history_months, forecast_month
        )
        return self.predict_matrix(features, batch_size=batch_size)

    def run_self_test(self, path: str) -> Dict[str, object]:
        test_path = Path(path)
        with np.load(str(test_path), allow_pickle=False) as payload:
            raw_features = payload["raw_features"]
            expected = payload["expected_predictions"]
            files = set(payload.files)
            history = payload["monthly_history"] if "monthly_history" in files else None
            history_months = payload["history_months"] if "history_months" in files else None
            forecast_month = payload["forecast_month"] if "forecast_month" in files else None
            expected_history = (
                payload["expected_history_predictions"]
                if "expected_history_predictions" in files else None
            )
        actual = self.predict_matrix(raw_features)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.25)
        differences = [float(np.max(np.abs(actual - expected)))]
        if history is not None:
            actual_history = self.predict_history(
                history, history_months, forecast_month
            )
            np.testing.assert_allclose(
                actual_history, expected_history, rtol=2e-5, atol=0.25
            )
            differences.append(float(np.max(np.abs(actual_history - expected_history))))
        return {
            "status": "OK",
            "rows": int(len(actual)),
            "history_builder_checked": history is not None,
            "max_absolute_difference": max(differences),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Путь к model.pt")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0 и т. п.")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--self-test", default=None, help="Путь к runtime_self_test.npz")
    parser.add_argument("--input-npy", default=None, help="Матрица сырых признаков (N, F)")
    parser.add_argument("--output-npy", default=None, help="Куда записать прогноз (N, 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = MonthlyCashflowRuntime(args.model, args.device, args.cpu_threads)
    description = runtime.describe()
    print(
        "Модель: {} | устройство: {} | входов: {} | слои: {}".format(
            description["model_id"], description["device"],
            description["input_width"], description["layers"],
        )
    )
    print("Выходы: {}".format(", ".join(description["outputs"])))
    if args.describe:
        print("Порядок входных признаков:")
        for index, name in enumerate(runtime.feature_names):
            print("{:3d}: {}".format(index, name))
    if args.self_test:
        print("Self-test: {}".format(runtime.run_self_test(args.self_test)))
    if args.input_npy:
        if not args.output_npy:
            raise ValueError("Вместе с --input-npy обязательно укажите --output-npy.")
        values = np.load(args.input_npy, allow_pickle=False)
        prediction = runtime.predict_matrix(values, args.batch_size)
        np.save(args.output_npy, prediction, allow_pickle=False)
        print("Прогнозы: {} строк -> {}".format(len(prediction), Path(args.output_npy).resolve()))
    elif not args.describe and not args.self_test:
        raise ValueError("Укажите --describe, --self-test или --input-npy/--output-npy.")


if __name__ == "__main__":
    main()
