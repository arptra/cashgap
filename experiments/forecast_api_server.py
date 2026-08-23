#!/usr/bin/env python3
"""Простой FastAPI-сервер месячных прогнозов по ИНН и периоду."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from typing import Dict, Optional


def _check_dependencies() -> None:
    required = {
        "fastapi": "fastapi==0.103.2" if sys.version_info[:2] == (3, 8) else "fastapi",
        "uvicorn": "uvicorn==0.23.2" if sys.version_info[:2] == (3, 8) else "uvicorn",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        executable = shlex.quote(sys.executable)
        print("ОШИБКА: не установлены библиотеки API: {}".format(", ".join(missing)))
        print("Установка: {} -m pip install {}".format(executable, " ".join(missing)))
        print("Jupyter: %pip install {}".format(" ".join(missing)))
        print("После установки перезапустите kernel Jupyter.")
        raise SystemExit(2)
    print("Проверка зависимостей: библиотеки API установлены.")


if __name__ == "__main__":
    _check_dependencies()

import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field


def normalize_period(value: str) -> str:
    text = str(value).strip()
    if len(text) == 6 and text.isdigit():
        text = "{}-{}".format(text[:4], text[4:])
    try:
        period = pd.Period(text, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError("Период должен иметь формат YYYY-MM, например 2025-06.") from error
    return str(period)


class ForecastRequest(BaseModel):
    inn: str = Field(..., description="ИНН компании")
    period: str = Field(..., description="Месяц прогноза в формате YYYY-MM")


class ForecastStore:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        metadata_path = model_dir / "model_metadata.json"
        forecasts_path = model_dir / "forecasts_api.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError("Не найден файл метаданных: {}".format(metadata_path))
        if not forecasts_path.exists():
            raise FileNotFoundError("Не найдена таблица прогнозов: {}".format(forecasts_path))
        self.metadata: Dict[str, object] = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(forecasts_path)
        required = {
            "inn", "period", "model", "forecast_step", "forecast_type",
            "predicted_inflow", "predicted_outflow", "predicted_net_flow",
            "negative_net_flow",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("В forecasts_api.parquet отсутствуют поля: {}".format(missing))
        frame["inn"] = frame["inn"].astype(str).str.strip()
        frame["period"] = frame["period"].astype(str).map(normalize_period)
        if frame.duplicated(["inn", "period"]).any():
            raise ValueError("Таблица API содержит повторяющиеся пары ИНН/период.")
        self.periods = sorted(frame["period"].unique().tolist())
        self.inn_count = int(frame["inn"].nunique())
        self.frame = frame.set_index(["inn", "period"]).sort_index()

    def forecast(self, inn: str, period: str) -> Dict[str, object]:
        normalized_inn = str(inn).strip()
        normalized_period = normalize_period(period)
        try:
            row = self.frame.loc[(normalized_inn, normalized_period)]
        except KeyError as error:
            inn_exists = normalized_inn in self.frame.index.get_level_values("inn")
            if not inn_exists:
                message = "ИНН {} отсутствует в пакете модели.".format(normalized_inn)
            else:
                message = (
                    "Для ИНН {} нет периода {}. Доступный диапазон: {} — {}."
                    .format(normalized_inn, normalized_period, self.periods[0], self.periods[-1])
                )
            raise LookupError(message) from error
        return {
            "инн": normalized_inn,
            "период": normalized_period,
            "код_модели": str(row["model"]),
            "название_модели": str(self.metadata.get("model_name_ru", row["model"])),
            "шаг_прогноза_месяцев": int(row["forecast_step"]),
            "тип_прогноза": "прямой" if row["forecast_type"] == "direct" else "рекурсивный",
            "прогноз_зачислений": round(float(row["predicted_inflow"]), 2),
            "прогноз_списаний": round(float(row["predicted_outflow"]), 2),
            "прогноз_чистого_потока": round(float(row["predicted_net_flow"]), 2),
            "отрицательный_чистый_поток": bool(row["negative_net_flow"]),
            "предупреждение": (
                "Отрицательный чистый поток не равен кассовому разрыву: "
                "для него нужны остатки на счетах."
            ),
        }


def create_app(store: ForecastStore, api_key: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="API прогноза денежных потоков",
        description=(
            "Возвращает месячный прогноз зачислений, списаний и чистого потока "
            "по ИНН и периоду."
        ),
        version="1.0.0",
    )

    def authorize(value: Optional[str]) -> None:
        if api_key and value != api_key:
            raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key.")

    @app.get("/health", summary="Проверить сервер")
    def health(x_api_key: Optional[str] = Header(default=None)) -> Dict[str, object]:
        authorize(x_api_key)
        return {
            "статус": "работает",
            "модель": store.metadata.get("model_name_ru", store.metadata.get("model_id")),
            "количество_инн": store.inn_count,
            "период_с": store.periods[0],
            "период_по": store.periods[-1],
        }

    @app.get("/forecast", summary="Получить прогноз")
    def forecast_get(
        inn: str = Query(..., description="ИНН компании"),
        period: str = Query(..., description="Период YYYY-MM"),
        x_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, object]:
        authorize(x_api_key)
        try:
            return store.forecast(inn, period)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/forecast", summary="Получить прогноз JSON-запросом")
    def forecast_post(
        request: ForecastRequest,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, object]:
        authorize(x_api_key)
        try:
            return store.forecast(request.inn, request.period)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Каталог saved_model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key", default=None,
        help="Если задан, клиенты должны передавать заголовок X-API-Key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ForecastStore(Path(args.model_dir))
    app = create_app(store, args.api_key)
    print("\n=== API ПРОГНОЗА ДЕНЕЖНЫХ ПОТОКОВ ===")
    print("Модель: {}".format(store.metadata.get("model_name_ru", store.metadata.get("model_id"))))
    print("ИНН: {:,} | периоды: {} — {}".format(
        store.inn_count, store.periods[0], store.periods[-1]
    ))
    print("Swagger: http://{}:{}/docs".format(args.host, args.port))
    if args.host == "0.0.0.0" and not args.api_key:
        print("ВНИМАНИЕ: сервер доступен по сети без API-ключа.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
