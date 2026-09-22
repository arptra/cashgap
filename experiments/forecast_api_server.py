#!/usr/bin/env python3
"""Простой FastAPI-сервер месячных прогнозов по ИНН и периоду."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import importlib.util
import json
import math
import secrets
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
from fastapi.responses import HTMLResponse
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
        if frame.empty:
            raise ValueError("Таблица прогнозов пуста: показывать нечего.")
        for column in ("predicted_inflow", "predicted_outflow", "predicted_net_flow"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not frame[column].map(math.isfinite).all():
                raise ValueError("В {} есть NaN или бесконечные суммы.".format(column))
        if (frame[["predicted_inflow", "predicted_outflow"]] < 0).any().any():
            raise ValueError("Суммы зачислений и списаний не должны быть отрицательными.")
        calculated_net = frame["predicted_inflow"] - frame["predicted_outflow"]
        tolerance = 0.05 + 1e-7 * (frame["predicted_inflow"] + frame["predicted_outflow"])
        if ((calculated_net - frame["predicted_net_flow"]).abs() > tolerance).any():
            raise ValueError("Чистый поток в файле не совпадает с зачислениями минус списания.")
        if frame.duplicated(["inn", "period"]).any():
            raise ValueError("Таблица API содержит повторяющиеся пары ИНН/период.")
        self.periods = sorted(frame["period"].unique().tolist())
        self.inn_count = int(frame["inn"].nunique())
        self.inns = sorted(frame["inn"].unique().tolist())
        self.frame = frame.set_index(["inn", "period"]).sort_index()

    def client_search(self, query: str, limit: int = 15):
        query = query.strip()
        start = bisect_left(self.inns, query)
        result = []
        for inn in self.inns[start:start + limit]:
            if not inn.startswith(query):
                break
            result.append(inn)
        return result

    def client_timeline(self, inn: str):
        inn = inn.strip()
        try:
            rows = self.frame.xs(inn, level="inn")
        except KeyError:
            raise LookupError("ИНН {} отсутствует в таблице прогнозов.".format(inn))
        # Keep browser payloads bounded even for atypically large exports.
        periods = rows.index.tolist()
        return {
            "inn": inn,
            "total_periods": len(periods),
            "truncated": len(periods) > 120,
            "forecasts": [self.forecast(inn, period) for period in periods[:120]],
        }

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

    @app.middleware("http")
    async def private_responses(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def authorize(value: Optional[str]) -> None:
        if api_key and value != api_key:
            raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key.")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def business_ui():
        ui_path = Path(__file__).with_name("forecast_ui.html")
        if not ui_path.exists():
            raise HTTPException(status_code=503, detail="Скопируйте forecast_ui.html рядом с forecast_api_server.py.")
        return HTMLResponse(ui_path.read_text(encoding="utf-8"), headers={
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        })

    @app.get("/ui/meta", summary="Описание набора прогнозов для бизнес-экрана")
    def ui_meta(x_api_key: Optional[str] = Header(default=None)):
        authorize(x_api_key)
        return {
            "model_name": store.metadata.get("model_name_ru", store.metadata.get("model_id", "Не указана")),
            "last_complete_month": store.metadata.get("last_complete_month"),
            "inn_count": store.inn_count,
            "first_period": store.periods[0],
            "last_period": store.periods[-1],
            "example_inn": store.inns[0],
            "source": "forecasts_api.parquet",
            "mode": "saved_monthly_forecasts",
        }

    @app.get("/ui/clients", summary="Поиск ИНН по началу номера")
    def ui_clients(q: str = Query(default="", max_length=64),
                   x_api_key: Optional[str] = Header(default=None)):
        authorize(x_api_key)
        return {"inns": store.client_search(q)}

    @app.get("/ui/example", summary="Случайный ИНН из имеющихся прогнозов")
    def ui_example(exclude: str = Query(default="", max_length=64),
                   x_api_key: Optional[str] = Header(default=None)):
        authorize(x_api_key)
        position = bisect_left(store.inns, exclude)
        found = position < len(store.inns) and store.inns[position] == exclude
        if found and len(store.inns) > 1:
            choice = secrets.randbelow(len(store.inns) - 1)
            choice += choice >= position
        else:
            choice = secrets.randbelow(len(store.inns))
        return {"inn": store.inns[choice]}

    @app.get("/ui/timeline", summary="Прогнозы одного ИНН по доступным месяцам")
    def ui_timeline(inn: str = Query(..., min_length=1, max_length=64),
                    x_api_key: Optional[str] = Header(default=None)):
        authorize(x_api_key)
        try:
            return store.client_timeline(inn)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

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
    print("Бизнес-интерфейс: http://{}:{}/".format(args.host, args.port))
    if args.host == "0.0.0.0" and not args.api_key:
        print("ВНИМАНИЕ: сервер доступен по сети без API-ключа.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
