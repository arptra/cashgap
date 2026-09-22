"""Adapt saved forecasts to the supplied TypeScript dashboard (no model training)."""
from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional


def kopecks(amount: float) -> int:
    return int((Decimal(str(amount)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def dashboard_payload(store, daily_enabled: bool, inn: Optional[str] = None,
                      period: Optional[str] = None, start_date: Optional[str] = None,
                      opening_balance: Optional[float] = None):
    """Only liquidityAnalysis is model-backed; other original cards are explicit demos."""
    inn = inn.strip() if inn else store.inns[0]
    timeline = store.client_timeline(inn)
    periods = [row["период"] for row in timeline["forecasts"]]
    selected = store.forecast(inn, period or periods[0])
    period = selected["период"]
    if opening_balance is not None and (not math.isfinite(opening_balance) or abs(opening_balance) > 1e12):
        raise ValueError("Начальный остаток должен быть конечной суммой не больше 1 трлн ₽ по модулю.")
    balance_cents = None if opening_balance is None else kopecks(opening_balance)
    dates = store.daily_dates(inn, period)["dates"] if daily_enabled else []
    if start_date and not daily_enabled:
        raise ValueError("Для распределения по дням запустите сервер с --daily-allocation.")
    if daily_enabled:
        if not dates:
            raise ValueError("Для выбранного месяца нет полного окна на 14 дней.")
        start_date = start_date or dates[0]
        if start_date[:7] != period:
            raise ValueError("Начальная дата должна относиться к выбранному месяцу.")
        allocation = store.daily_allocation(inn, start_date)
        rows = allocation["rows"]
        totals = allocation["totals"]
        end_date = allocation["end_date"]
        source_months = allocation["source_months"]
        warning = allocation["explanation"]
    else:
        inflow = kopecks(selected["прогноз_зачислений"])
        outflow = kopecks(selected["прогноз_списаний"])
        totals = {"inflow": inflow / 100, "outflow": outflow / 100, "net_flow": (inflow - outflow) / 100}
        rows = [{"date": period, "source_period": period, **totals,
                 "cumulative_net_flow": totals["net_flow"]}]
        end_date = None
        source_months = [selected]
        warning = "Месячный прогноз. Для распределения на 14 дней запустите сервер с --daily-allocation."
    rows = [{**row, "closing_balance": None if balance_cents is None else
             (balance_cents + kopecks(row["cumulative_net_flow"])) / 100} for row in rows]
    first_negative = next((row["date"] for row in rows
                           if row["closing_balance"] is not None and row["closing_balance"] < 0), None)
    with (Path(__file__).parent / "liquidity_ui" / "demo_dashboard.json").open(encoding="utf-8") as stream:
        data = json.load(stream)
    data["liquidityAnalysis"] = {
        "series": [
            {"id": "inflow", "label": "Поступления", "color": "#5144d8"},
            {"id": "outflow", "label": "Списания", "color": "#c018a5"},
            {"id": "balance" if balance_cents is not None else "net",
             "label": "Сценарный остаток" if balance_cents is not None else "Чистый поток",
             "color": "#078f7e"},
        ],
        "categories": [{"label": row["date"], "inflow": row["inflow"], "outflow": row["outflow"],
                        "net": row["net_flow"], "balance": row["closing_balance"]} for row in rows],
        "rows": rows, "totals": totals,
        "context": {
            "inn": inn, "period": period, "startDate": start_date, "endDate": end_date,
            "availablePeriods": periods, "availableDates": dates,
            "dailyAllocationEnabled": daily_enabled, "isDailyModel": False,
            "openingBalance": None if balance_cents is None else balance_cents / 100,
            "closingBalance": rows[-1]["closing_balance"], "firstNegativeDate": first_negative,
            "modelName": selected["название_модели"], "historyEnd": store.metadata.get("last_complete_month"),
            "source": "forecasts_api.parquet", "sourceMonths": source_months, "warning": warning,
        },
    }
    data["demoNotice"] = (
        "К прогнозу подключён только виджет «Анализ ликвидности». "
        "Остальные карточки, остаток в шапке, рекомендации и чат — демонстрационные данные дизайна, "
        "они не относятся к выбранному ИНН."
    )
    return data
