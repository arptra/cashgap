#!/usr/bin/env python3
"""Простой помесячный прогноз прихода и расхода по ИНН на 2025 год.

Можно перенести только этот файл. Зависимости:
    python -m pip install numpy pandas pyarrow scikit-learn

Запуск:
    python forecast_2025.py --inflow inflow.parquet --outflow outflow.parquet

Приход: tr_date, kt_inn, tr_sum. Расход: tr_date, dt_inn, tr_sum.
Суммы неотрицательные, в одной валюте. История должна содержать полные
календарные месяцы до 31 декабря предыдущего года включительно. Месяцы без
операций считаются нулевыми. Операции прогнозного года и позднее игнорируются.
Модель обучается один раз на CPU; все 12 месяцев прогнозируются из одной
точки — конца предыдущего года, без подстановки будущего факта.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
except ImportError as error:
    raise SystemExit(
        "Не хватает библиотек. Выполните: python -m pip install numpy pandas pyarrow scikit-learn"
    ) from error


FLOWS = ["inflow", "outflow"]
PREDICTIONS = ["predicted_inflow", "predicted_outflow", "predicted_net_flow"]
MIN_HISTORY = 3


def parse_dates(values):
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        typical = numeric.dropna().abs().median()
        if 10_000_000 <= typical <= 99_999_999:
            if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
                raise ValueError("Некорректная числовая дата YYYYMMDD.")
            result = pd.to_datetime(numeric.astype("int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            unit = ("s" if 1e9 <= typical < 1e11 else
                    "ms" if 1e11 <= typical < 1e14 else
                    "us" if 1e14 <= typical < 1e17 else
                    "ns" if 1e17 <= typical < 1e20 else
                    "D" if 10_000 <= typical < 1_000_000 else None)
            if unit is None:
                raise ValueError("Неизвестный числовой формат tr_date; используйте дату или YYYY-MM-DD.")
            result = pd.to_datetime(numeric, unit=unit, origin="unix", errors="coerce")
    else:
        # pandas 2 infers one string format for the entire column unless mixed
        # parsing is requested. pandas 1 (Python 3.8 installations) already mixes.
        options = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
        if (not pd.api.types.is_datetime64_any_dtype(values.dtype)
                and values.astype("string").str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False).any()):
            # pandas 3 rejects mixed offsets; parse those rare textual inputs
            # individually so the source calendar day survives conversion.
            try:
                result = pd.to_datetime(values.map(lambda v: pd.Timestamp(v).tz_localize(None).normalize()))
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError("В tr_date есть некорректные даты или часовые пояса.") from error
        else:
            result = pd.to_datetime(values, errors="coerce", **options)
    if result.isna().any():
        raise ValueError("В tr_date есть пустые или некорректные даты.")
    if not pd.api.types.is_datetime64_any_dtype(result.dtype):
        # Mixed UTC offsets produce object dtype. Keep each source's local day.
        try:
            return pd.to_datetime(result.map(lambda value: pd.Timestamp(value).tz_localize(None).normalize()))
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("Не удалось определить календарную дату tr_date.") from error
    if result.dt.tz is not None:
        # Preserve the calendar day recorded by the source, rather than converting to UTC.
        result = result.dt.tz_localize(None)
    return result.dt.normalize()


def normalize_inn(values):
    error = "ИНН должен быть строкой цифр или точным неотрицательным целым числом."
    if pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(error)
    if pd.api.types.is_float_dtype(values.dtype):
        dtype = getattr(values.dtype, "numpy_dtype", values.dtype)
        limit = (1 << (np.finfo(dtype).nmant + 1)) - 1
        numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if (not np.isfinite(numbers).all() or (numbers < 0).any()
                or (numbers > limit).any() or (numbers != np.floor(numbers)).any()):
            raise ValueError(error)
        result = pd.Series(numbers.astype(np.int64), index=values.index).astype("string")
    else:
        result = values.astype("string").str.strip()
    if result.isna().any() or not result.str.fullmatch(r"[0-9]+").all():
        raise ValueError(error)
    return result


def read_monthly(path, inn_column, flow, cutoff):
    """Read only three columns in batches; retain monthly aggregates, not raw transactions."""
    dataset = ds.dataset(str(path), format="parquet")
    columns = {}
    for expected in ("tr_date", inn_column, "tr_sum"):
        key = "".join(c for c in expected.casefold() if c.isalnum())
        matches = [name for name in dataset.schema.names
                   if "".join(c for c in name.casefold() if c.isalnum()) == key]
        if len(matches) != 1:
            raise ValueError("{}: не найдена однозначная колонка {}. Поля: {}".format(
                path, expected, dataset.schema.names))
        columns[matches[0]] = {"tr_date": "date", inn_column: "inn", "tr_sum": flow}[expected]
    combined = None
    used_rows = ignored_rows = 0
    first_date = last_date = None
    for batch in dataset.scanner(columns=list(columns), batch_size=250_000).to_batches():
        frame = batch.to_pandas().rename(columns=columns)
        if frame.empty:
            continue
        frame["date"] = parse_dates(frame["date"])
        past = frame["date"].lt(cutoff)
        ignored_rows += int((~past).sum())
        frame = frame.loc[past].copy()
        if frame.empty:
            continue
        used_rows += len(frame)
        frame["inn"] = normalize_inn(frame["inn"])
        frame[flow] = pd.to_numeric(frame[flow], errors="coerce")
        amounts = frame[flow].to_numpy(dtype=float)
        if not np.isfinite(amounts).all() or (amounts < 0).any():
            raise ValueError("{}: tr_sum должны быть конечными и неотрицательными.".format(path))
        low, high = frame["date"].min(), frame["date"].max()
        first_date = low if first_date is None else min(first_date, low)
        last_date = high if last_date is None else max(last_date, high)
        frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
        monthly = frame.groupby(["inn", "month"])[flow].sum()
        combined = monthly if combined is None else combined.add(monthly, fill_value=0.0)
    if combined is None:
        combined = pd.Series(dtype=float, name=flow,
                             index=pd.MultiIndex.from_arrays([[], []], names=["inn", "month"]))
    combined.name = flow
    return combined, {
        "path": str(Path(path).resolve()), "used_transactions": used_rows,
        "ignored_future_transactions": ignored_rows,
        "first_transaction": None if first_date is None else str(first_date.date()),
        "last_transaction": None if last_date is None else str(last_date.date()),
    }


def load_history(inflow, outflow, year):
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    credit, credit_info = read_monthly(inflow, "kt_inn", "inflow", cutoff)
    debit, debit_info = read_monthly(outflow, "dt_inn", "outflow", cutoff)
    observed = pd.concat([credit, debit], axis=1).fillna(0.0).reset_index()
    if observed.empty:
        raise ValueError("Нет истории до {}. Для прогноза {} года нужны данные прошлых лет.".format(
            cutoff.date(), year))
    if not np.isfinite(observed[FLOWS].to_numpy(float)).all():
        raise ValueError("Переполнение при суммировании операций.")
    # Retain dormant companies through December, including months with no transactions.
    last_month = cutoff - pd.offsets.MonthBegin(1)
    parts = []
    for inn, group in observed.groupby("inn", sort=True):
        dates = pd.date_range(group["month"].min(), last_month, freq="MS")
        panel = group.set_index("month")[FLOWS].reindex(dates, fill_value=0.0)
        panel.index.name = "month"
        panel["inn"] = str(inn)
        parts.append(panel.reset_index())
    return pd.concat(parts, ignore_index=True), {"inflow": credit_info, "outflow": debit_info}


def make_features(panel):
    """One row predicts its month using only strictly preceding calendar months."""
    ordered = panel.sort_values(["inn", "month"]).reset_index(drop=True)
    grouped = ordered.groupby("inn", sort=False)
    X = pd.DataFrame(index=ordered.index)
    for flow in FLOWS:
        for lag in (1, 2, 3, 6, 12):
            X["{}_lag_{}".format(flow, lag)] = grouped[flow].shift(lag)
        for window in (3, 6, 12):
            X["{}_mean_{}".format(flow, window)] = grouped[flow].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            X["{}_std_{}".format(flow, window)] = grouped[flow].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std(ddof=0))
    X["month_sin"] = np.sin(2 * np.pi * ordered["month"].dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * ordered["month"].dt.month / 12)
    history_length = grouped.cumcount()
    X = X.fillna(0.0)
    baseline = X[["inflow_mean_3", "outflow_mean_3"]].to_numpy(float)
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("Признаки содержат неконечные значения; проверьте масштаб сумм.")
    return ordered, X, baseline, history_length


def forecast_year(history, year, alpha=1000.0):
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    if history["month"].ge(cutoff).any():
        raise ValueError("Обучающая история не должна содержать прогнозный год.")
    ordered, X, baseline, history_length = make_features(history)
    train_mask = history_length.ge(MIN_HISTORY)
    if not train_mask.any():
        raise ValueError("Нужно минимум 4 календарных месяца истории хотя бы одного ИНН до {} года.".format(year))
    scaler = StandardScaler().fit(X.loc[train_mask])
    active = scaler.var_ > 1e-12

    def transform(values):
        normalized = scaler.transform(values)
        normalized[:, ~active] = 0.0
        return np.clip(normalized, -10.0, 10.0)

    target = ordered.loc[train_mask, FLOWS].to_numpy(float) - baseline[train_mask]
    model = Ridge(alpha=alpha)
    with threadpool_limits(limits=2):
        model.fit(transform(X.loc[train_mask]), target)
        # Only twelve historical months are needed for recursive features.
        state = ordered.groupby("inn", sort=False).tail(12).copy()
        inns = sorted(ordered["inn"].unique())
        forecasts = []
        for step, month in enumerate(pd.date_range(cutoff, periods=12, freq="MS"), start=1):
            future = pd.DataFrame({"inn": inns, "month": month, "inflow": 0.0, "outflow": 0.0})
            augmented, features, base, _ = make_features(pd.concat([state, future], ignore_index=True))
            mask = augmented["month"].eq(month)
            predicted = np.maximum(base[mask] + model.predict(transform(features.loc[mask])), 0.0)
            if not np.isfinite(predicted).all():
                raise ValueError("Модель выдала неконечный прогноз.")
            row = pd.DataFrame({
                "inn": augmented.loc[mask, "inn"].to_numpy(), "month": month.strftime("%Y-%m"),
                "predicted_inflow": predicted[:, 0], "predicted_outflow": predicted[:, 1],
                "predicted_net_flow": predicted[:, 0] - predicted[:, 1], "forecast_step": step,
            })
            forecasts.append(row)
            augmented.loc[mask, FLOWS] = predicted
            state = augmented.groupby("inn", sort=False).tail(12).copy()
    result = pd.concat(forecasts, ignore_index=True).sort_values(["inn", "month"]).reset_index(drop=True)
    # Round each client's flows before computing net and portfolio/year totals.
    # This keeps all exported CSVs consistent down to the displayed monetary unit.
    result[["predicted_inflow", "predicted_outflow"]] = result[["predicted_inflow", "predicted_outflow"]].round(2)
    result["predicted_net_flow"] = (result["predicted_inflow"] - result["predicted_outflow"]).round(2)
    return result, {
        "model": "Ridge correction to trailing 3-month mean", "alpha": alpha,
        "training_rows": int(train_mask.sum()), "features": list(X.columns),
        "history_first_month": str(ordered["month"].min().date()),
        "history_last_month": str(ordered["month"].max().date()),
        "clients": len(inns), "forecast_year": year, "training_cutoff_exclusive": str(cutoff.date()),
        "forecast_method": "12 recursive months; no actuals from forecast year",
        "input_assumption": "Complete historical calendar months through December 31 of the preceding year; missing months are zero.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inflow", required=True, help="Parquet прихода: tr_date, kt_inn, tr_sum")
    parser.add_argument("--outflow", required=True, help="Parquet расхода: tr_date, dt_inn, tr_sum")
    parser.add_argument("--year", type=int, default=2025, help="Год прогноза (по умолчанию 2025)")
    parser.add_argument("--output-dir", help="Папка результатов (по умолчанию artifacts/forecast_<год>)")
    args = parser.parse_args(argv)
    if not 1971 <= args.year <= 2200:
        parser.error("--year должен быть между 1971 и 2200")
    try:
        print("Читаю данные до 01.01.{}; более поздние операции исключаются.".format(args.year), flush=True)
        history, sources = load_history(args.inflow, args.outflow, args.year)
        for flow, info in sources.items():
            print("{}: операций {:,}, последняя историческая дата {}, исключено будущих {:,}".format(
                flow, info["used_transactions"], info["last_transaction"], info["ignored_future_transactions"]))
        print("Предполагается полная выгрузка по 31.12.{}; месяцы без операций = 0.".format(args.year - 1))
        print("Обучаю Ridge на CPU и прогнозирую 12 месяцев…", flush=True)
        forecast, info = forecast_year(history, args.year)
        monthly = forecast.groupby("month", as_index=False)[PREDICTIONS].sum()
        annual = forecast.groupby("inn", as_index=False)[PREDICTIONS].sum()
        if not np.isfinite(monthly[PREDICTIONS].to_numpy()).all() or not np.isfinite(annual[PREDICTIONS].to_numpy()).all():
            raise ValueError("Переполнение итоговых сумм.")
        output = Path(args.output_dir or "artifacts/forecast_{}".format(args.year))
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in (("forecast_by_inn.csv", forecast), ("monthly_totals.csv", monthly),
                            ("annual_by_inn.csv", annual)):
            frame.to_csv(output / name, index=False, encoding="utf-8-sig", float_format="%.2f")
        (output / "run_info.json").write_text(
            json.dumps({**info, "sources": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nПрогноз по всем ИНН (суммы в валюте исходных файлов):")
        print(monthly.to_string(index=False, float_format=lambda x: "{:,.2f}".format(x)))
        totals = monthly[PREDICTIONS].sum()
        print("\nЗа {} год: приход {:,.2f}; расход {:,.2f}; чистый поток {:,.2f}".format(
            args.year, *totals.tolist()))
        print("Результаты: {}".format(output.resolve()))
        return 0
    except (ValueError, OSError) as error:
        print("Ошибка: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
