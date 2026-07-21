from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CATEGORIES = [
    "CUSTOMER_PAYMENTS",
    "ACQUIRING",
    "SALARY",
    "TAX",
    "SUPPLIERS",
    "RENT",
    "LOAN_PAYMENT",
    "LOAN_INFLOW",
    "INTERNAL_TRANSFER",
    "OTHER",
]
CLIENT_TYPES = [
    "stable",
    "growing",
    "seasonal",
    "declining_income",
    "growing_expenses",
    "concentrated_income",
    "rare_large_expenses",
]

CREDIT_PROBS = np.array([0.50, 0.23, 0.01, 0.0, 0.0, 0.0, 0.0, 0.08, 0.12, 0.06])
DEBIT_PROBS = np.array([0.0, 0.0, 0.20, 0.10, 0.32, 0.09, 0.07, 0.0, 0.13, 0.09])


@dataclass(frozen=True)
class SyntheticConfig:
    n_clients: int = 3000
    n_months: int = 24
    random_seed: int = 42
    target_gap_rate: float = 0.10
    noise_level: float = 0.15
    overdraft_share: float = 0.55

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SyntheticConfig":
        return cls(**{key: values[key] for key in cls.__annotations__ if key in values})


def _profile_factors(profile_index: np.ndarray, month_index: int, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    progress = month_index / max(n_months - 1, 1)
    credit = np.ones(len(profile_index))
    debit = np.ones(len(profile_index))
    credit[profile_index == 1] = 1.0 + 0.55 * progress
    credit[profile_index == 2] = 1.0 + 0.30 * np.sin(2 * np.pi * month_index / 12)
    credit[profile_index == 3] = 1.0 - 0.42 * progress
    debit[profile_index == 4] = 1.0 + 0.50 * progress
    return np.clip(credit, 0.35, None), debit


def _active_weights(
    rng: np.random.Generator,
    n_clients: int,
    n_days: int,
    active_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random((n_clients, n_days)) < active_probability[:, None]
    # Always keep at least one active day per client.
    missing = ~mask.any(axis=1)
    if missing.any():
        mask[np.flatnonzero(missing), rng.integers(0, n_days, size=int(missing.sum()))] = True
    raw = rng.gamma(1.4, 1.0, size=(n_clients, n_days)) * mask
    weights = raw / raw.sum(axis=1, keepdims=True)
    counts = np.where(mask, rng.poisson(1.5, size=(n_clients, n_days)) + 1, 0)
    return weights, counts


def generate_synthetic_dataset(
    config: SyntheticConfig,
    output_dir: Path,
    source_dataset: str = "synthetic",
) -> dict[str, Any]:
    """Generate hidden daily data and model-visible monthly category aggregates.

    Data is streamed to Parquet one month at a time so the 3,000 x 24 default
    remains practical on a laptop.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily_hidden.parquet"
    monthly_path = output_dir / "monthly_aggregates.parquet"
    target_path = output_dir / "target.parquet"
    liquidity_path = output_dir / "monthly_liquidity.parquet"

    rng = np.random.default_rng(config.random_seed)
    n = config.n_clients
    client_ids = np.array([f"client_{index:06d}" for index in range(1, n + 1)])
    profile_index = rng.integers(0, len(CLIENT_TYPES), size=n)
    client_types = np.array(CLIENT_TYPES, dtype=object)[profile_index]
    has_overdraft = rng.random(n) < config.overdraft_share
    base_credit = rng.lognormal(mean=np.log(900_000), sigma=0.72, size=n)
    overdraft = np.where(has_overdraft, base_credit * rng.uniform(0.12, 0.45, n), 0.0)
    balance = base_credit * rng.uniform(0.18, 0.75, n)
    expense_ratio = rng.uniform(0.72, 0.95, n)

    end_month = pd.Timestamp.today().normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    months = pd.date_range(end=end_month, periods=config.n_months, freq="MS")
    daily_writer: pq.ParquetWriter | None = None
    monthly_frames: list[pd.DataFrame] = []
    gap_by_month: list[np.ndarray] = []
    liquidity_frames: list[pd.DataFrame] = []
    total_daily_rows = 0

    try:
        for month_index, month in enumerate(months):
            n_days = int(month.days_in_month)
            dates = pd.date_range(month, periods=n_days, freq="D")
            credit_factor, debit_factor = _profile_factors(profile_index, month_index, config.n_months)
            local_noise = max(config.noise_level, 0.001)
            credit_month = base_credit * credit_factor * rng.lognormal(0, local_noise * 0.32, n)
            debit_month = base_credit * expense_ratio * debit_factor * rng.lognormal(0, local_noise * 0.26, n)

            profile_risk = np.array([0.75, 0.65, 1.05, 1.35, 1.35, 1.25, 1.20])[profile_index]
            raw_probability = config.target_gap_rate * profile_risk / profile_risk.mean()
            stress = rng.random(n) < np.clip(raw_probability, 0.002, 0.75)
            opening_month_balance = np.minimum(balance, base_credit * 2.5)
            # Non-stress months can have negative flow, but are solvent over the full month.
            affordable = (opening_month_balance + overdraft + credit_month) * 0.97
            debit_month[~stress] = np.minimum(debit_month[~stress], affordable[~stress])

            # Client archetypes affect transaction concentration and shocks.
            credit_active = np.full(n, 0.54)
            credit_active[profile_index == 5] = 0.09
            debit_active = np.full(n, 0.62)
            debit_active[profile_index == 6] = 0.22
            credit_weights, credit_counts = _active_weights(rng, n, n_days, credit_active)
            debit_weights, debit_counts = _active_weights(rng, n, n_days, debit_active)
            daily_credit = credit_month[:, None] * credit_weights
            daily_debit = debit_month[:, None] * debit_weights

            stress_day = rng.integers(max(1, n_days // 3), n_days, size=n)
            stress_clients = np.flatnonzero(stress)
            if len(stress_clients):
                # This liquidity shock guarantees an actual gap even after all monthly receipts.
                shock = (
                    opening_month_balance
                    + overdraft
                    + credit_month
                    + np.maximum(0.08 * credit_month, 1.0)
                )
                daily_debit[stress_clients, stress_day[stress_clients]] += shock[stress_clients]
                debit_counts[stress_clients, stress_day[stress_clients]] += 1

            credit_cat = rng.choice(len(CATEGORIES), size=(n, n_days), p=CREDIT_PROBS)
            debit_cat = rng.choice(len(CATEGORIES), size=(n, n_days), p=DEBIT_PROBS)
            openings = np.empty_like(daily_credit)
            closings = np.empty_like(daily_credit)
            gaps = np.empty_like(daily_credit)
            current_balance = opening_month_balance
            deferred_debit = np.zeros(n)
            non_stress = ~stress
            for day in range(n_days):
                openings[:, day] = current_balance
                # A solvent client defers an uncovered payment instead of creating a
                # false cash-gap event solely because receipts arrive later in the month.
                planned_debit = daily_debit[:, day] + deferred_debit
                available_today = np.maximum(current_balance + daily_credit[:, day] + overdraft, 0.0)
                paid_today = np.minimum(planned_debit, available_today)
                daily_debit[non_stress, day] = paid_today[non_stress]
                deferred_debit[non_stress] = (planned_debit - paid_today)[non_stress]
                zero_payment = daily_debit[:, day] <= 0
                debit_counts[zero_payment, day] = 0
                moved_payment = (daily_debit[:, day] > 0) & (debit_counts[:, day] == 0)
                debit_counts[moved_payment, day] = 1
                liquidity = current_balance + daily_credit[:, day] - daily_debit[:, day]
                gaps[:, day] = np.maximum(-(liquidity + overdraft), 0.0)
                current_balance = np.maximum(liquidity, -overdraft)
                closings[:, day] = current_balance
            balance = current_balance
            month_gap = (gaps > 0).any(axis=1)
            gap_by_month.append(month_gap)
            liquidity_frames.append(
                pd.DataFrame(
                    {
                        "client_id": client_ids,
                        "month": month.strftime("%Y-%m"),
                        "opening_balance": openings[:, 0],
                        "closing_balance": closings[:, -1],
                        "minimum_observed_balance": closings.min(axis=1),
                        "maximum_observed_balance": closings.max(axis=1),
                        "balance_observations_count": n_days,
                        "balance_breach_proxy": (closings.min(axis=1) + overdraft < 0).astype(np.int8),
                    }
                )
            )

            repeated_clients = np.repeat(client_ids, n_days)
            repeated_types = np.repeat(client_types, n_days)
            daily = pd.DataFrame(
                {
                    "client_id": repeated_clients,
                    "date": np.tile(dates.to_numpy(), n),
                    "client_type": repeated_types,
                    "credit_sum": daily_credit.ravel(),
                    "debit_sum": daily_debit.ravel(),
                    "credit_nonzero_count": credit_counts.ravel(),
                    "debit_nonzero_count": debit_counts.ravel(),
                    "credit_category": np.array(CATEGORIES, dtype=object)[credit_cat.ravel()],
                    "debit_category": np.array(CATEGORIES, dtype=object)[debit_cat.ravel()],
                    "opening_balance": openings.ravel(),
                    "closing_balance": closings.ravel(),
                    "available_overdraft": np.repeat(overdraft, n_days),
                    "cash_gap_amount": gaps.ravel(),
                    "cash_gap": (gaps.ravel() > 0).astype(np.int8),
                }
            )
            table = pa.Table.from_pandas(daily, preserve_index=False)
            if daily_writer is None:
                daily_writer = pq.ParquetWriter(daily_path, table.schema, compression="zstd")
            daily_writer.write_table(table)
            total_daily_rows += len(daily)

            month_string = month.strftime("%Y-%m")
            for category_index, category in enumerate(CATEGORIES):
                credit_mask = credit_cat == category_index
                debit_mask = debit_cat == category_index
                monthly_frames.append(
                    pd.DataFrame(
                        {
                            "client_id": client_ids,
                            "month": month_string,
                            "transaction_label": category,
                            "debit_sum": (daily_debit * debit_mask).sum(axis=1),
                            "credit_sum": (daily_credit * credit_mask).sum(axis=1),
                            "debit_nonzero_count": (debit_counts * debit_mask).sum(axis=1),
                            "credit_nonzero_count": (credit_counts * credit_mask).sum(axis=1),
                            "source_dataset": source_dataset,
                            "source_provider": "synthetic",
                            "currency": "SYN",
                            "data_quality_flags": None,
                        }
                    )
                )
    finally:
        if daily_writer is not None:
            daily_writer.close()

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly["month"] = monthly["month"].astype("string")
    monthly["transaction_label"] = monthly["transaction_label"].astype("category")
    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    pd.concat(liquidity_frames, ignore_index=True).to_parquet(liquidity_path, index=False, compression="zstd")

    gap_matrix = np.column_stack(gap_by_month)
    target = pd.DataFrame(
        {
            "client_id": np.repeat(client_ids, config.n_months),
            "month": np.tile(months.strftime("%Y-%m"), n),
            "cash_gap_current_month": gap_matrix.ravel().astype(np.int8),
            "cash_gap_next_month": np.column_stack(
                [gap_matrix[:, 1:], np.full((n, 1), np.nan)]
            ).ravel(),
        }
    )
    target.to_parquet(target_path, index=False, compression="zstd")
    valid_target = target["cash_gap_next_month"].dropna()
    return {
        "daily_rows": total_daily_rows,
        "monthly_rows": int(len(monthly)),
        "model_observations": int(len(valid_target)),
        "n_clients": n,
        "n_months": config.n_months,
        "first_month": months[0].strftime("%Y-%m"),
        "last_month": months[-1].strftime("%Y-%m"),
        "target_rate": float(valid_target.mean()),
        "requested_target_rate": config.target_gap_rate,
        "files": {
            "daily_hidden": str(daily_path),
            "monthly_aggregates": str(monthly_path),
            "target": str(target_path),
            "liquidity": str(liquidity_path),
        },
    }
