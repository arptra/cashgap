from __future__ import annotations

from typing import Any


def build_compatibility_report(profile: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    supported = set(source.get("supported_tasks", []))
    limitations = set(source.get("limitations", []))
    reasons: list[str] = []

    classification = bool(profile.get("has_cash_gap_target"))
    if not classification:
        reasons.append("No agreed cash-gap target; fraud/AML/anomaly labels are not accepted")
    if profile.get("clients", 0) < 2 or "one_company" in limitations or "not_client_level" in limitations:
        classification = False
        reasons.append("Client-level classification requires at least two clients")
    if profile.get("months", 0) < 6:
        classification = False
        reasons.append("At least six monthly periods are required")
    if not (profile.get("has_debit") and profile.get("has_credit")):
        classification = False
        reasons.append("Cash-gap classification requires both debit and credit flows")

    forecasting = bool(
        profile.get("months", 0) >= 6
        and (profile.get("has_debit") or profile.get("has_credit"))
        and ({"flow_forecasting", "debit_flow_forecasting"} & supported or source.get("id") == "synthetic")
    )
    proxy = bool(
        profile.get("has_balance")
        or "proxy_risk" in supported
        or "balance_breach_proxy" in supported
    )
    categorization = bool("transaction_categorization" in supported)

    return {
        "classification_eligible": classification,
        "forecasting_eligible": forecasting,
        "proxy_eligible": proxy,
        "categorization_eligible": categorization,
        "reasons": list(dict.fromkeys(reasons)),
        "limitations": sorted(limitations),
        "supported_tasks": sorted(supported),
    }

