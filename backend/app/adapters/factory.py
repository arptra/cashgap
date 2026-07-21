from __future__ import annotations

from app.adapters.agami_indian_statements import AgamiIndianStatementsAdapter
from app.adapters.banksim import BankSimAdapter
from app.adapters.base import AdapterError, BaseAdapter
from app.adapters.ibm_aml import IbmAmlAdapter
from app.adapters.mindweave_us import MindweaveUsAdapter
from app.adapters.paysim import PaySimAdapter
from app.adapters.shell_cashflow import ShellCashflowAdapter
from app.adapters.transaction_categorization import TransactionCategorizationAdapter


ADAPTERS: dict[str, type[BaseAdapter]] = {
    "paysim": PaySimAdapter,
    "banksim": BankSimAdapter,
    "ibm_aml": IbmAmlAdapter,
    "agami_indian_statements": AgamiIndianStatementsAdapter,
    "mindweave_us": MindweaveUsAdapter,
    "shell_cashflow": ShellCashflowAdapter,
    "transaction_categorization": TransactionCategorizationAdapter,
}


def create_adapter(source: dict, options: dict | None = None) -> BaseAdapter:
    name = (options or {}).get("adapter") or source.get("adapter")
    adapter_class = ADAPTERS.get(name)
    if adapter_class is None:
        raise AdapterError(
            f"Adapter '{name}' cannot be inferred. Choose one of: {', '.join(sorted(ADAPTERS))}"
        )
    return adapter_class(source, options)

