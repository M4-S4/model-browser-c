from typing import Any


def never_applies(request: Any) -> bool:
    return False


def ledger_total(context: Any) -> float:
    return float(context.events.total())
