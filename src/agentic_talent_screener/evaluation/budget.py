"""Spend guard. The evaluation layer is free-tier only: any model id not ending in ':free' is refused."""

from __future__ import annotations

import math
import re


class BudgetExceeded(RuntimeError):
    pass


PAID_NEEDS_APPROVAL = "paid runs need user approval and a separate task; only ':free' models are allowed"


# Exactly vendor/model plus the single ':free' suffix; ASCII only, no variant tags, no whitespace or control chars.
_FREE_MODEL = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*:free", re.ASCII)


def is_free_model(model: str) -> bool:
    return isinstance(model, str) and _FREE_MODEL.fullmatch(model) is not None


class BudgetGuard:
    """Budget is fixed at 0 USD. Kept as a class so callers have one place to ask permission."""

    def __init__(self, budget_usd: float = 0.0) -> None:
        if not math.isfinite(budget_usd) or budget_usd < 0:
            raise ValueError("budget_usd must be a finite number >= 0")
        if budget_usd != 0:
            raise ValueError(f"budget_usd must be 0: {PAID_NEEDS_APPROVAL}")
        self.budget_usd = 0.0

    def check(self, model: str) -> None:
        """Raise before any network call for a model that is not ':free'."""
        if not is_free_model(model):
            raise BudgetExceeded(f"model {model!r} refused: {PAID_NEEDS_APPROVAL}")
