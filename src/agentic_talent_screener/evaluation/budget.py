"""Spend guard. Cap defaults to 0 USD, which allows only ':free' models."""

from __future__ import annotations

import math


class BudgetExceeded(RuntimeError):
    pass


def is_free_model(model: str) -> bool:
    return model.endswith(":free")


class BudgetGuard:
    def __init__(self, budget_usd: float = 0.0) -> None:
        if not math.isfinite(budget_usd) or budget_usd < 0:
            raise ValueError("budget_usd must be a finite number >= 0")
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def check(self, model: str, estimated_usd: float = 0.0) -> None:
        """Raise before an uncached call that the budget does not cover."""
        if is_free_model(model):
            return
        if self.budget_usd <= 0:
            raise BudgetExceeded(f"model {model!r} is not ':free' and --budget-usd is 0")
        if not math.isfinite(estimated_usd) or estimated_usd <= 0:
            raise BudgetExceeded(f"paid model {model!r} needs a positive cost estimate; refusing (fail closed)")
        if self.spent_usd + estimated_usd > self.budget_usd:
            raise BudgetExceeded(f"budget {self.budget_usd} USD would be exceeded by model {model!r}")

    def record(self, usd: float) -> None:
        if math.isfinite(usd):
            self.spent_usd += max(usd, 0.0)
