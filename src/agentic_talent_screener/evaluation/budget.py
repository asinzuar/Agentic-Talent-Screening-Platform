"""Spend guard. Cap defaults to 0 USD, which allows only ':free' models."""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    pass


def is_free_model(model: str) -> bool:
    return model.endswith(":free")


class BudgetGuard:
    def __init__(self, budget_usd: float = 0.0) -> None:
        if budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def check(self, model: str, estimated_usd: float = 0.0) -> None:
        """Raise before an uncached call that the budget does not cover."""
        if is_free_model(model):
            return
        if self.budget_usd <= 0:
            raise BudgetExceeded(f"model {model!r} is not ':free' and --budget-usd is 0")
        if self.spent_usd + estimated_usd > self.budget_usd:
            raise BudgetExceeded(f"budget {self.budget_usd} USD would be exceeded by model {model!r}")

    def record(self, usd: float) -> None:
        self.spent_usd += max(usd, 0.0)
