"""EvalConfig: suites, sample sizes, budgets, seeds, targets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Suite = Literal["ranking", "robustness", "detection", "fairness", "extraction"]
SUITES: tuple[str, ...] = ("ranking", "robustness", "detection", "fairness", "extraction")


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suites: tuple[Suite, ...] = SUITES
    systems: tuple[str, ...] = ("full",)
    seed: int = 0
    sample_size: int = Field(default=50, ge=1)
    bootstrap_resamples: int = Field(default=1000, ge=100)
    budget_usd: float = Field(default=0.0, ge=0, le=0.50, allow_inf_nan=False)  # per-run USD ceiling; > 0 needs --allow-paid
    offline: bool = False
    data_dir: str = "data/eval"
    runs_dir: str = "runs"
    cache_dir: str = "data/eval/cache"
    targets: dict[str, float] = Field(default_factory=dict)
    min_verdict_n: int = Field(default=30, ge=1)  # E12: fewer samples than this cannot yield a slide-9 pass
