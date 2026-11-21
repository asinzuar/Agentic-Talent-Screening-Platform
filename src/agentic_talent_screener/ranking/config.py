"""Centralised, validated scoring configuration (SPEC §6.1, §6.3, §13)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import EvidenceLevel


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LadderCredits(_Frozen):
    """Credit per evidence level. E1 is deliberately low: keyword stuffing is capped there."""

    e0: float = Field(default=0.0, ge=0.0, le=1.0)
    e1: float = Field(default=0.10, ge=0.0, le=1.0)
    e2: float = Field(default=0.40, ge=0.0, le=1.0)
    e3: float = Field(default=0.75, ge=0.0, le=1.0)
    e4: float = Field(default=1.00, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _monotone(self) -> LadderCredits:
        vals = [self.e0, self.e1, self.e2, self.e3, self.e4]
        if vals != sorted(vals):
            raise ValueError("ladder credits must be non-decreasing from E0 to E4")
        if self.e0 != 0.0:
            raise ValueError("E0 (no evidence) must earn zero credit")
        return self

    def credit(self, level: EvidenceLevel | int) -> float:
        return (self.e0, self.e1, self.e2, self.e3, self.e4)[int(level)]


class TierCutoffs(_Frozen):
    a: float = Field(default=0.70, ge=0.0, le=1.0)
    b: float = Field(default=0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> TierCutoffs:
        if not self.a > self.b:
            raise ValueError("tier A cutoff must be above tier B cutoff")
        return self


class ScoringConfig(_Frozen):
    config_version: str = "scoring-v0.3"

    credits: LadderCredits = LadderCredits()
    tiers: TierCutoffs = TierCutoffs()

    # Requirement matching _(inferred)_
    adjacent_factor: float = Field(default=0.6, ge=0.0, le=1.0)  # credit = factor*sim*item credit
    mandatory_multiplier: float = Field(default=2.0, ge=1.0)
    education_credit: float = Field(default=1.0, ge=0.0, le=1.0)  # a grounded education fact (no ladder)

    # Soft must-have gate (shape is _(inferred)_, SPEC §13.5)
    gate_threshold: float = Field(default=0.70, gt=0.0, le=1.0)
    gate_floor: float = Field(default=0.50, ge=0.0, le=1.0)

    # L4: which flags are excluded from the final score
    exclude_flag_risks: tuple[str, ...] = ("medium", "high")

    # Shortlist: how many candidates just below the cut-off to point out
    near_cutoff_count: int = Field(default=3, ge=0)


DEFAULT_CONFIG = ScoringConfig()
