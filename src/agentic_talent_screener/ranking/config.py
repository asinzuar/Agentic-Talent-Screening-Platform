"""Centralised, validated scoring configuration (SPEC §6.1, §6.4, §13)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import EvidenceLevel


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentWeights(_Frozen):
    skill_match: float = Field(default=0.45, ge=0.0, le=1.0)
    experience: float = Field(default=0.20, ge=0.0, le=1.0)
    projects: float = Field(default=0.15, ge=0.0, le=1.0)
    education: float = Field(default=0.10, ge=0.0, le=1.0)
    tenure: float = Field(default=0.05, ge=0.0, le=1.0)
    location: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> ComponentWeights:
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"component weights must sum to 1.0, got {total:.6f}")
        if self.skill_match <= 0.0:
            raise ValueError("skill_match weight must be > 0 (it is always present)")
        return self

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


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
    config_version: str = "scoring-v0.1"

    weights: ComponentWeights = ComponentWeights()
    credits: LadderCredits = LadderCredits()
    tiers: TierCutoffs = TierCutoffs()

    # Skill matching
    adjacent_factor: float = Field(default=0.6, ge=0.0, le=1.0)  # credit = factor*sim*best_credit
    mandatory_multiplier: float = Field(default=2.0, ge=1.0)

    # Soft must-have gate (SPEC §12 Q-style assumption: shape of the penalty is _(inferred)_)
    gate_threshold: float = Field(default=0.70, gt=0.0, le=1.0)
    gate_min_level: EvidenceLevel = EvidenceLevel.E2
    gate_floor: float = Field(default=0.50, ge=0.0, le=1.0)

    # Component shapes _(inferred)_
    experience_target_years: float = Field(default=4.0, gt=0.0)
    internship_month_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    projects_target: float = Field(default=2.0, gt=0.0)
    education_level_scores: dict[str, float] = {
        "secondary": 0.30,
        "diploma": 0.50,
        "bachelor": 0.70,
        "master": 0.85,
        "doctorate": 1.00,
    }
    education_field_blend: float = Field(default=0.4, ge=0.0, le=1.0)
    tenure_min_job_roles: int = Field(default=3, ge=3)  # never score tenure on < 3 roles
    tenure_floor: float = Field(default=0.60, ge=0.5, le=1.0)  # bounds the penalty (fairness)
    tenure_full_months: float = Field(default=24.0, gt=0.0)

    # L4: which flags are excluded from the final score
    exclude_flag_risks: tuple[str, ...] = ("medium", "high")

    # Uncertainty
    mc_runs: int = Field(default=200, ge=1)
    mc_seed: int = 20260101
    mc_component_sd: float = Field(default=0.05, ge=0.0)
    mc_missing_sd: float = Field(default=0.20, ge=0.0)  # spread assumed for a missing component

    @model_validator(mode="after")
    def _check_education_scores(self) -> ScoringConfig:
        for k, v in self.education_level_scores.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"education score for {k!r} must be in [0, 1]")
        return self


DEFAULT_CONFIG = ScoringConfig()
