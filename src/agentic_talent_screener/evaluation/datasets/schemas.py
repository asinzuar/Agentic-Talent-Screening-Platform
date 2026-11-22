"""Dataset schemas for the evaluation harness (offline, synthetic or public data only)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FactKind = Literal["skill", "role", "date_range", "education", "institution"]
EvidenceLevel = Literal["E1", "E2", "E3", "E4"]


class GoldFact(BaseModel):
    """One ground-truth fact, known by construction because the resume text was generated from it.

    ``value`` is normalised (dates as ``YYYY-MM``); ``source_text`` is the exact rendered line that
    carries the fact, so predicted facts can be matched by kind + value + span overlap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FactKind
    value: str = Field(min_length=1)
    evidence_level: EvidenceLevel | None = None  # skills only
    section: str
    source_text: str
    role_index: int | None = None  # which role a role/date_range fact (or an E3/E4 skill) belongs to


class Identity(BaseModel):
    """Swappable identity fields (E10 fairness swaps). Nothing else in a resume depends on them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    gender: Literal["female", "male", "nonbinary"]
    college: str
    location: str
