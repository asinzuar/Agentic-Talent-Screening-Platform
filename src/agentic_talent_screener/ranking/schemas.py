"""Strict Pydantic schemas for the score-based ranking module (SPEC.md §6.3, §6.4, §13).

Design constraints enforced here:

* Blind candidate IDs only (``C-0932``); no identity / PII fields (``extra="forbid"`` plus an
  explicit deny-list so the error message is clear).
* Skills use canonical catalogue IDs (``esco:<slug>`` or project-extension ``ext:<slug>``).
* Requirements are tagged Mandatory (M) / Preferred (P), carry provenance, and live in a
  versioned profile that must be human-approved before anything is scored against it.
* Evidence levels are E0-E4 and every claim cites spans (location + quote) in the resume.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# --------------------------------------------------------------------------------------
# Primitive types
# --------------------------------------------------------------------------------------

BlindCandidateId = Annotated[str, StringConstraints(pattern=r"^C-\d{4}$")]
"""Blind candidate ID, e.g. ``C-0932`` (SPEC §9.2)."""

SkillId = Annotated[str, StringConstraints(pattern=r"^(esco|ext):[a-z0-9][a-z0-9._\-]*$")]
"""Canonical skill ID. ``esco:`` = ESCO catalogue, ``ext:`` = reviewed project extension (§6.2)."""

Score01 = Annotated[float, Field(ge=0.0, le=1.0)]

LocationRef = Annotated[str, StringConstraints(pattern=r"^p\d+:l\d+$")]
"""Span location in the visible resume text, e.g. ``p1:l12`` (SPEC §10)."""

# Field names that must never appear on a candidate record (SPEC §9.2, §8 fairness).
FORBIDDEN_IDENTITY_FIELDS = frozenset(
    {
        "name",
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "gender",
        "sex",
        "age",
        "date_of_birth",
        "dob",
        "nationality",
        "religion",
        "caste",
        "marital_status",
        "photo",
        "institution",
        "college",
        "university",
        "employer",
        "linkedin",
        "url",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EvidenceLevel(IntEnum):
    """Evidence ladder (SPEC §6.1). E0 = no usable evidence (added in SPEC v0.2)."""

    E0 = 0
    E1 = 1
    E2 = 2
    E3 = 3
    E4 = 4


class Tier(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class RequirementTag(StrEnum):
    MANDATORY = "M"
    PREFERRED = "P"


# --------------------------------------------------------------------------------------
# Skill catalogue (adjacency)
# --------------------------------------------------------------------------------------


class AdjacencyEdge(_Strict):
    """Directional: holding ``has_skill`` earns partial credit towards ``counts_towards``."""

    has_skill: SkillId
    counts_towards: SkillId
    similarity: Score01


class SkillCatalogue(_Strict):
    """Versioned slice of the catalogue needed for adjacency lookups (SPEC §6.2)."""

    version: str = Field(min_length=1)
    adjacency: tuple[AdjacencyEdge, ...] = ()


# --------------------------------------------------------------------------------------
# RequirementProfile (trusted lane)
# --------------------------------------------------------------------------------------


class Requirement(_Strict):
    skill_id: SkillId
    label: str = Field(min_length=1)
    tag: RequirementTag
    weight: float = Field(gt=0.0, le=10.0)
    provenance: Literal["jd", "suggested"]
    jd_quote: str | None = None  # required when provenance == "jd"
    rationale: str | None = None  # required when provenance == "suggested"
    onet_source: str | None = None
    accepted: bool = True  # suggested requirements are inactive until a human accepts them

    @model_validator(mode="after")
    def _check_provenance(self) -> Requirement:
        if self.provenance == "jd" and not (self.jd_quote and self.jd_quote.strip()):
            raise ValueError("requirements from the JD must carry the quoted JD text")
        if self.provenance == "suggested" and not (self.rationale and self.rationale.strip()):
            raise ValueError("suggested requirements must carry the agent's rationale")
        return self

    @property
    def active(self) -> bool:
        """Suggested requirements only count once a human has accepted them (§6.3)."""
        return self.provenance == "jd" or self.accepted


class RequirementProfile(_Strict):
    profile_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    role_title: str = Field(min_length=1)
    requirements: tuple[Requirement, ...] = Field(min_length=1)
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> RequirementProfile:
        if self.approved and not (self.approved_by and self.approved_at):
            raise ValueError("an approved profile must record approved_by and approved_at")
        ids = [r.skill_id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("each skill_id may appear only once in a requirement profile")
        return self

    @property
    def active_requirements(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.active)


# --------------------------------------------------------------------------------------
# VerifiedProfile (output of the quarantined lane; the only candidate data scoring sees)
# --------------------------------------------------------------------------------------


class Span(_Strict):
    """A cited span in the resume text (SPEC §10: ``p1:l12``)."""

    location: LocationRef
    quote: str
    visible: bool = True  # False = hidden / off-page / behind-image text per L0


class SkillClaim(_Strict):
    skill_id: SkillId
    evidence_level: EvidenceLevel
    spans: tuple[Span, ...] = ()


class Role(_Strict):
    kind: Literal["job", "internship"]
    months: int = Field(ge=0, le=600)
    skill_ids: tuple[SkillId, ...] = ()
    spans: tuple[Span, ...] = ()


class Project(_Strict):
    skill_ids: tuple[SkillId, ...] = ()
    evidence_level: EvidenceLevel
    spans: tuple[Span, ...] = ()


EducationLevel = Literal["secondary", "diploma", "bachelor", "master", "doctorate"]


class Education(_Strict):
    # Institution is deliberately absent: college may be identity (SPEC §12 Q9).
    highest_level: EducationLevel | None = None
    field_relevance: Score01 | None = None


class SecurityFlag(_Strict):
    flag_id: str = Field(min_length=1)
    attack_family: Literal["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    layer: Literal["L0", "L1", "L2", "L3", "L4", "L5"]
    risk: Literal["low", "medium", "high"]
    location: LocationRef
    quote: str = ""
    note: str = ""


class VerifiedProfile(_Strict):
    """Schema-validated, span-verified facts about one candidate. No identity fields."""

    candidate_id: BlindCandidateId
    claims: tuple[SkillClaim, ...] = ()
    roles: tuple[Role, ...] | None = None  # None = not extracted -> treated as missing
    projects: tuple[Project, ...] | None = None
    education: Education | None = None
    location_match: Score01 | None = None  # precomputed vs the job's location policy (inferred)
    security_flags: tuple[SecurityFlag, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _reject_identity_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            bad = sorted(FORBIDDEN_IDENTITY_FIELDS & {str(k).lower() for k in data})
            if bad:
                raise ValueError(f"identity/PII fields are not allowed in a VerifiedProfile: {bad}")
        return data


# --------------------------------------------------------------------------------------
# RankingResult (decision support output)
# --------------------------------------------------------------------------------------


class ComponentScore(_Strict):
    name: Literal["skill_match", "experience", "projects", "education", "tenure", "location"]
    present: bool
    score: Score01 | None
    nominal_weight: float
    effective_weight: float  # after renormalising over present components
    note: str = ""


class RequirementExplanation(_Strict):
    skill_id: SkillId
    label: str
    tag: RequirementTag
    declared_level: EvidenceLevel
    effective_level: EvidenceLevel
    credit: Score01
    via_adjacent_skill: SkillId | None = None
    quote: str | None = None
    location: LocationRef | None = None
    adjustment: str | None = None  # e.g. "L3: E3 capped to E1 (no cited visible span)"
    gap: str | None = None


class RankingResult(_Strict):
    candidate_id: str = Field(min_length=1)  # str, not BlindCandidateId: may echo a bad input
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    profile_id: str
    profile_version: int
    config_version: str
    catalogue_version: str

    fit_score: Score01 = 0.0  # final score, flagged spans excluded (0.0-1.0, SPEC §6.4)
    ci_low: Score01 = 0.0
    ci_high: Score01 = 0.0
    raw_score: Score01 = 0.0  # score if flags were ignored (L4 counterfactual)
    counterfactual_delta: float = 0.0  # raw_score - fit_score: inflation the attack attempted

    tier: Tier | None = None
    rank: int | None = Field(default=None, ge=1)
    percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    tie_band: int | None = None
    tie_with: tuple[str, ...] = ()  # candidates whose 95% CI overlaps this one

    mandatory_coverage: Score01 | None = None
    mandatory_gate_multiplier: Score01 = 1.0
    components: tuple[ComponentScore, ...] = ()
    explanations: tuple[RequirementExplanation, ...] = ()
    security_flags: tuple[SecurityFlag, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.security_flags)


class PoolRanking(_Strict):
    profile_id: str
    profile_version: int
    config_version: str
    catalogue_version: str
    results: tuple[RankingResult, ...]
    n_scored: int
    n_errors: int
    n_flagged: int
