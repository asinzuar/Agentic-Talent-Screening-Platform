"""Strict Pydantic schemas for the score-based ranking module (SPEC.md §6.2, §6.3, §13).

Design constraints enforced here:

* Blind candidate IDs only (``C-0932``); no identity / PII fields (``extra="forbid"`` plus an
  explicit deny-list so the error message is clear).
* No skill catalogue: requirements and candidate evidence are free text. The Semantic Matching
  Agent links them (``CandidateMatches``); the scorer never compares strings itself.
* Requirements carry the full SPEC §6.2 field set and live in a versioned profile that must be
  human-approved before anything is scored against it.
* Evidence levels are E0-E4 and every piece of evidence cites spans (location + quote).
* A candidate is a flat list of free-text facts, as the intake lane emits them (SPEC §13.2,
  v0.4): there are no structured roles, projects or education records.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal

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

Ident = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")]
"""Short identifier for requirements, options and evidence items, e.g. ``M1``, ``M1.a``, ``r2``."""

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

Score01 = Annotated[float, Field(ge=0.0, le=1.0)]

LocationRef = Annotated[str, StringConstraints(pattern=r"^p\d+:s\d{4}$")]
"""Span ID in the resume text, e.g. ``p1:s0014`` (SPEC §13.2, RESUME_INTAKE_SPEC §3.2)."""

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


class Priority(StrEnum):
    MANDATORY = "M"
    PREFERRED = "P"


RequirementKind = Literal["skill", "experience", "education", "leadership", "achievement"]
FactKind = Literal["skill", "experience", "project", "education", "certification"]
ALL_FACT_KINDS: tuple[FactKind, ...] = ("skill", "experience", "project", "education", "certification")
RiskLevel = Literal["low", "medium", "high"]
Provenance = Literal["jd_qualification", "jd_responsibility", "model_suggestion"]
MatchRelation = Literal["equivalent", "adjacent"]


# --------------------------------------------------------------------------------------
# RequirementProfile (trusted lane, SPEC §6.2)
# --------------------------------------------------------------------------------------


class Criterion(_Strict):
    """One option or substitute of a requirement, in free text (e.g. "Python")."""

    option_id: Ident
    text: Text
    preferred: bool = False  # shown to the reviewer; no effect on the score


class Requirement(_Strict):
    requirement_id: Ident
    statement: Text
    kind: RequirementKind
    options: tuple[Criterion, ...] = Field(min_length=1)
    combine: Literal["all_of", "any_of"] = "all_of"
    min_years: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)  # experience requirements only
    min_evidence: EvidenceLevel | None = None  # None = any usable evidence counts as met
    substitutes: tuple[Criterion, ...] = ()
    priority: Priority
    weight: float = Field(gt=0.0, le=10.0)
    provenance: Provenance | Literal["hr_added"]
    quote: str | None = None  # required for JD provenance
    rationale: str | None = None  # required for model suggestions
    onet_source: str | None = None
    status: Literal["active", "inactive"]
    flags: tuple[Text, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _default_status(cls, data: Any) -> Any:
        # SPEC §6.2: only JD qualifications start active; responsibilities and model
        # suggestions stay inactive until a reviewer accepts them.
        if isinstance(data, dict) and data.get("status") is None:
            active = data.get("provenance") == "jd_qualification"
            data = {**data, "status": "active" if active else "inactive"}
        return data

    @model_validator(mode="after")
    def _check(self) -> Requirement:
        if self.provenance in {"jd_qualification", "jd_responsibility"} and not (self.quote and self.quote.strip()):
            raise ValueError("requirements from the JD must carry the quoted JD text")
        if self.provenance == "model_suggestion" and not (self.rationale and self.rationale.strip()):
            raise ValueError("model-suggested requirements must carry the agent's rationale")
        if self.provenance == "hr_added" and not (self.rationale and self.rationale.strip()):
            raise ValueError("HR-added requirements must carry the reviewer's rationale")
        if self.min_years is not None and self.kind != "experience":
            raise ValueError("min_years only applies to experience requirements")
        if sum(o.preferred for o in self.options) > 1:
            raise ValueError("at most one option can be marked preferred")
        if any(o.preferred for o in self.options) and self.combine != "any_of":
            raise ValueError("a preferred option only makes sense in an any_of group")
        if any(s.preferred for s in self.substitutes):
            raise ValueError("substitutes cannot be marked preferred")
        return self

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return self.options + self.substitutes


class RequirementProfile(_Strict):
    profile_id: Text
    version: int = Field(ge=1)
    role_title: Text
    requirements: tuple[Requirement, ...] = Field(min_length=1)
    not_used_for_scoring: tuple[Text, ...] = ()  # JD content shown to the reviewer, not scored
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> RequirementProfile:
        if self.approved and not (self.approved_by and self.approved_at):
            raise ValueError("an approved profile must record approved_by and approved_at")
        ids = [r.requirement_id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("each requirement_id may appear only once in a profile")
        opts = [c.option_id for r in self.requirements for c in r.criteria]
        if len(opts) != len(set(opts)):
            raise ValueError("each option_id may appear only once in a profile")
        return self

    @property
    def active_requirements(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.active)


# --------------------------------------------------------------------------------------
# VerifiedProfile (output of the quarantined lane; the only candidate data scoring sees)
# --------------------------------------------------------------------------------------


class Span(_Strict):
    """A cited span in the resume text (SPEC §13.2: ``p1:s0014``).

    ``quote`` is the bounded, PII-screened text of the span (never the claim itself). It is
    part of the fact: used for evidence cards and review, and shown to the Matching Agent.
    """

    location: LocationRef
    quote: str
    visible: bool = True  # False = hidden / off-page / behind-image text per L0


class Fact(_Strict):
    """One verified, free-text fact. Titles, levels and fields are left to the Matching Agent."""

    fact_id: Ident
    kind: FactKind
    statement: Text  # free text, as written on the resume
    evidence_level: EvidenceLevel  # E1-E4 from L3; E0 only arises from the scorer's grounding
    dates: tuple[str, ...] = ()  # date strings as written (e.g. "2023-25"); no duration (Q18)
    spans: tuple[Span, ...] = ()


class SecurityFlag(_Strict):
    """One intake finding (L0 or L1) that cites at least one span.

    There is deliberately no ``quote``: flagged text is raw, suspicious candidate text, so it
    never crosses into ranking. The HR view fetches the excerpt by span ID from the audit lane.
    """

    flag_id: str = Field(min_length=1)
    attack_family: Literal["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    layer: Literal["L0", "L1", "L2", "L3", "L4", "L5"]
    risk: RiskLevel  # this finding's own severity, not the document's
    locations: tuple[LocationRef, ...] = Field(min_length=1)
    note: str = ""


class VerifiedProfile(_Strict):
    """Schema-validated, span-verified facts about one candidate. No identity fields.

    ``extracted_kinds`` lists the fact kinds the intake lane attempted to extract: a kind that
    is absent was *not extracted* (missing); a listed kind with no facts means none were found.
    ``intake_status`` and ``overall_risk`` come from intake and are echoed to the reviewer
    (they do not change the score); they are ``None`` for profiles not produced by intake.
    """

    candidate_id: BlindCandidateId
    facts: tuple[Fact, ...] = ()
    extracted_kinds: tuple[FactKind, ...] = ALL_FACT_KINDS
    intake_status: Literal["verified", "verified_with_flag"] | None = None
    overall_risk: RiskLevel | None = None
    security_flags: tuple[SecurityFlag, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _reject_identity_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            bad = sorted(FORBIDDEN_IDENTITY_FIELDS & {str(k).lower() for k in data})
            if bad:
                raise ValueError(f"identity/PII fields are not allowed in a VerifiedProfile: {bad}")
        return data

    @model_validator(mode="after")
    def _unique_fact_ids(self) -> VerifiedProfile:
        ids = [f.fact_id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact_id must be unique within a profile")
        return self

    def matcher_view(self) -> MatcherView:
        """The least a Matching Agent needs: no quotes, span IDs, flags, status or risk."""
        return MatcherView(
            candidate_id=self.candidate_id,
            facts=tuple(
                MatcherFact(
                    fact_id=f.fact_id, kind=f.kind, statement=f.statement,
                    evidence_level=f.evidence_level, dates=f.dates,
                )
                for f in self.facts
            ),
        )


# --------------------------------------------------------------------------------------
# What the Matching Agent sees (SPEC §13.2, "Quote and matcher-input rules")
# --------------------------------------------------------------------------------------


class MatcherFact(_Strict):
    """A fact as the Matching Agent sees it: statements only, never the resume text around them."""

    fact_id: Ident
    kind: FactKind
    statement: Text
    evidence_level: EvidenceLevel
    dates: tuple[str, ...] = ()


class MatcherView(_Strict):
    """One candidate for matching. Quotes stay out: they can carry college and employer names."""

    candidate_id: BlindCandidateId
    facts: tuple[MatcherFact, ...] = ()


# --------------------------------------------------------------------------------------
# Semantic Matching Agent output (SPEC §6.3, §13.3)
# --------------------------------------------------------------------------------------


class OptionMatch(_Strict):
    """The matcher's claim that one fact supports one requirement option or substitute.

    The matcher only links; it cannot set evidence levels or quote text. Both come from the
    linked fact in the VerifiedProfile.
    """

    option_id: Ident
    fact_id: Ident
    relation: MatchRelation
    similarity: Score01 = 1.0  # used to scale credit for adjacent matches
    rationale: str = ""


class CandidateMatches(_Strict):
    candidate_id: BlindCandidateId
    profile_id: Text
    profile_version: int = Field(ge=1)
    matcher_version: Text
    matches: tuple[OptionMatch, ...] = ()


# --------------------------------------------------------------------------------------
# RankingResult (decision support output)
# --------------------------------------------------------------------------------------


class CriterionResult(_Strict):
    option_id: str
    text: str
    is_substitute: bool
    credit: Score01
    met: bool
    evidence_fact: str | None = None
    relation: MatchRelation | None = None
    similarity: Score01 | None = None
    declared_level: EvidenceLevel | None = None
    effective_level: EvidenceLevel | None = None  # None for education facts (no ladder)
    quote: str | None = None
    location: LocationRef | None = None
    adjustment: str | None = None  # e.g. "L3: E3 capped to E1 (no cited visible span)"


class RequirementExplanation(_Strict):
    requirement_id: str
    statement: str
    kind: RequirementKind
    priority: Priority
    weight: float
    effective_weight: float  # share of the score after the M multiplier and renormalisation
    missing: bool  # evidence source not extracted: excluded and weight renormalised
    met: bool
    credit: Score01
    via_substitute: bool = False
    # The criterion that decided the credit (weakest option for all_of, best for any_of).
    deciding: CriterionResult | None = None
    criteria: tuple[CriterionResult, ...] = ()
    gap: str | None = None


class RankingResult(_Strict):
    candidate_id: str = Field(min_length=1)  # str, not BlindCandidateId: may echo a bad input
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    profile_id: str
    profile_version: int
    config_version: str
    matcher_version: str

    fit_score: Score01 = 0.0  # final score, flagged spans excluded (0.0-1.0, SPEC §6.3)
    raw_score: Score01 = 0.0  # score if flags were ignored (L4 counterfactual)
    counterfactual_delta: float = 0.0  # raw_score - fit_score: inflation the attack attempted

    tier: Tier | None = None
    rank: int | None = Field(default=None, ge=1)
    shortlisted: bool | None = None  # None when no shortlist size was requested
    near_cutoff: bool = False  # among the first few candidates just below the shortlist

    mandatory_coverage: Score01 | None = None
    mandatory_gate_multiplier: Score01 = 1.0
    unassessed: tuple[str, ...] = ()  # requirement IDs whose evidence source wasn't extracted
    explanations: tuple[RequirementExplanation, ...] = ()
    security_flags: tuple[SecurityFlag, ...] = ()
    intake_status: Literal["verified", "verified_with_flag"] | None = None  # echoed from intake
    overall_risk: RiskLevel | None = None  # echoed from intake; does not change the score
    notes: tuple[str, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.security_flags)


class PoolRanking(_Strict):
    profile_id: str
    profile_version: int
    config_version: str
    matcher_version: str
    shortlist_size: int | None = None
    results: tuple[RankingResult, ...]
    n_scored: int
    n_errors: int
    n_flagged: int
