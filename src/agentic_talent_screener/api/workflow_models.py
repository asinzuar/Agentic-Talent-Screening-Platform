from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_talent_screener.ranking.schemas import BlindCandidateId

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Positive = Annotated[int, Field(ge=1)]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Revision(Body):
    expected_revision: NonBlank


class CreateRevision(Body):
    base_version: Positive


class FlagReview(Body):
    flag_id: NonBlank
    disposition: Literal["addressed", "acknowledged"]


class SaveDraft(Revision):
    role_title: NonBlank
    requirements: list[dict] = Field(max_length=200)
    not_used_for_scoring: list[NonBlank] = Field(default_factory=list, max_length=200)
    activate_requirement_ids: list[NonBlank] = Field(default_factory=list, max_length=200)
    flag_reviews: list[FlagReview] = Field(default_factory=list, max_length=1000)


class AttemptAction(Revision):
    attempt_id: NonBlank


class Review(AttemptAction):
    action: Literal["clear_for_scoring", "keep_on_hold"]
    reason: NonBlank


class ScoreRun(Body):
    plan_token: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Override(Body):
    candidate_id: BlindCandidateId
    action: Literal["include", "exclude"]
    reason: NonBlank


class OverrideDecision(Body):
    override_id: NonBlank
    decision: Literal["reconfirm", "drop"]


class ProposalEdit(Revision):
    size: Positive
    overrides: list[Override] = Field(default_factory=list, max_length=9999)
    prior_override_decisions: list[OverrideDecision] = Field(default_factory=list, max_length=9999)
