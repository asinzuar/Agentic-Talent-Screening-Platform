"""Ranking, benchmark, HR override and audit endpoints for the HR What-If workbench.

Runs on the synthetic sample pool only. The audit ledger lives on `app.state` in memory;
persistence is an open question (SPEC §12 Q12).
"""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_talent_screener.api.dependencies import LedgerDep
from agentic_talent_screener.ranking.audit import AuditEvent
from agentic_talent_screener.ranking.config import ScoringConfig
from agentic_talent_screener.ranking.eval_harness import run_benchmark, sample_ground_truth
from agentic_talent_screener.ranking.sample_data import (
    sample_matcher,
    sample_pool,
    sample_requirement_profile,
)
from agentic_talent_screener.ranking.schemas import PoolRanking
from agentic_talent_screener.ranking.scorer import score_pool

router = APIRouter(tags=["ranking"])

PROFILE = sample_requirement_profile()
MATCHER = sample_matcher()
KNOWN_IDS = frozenset(c.candidate_id for c in sample_pool())

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CandidateId = Annotated[str, StringConstraints(pattern=r"^C-\d{4}$")]


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_flagged: bool = True
    shortlist_size: int | None = Field(default=None, ge=1)
    actor: NonBlank = "system"


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ground_truth: dict[CandidateId, Annotated[float, Field(ge=0.0, le=5.0)]] | None = None
    actor: NonBlank = "system"


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: CandidateId
    actor: NonBlank
    action: NonBlank
    reason: NonBlank = Field(description="Required, non-blank (SPEC §10)")


@router.get("/sample")
def sample() -> dict[str, Any]:
    cfg = ScoringConfig()
    ranking = score_pool(sample_pool(), PROFILE, MATCHER, cfg)
    return {
        "config": cfg.model_dump(mode="json"),
        "profile": PROFILE.model_dump(mode="json"),
        "ranking": ranking.model_dump(mode="json"),
    }


@router.post("/rank", response_model=PoolRanking)
def rank(req: RankRequest, ledger: LedgerDep) -> PoolRanking:
    cfg = ScoringConfig(exclude_flag_risks=("medium", "high") if req.exclude_flagged else ())
    try:
        result = score_pool(sample_pool(), PROFILE, MATCHER, cfg, shortlist_size=req.shortlist_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ledger.record_pool_scoring(
        result,
        actor=req.actor,
        parameters={"exclude_flagged": req.exclude_flagged, "shortlist_size": req.shortlist_size},
    )
    return result


@router.get("/benchmark")
def benchmark_default() -> dict[str, Any]:
    """Read-only default benchmark (not logged), so read-only users can view results."""
    return run_benchmark(sample_pool(), PROFILE, MATCHER, ScoringConfig(), sample_ground_truth())


@router.post("/benchmark")
def benchmark(req: BenchmarkRequest, ledger: LedgerDep) -> dict[str, Any]:
    cfg = ScoringConfig()
    truth = req.ground_truth if req.ground_truth is not None else sample_ground_truth()
    result = run_benchmark(sample_pool(), PROFILE, MATCHER, cfg, truth)
    ledger.record_benchmark(
        result,
        actor=req.actor,
        parameters={"custom_ground_truth": req.ground_truth is not None},
    )
    return result


@router.post("/override")
def override(req: OverrideRequest, ledger: LedgerDep) -> dict[str, Any]:
    if req.candidate_id not in KNOWN_IDS:
        raise HTTPException(status_code=404, detail="unknown candidate_id")
    try:
        event: AuditEvent = ledger.record_hr_override(
            req.candidate_id, req.actor, req.action, req.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return event.model_dump(mode="json")


@router.get("/audit")
def audit(ledger: LedgerDep) -> dict[str, Any]:
    return {
        "chain_valid": ledger.verify_chain(),
        "events": [e.model_dump(mode="json") for e in ledger.events],
    }


@router.get("/audit/export")
def audit_export(ledger: LedgerDep) -> Response:
    return Response(
        content=ledger.to_jsonl() + "\n",
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="audit_ledger.jsonl"'},
    )
