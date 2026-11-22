"""End-to-end Job Workspace endpoints (docs/WEB_UI_FLOWS.md §§1-5).

Wires Job Analysis, Resume Intake (L0-L3), Hold Clearance, Pool Scoring, and Shortlist
Approval into a single stateful workspace backed by in-memory app state.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_talent_screener.api.dependencies import LedgerDep
from agentic_talent_screener.job_analysis import JobAnalysisError, draft_requirement_profile
from agentic_talent_screener.ranking.config import ScoringConfig
from agentic_talent_screener.ranking.matching import LLMMatcher, MatchingAgent
from agentic_talent_screener.ranking.sample_data import (
    sample_matcher,
    sample_pool,
    sample_requirement_profile,
)
from agentic_talent_screener.ranking.schemas import (
    PoolRanking,
    Priority,
    Requirement,
    RequirementProfile,
    VerifiedProfile,
)
from agentic_talent_screener.ranking.scorer import score_pool
from agentic_talent_screener.resume_intake.pipeline import PipelineResult, normalize_resume
from agentic_talent_screener.resume_intake.ranking_adapter import to_ranking_profile

router = APIRouter()

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ResumeUIStatus = Literal["processing", "ready_to_score", "review_required", "quarantined"]


@dataclass
class ResumeRecord:
    candidate_id: str
    filename: str
    status: ResumeUIStatus
    uploaded_at: datetime
    verified_profile: VerifiedProfile | None = None
    pipeline_status: str | None = None
    hold_reason: str | None = None
    review_history: list[dict[str, Any]] = field(default_factory=list)
    is_sample: bool = False


@dataclass
class JobWorkspace:
    job_id: str
    jd_text: str
    created_at: datetime
    draft_profile: RequirementProfile | None
    active_profile: RequirementProfile | None = None
    resumes: dict[str, ResumeRecord] = field(default_factory=dict)
    next_candidate_num: int = 1
    latest_ranking: PoolRanking | None = None
    last_scored_at: datetime | None = None
    shortlist_snapshots: list[dict[str, Any]] = field(default_factory=list)
    overrides: list[dict[str, Any]] = field(default_factory=list)


def _store(request: Request) -> dict[str, JobWorkspace]:
    if not hasattr(request.app.state, "jobs"):
        request.app.state.jobs = {}
    return request.app.state.jobs


def _get_job(request: Request, job_id: str) -> JobWorkspace:
    job = _store(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return job


def _serialize_resume(rec: ResumeRecord) -> dict[str, Any]:
    return {
        "candidate_id": rec.candidate_id,
        "filename": rec.filename,
        "status": rec.status,
        "uploaded_at": rec.uploaded_at.isoformat(),
        "pipeline_status": rec.pipeline_status,
        "hold_reason": rec.hold_reason,
        "has_verified_profile": rec.verified_profile is not None,
        "overall_risk": rec.verified_profile.overall_risk if rec.verified_profile else None,
        "flag_count": len(rec.verified_profile.security_flags) if rec.verified_profile else 0,
        "review_history": rec.review_history,
    }


def _serialize_job(job: JobWorkspace) -> dict[str, Any]:
    counts: dict[str, int] = {
        "processing": 0,
        "ready_to_score": 0,
        "review_required": 0,
        "quarantined": 0,
    }
    for r in job.resumes.values():
        counts[r.status] = counts.get(r.status, 0) + 1

    current_profile = job.active_profile or job.draft_profile
    return {
        "job_id": job.job_id,
        "role_title": current_profile.role_title if current_profile else "Untitled Role",
        "jd_text": job.jd_text,
        "created_at": job.created_at.isoformat(),
        "draft_profile": job.draft_profile.model_dump(mode="json") if job.draft_profile else None,
        "active_profile": job.active_profile.model_dump(mode="json") if job.active_profile else None,
        "resume_uploads_unlocked": job.active_profile is not None,
        "resume_counts": counts,
        "resumes": [_serialize_resume(r) for r in job.resumes.values()],
        "latest_ranking": job.latest_ranking.model_dump(mode="json") if job.latest_ranking else None,
        "last_scored_at": job.last_scored_at.isoformat() if job.last_scored_at else None,
        "shortlist_snapshots": job.shortlist_snapshots,
        "overrides": job.overrides,
    }


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jd_text: NonBlank
    use_sample: bool = False
    model: str | None = None


class RequirementPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: NonBlank
    status: Literal["active", "inactive"] | None = None
    priority: Priority | None = None
    weight: float | None = Field(default=None, gt=0.0, le=10.0)
    statement: NonBlank | None = None


class PatchRequirementsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: list[RequirementPatch] = Field(min_length=1)


class ApproveRequirementsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: NonBlank


class ReviewResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["clear_for_scoring", "keep_on_hold"]
    actor: NonBlank
    reason: NonBlank


class ScoreJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortlist_size: int | None = Field(default=None, ge=1)
    exclude_flagged: bool = True
    actor: NonBlank = "system"
    model: str | None = None


class ShortlistOverrideItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: NonBlank
    action: Literal["include", "exclude"]
    reason: NonBlank


class ApproveShortlistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: NonBlank
    shortlist_size: int = Field(ge=1)
    overrides: list[ShortlistOverrideItem] = Field(default_factory=list)


def _ingest_pipeline_result(
    job: JobWorkspace, filename: str, result: PipelineResult
) -> ResumeRecord:
    cid = f"C-{job.next_candidate_num:04d}"
    job.next_candidate_num += 1
    now = datetime.now(timezone.utc)

    if result.status == "quarantined":
        rec = ResumeRecord(
            candidate_id=cid,
            filename=filename,
            status="quarantined",
            uploaded_at=now,
            pipeline_status=result.status,
            hold_reason=f"Quarantined at {result.stopped_at}",
        )
        job.resumes[cid] = rec
        return rec

    if result.profile is None or result.security_assessment is None:
        rec = ResumeRecord(
            candidate_id=cid,
            filename=filename,
            status="review_required",
            uploaded_at=now,
            pipeline_status=result.status,
            hold_reason=f"No usable Verified Profile produced (stopped at {result.stopped_at})",
        )
        job.resumes[cid] = rec
        return rec

    reblinded = result.profile.model_copy(update={"blind_candidate_id": cid})
    handoff = to_ranking_profile(reblinded, result.security_assessment)
    needs_review = result.requires_human_review or result.status in {
        "manual_review",
        "normalized_with_flag",
    }
    rec = ResumeRecord(
        candidate_id=cid,
        filename=filename,
        status="review_required" if needs_review else "ready_to_score",
        uploaded_at=now,
        verified_profile=handoff.profile,
        pipeline_status=result.status,
        hold_reason="Flagged during intake; HR clearance required" if needs_review else None,
    )
    job.resumes[cid] = rec
    return rec


@router.get("")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return [_serialize_job(job) for job in _store(request).values()]


@router.post("")
def create_job(req: CreateJobRequest, request: Request) -> dict[str, Any]:
    job_id = f"job-{uuid4().hex[:8]}"
    if req.use_sample:
        base = sample_requirement_profile()
        draft = base.model_copy(
            update={"profile_id": f"{job_id}-v1", "version": 1, "approved": False, "approved_by": None, "approved_at": None}
        )
    else:
        try:
            draft = draft_requirement_profile(req.jd_text, profile_id=f"{job_id}-v1", version=1, model=req.model)
        except JobAnalysisError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Job analysis failed: {exc}") from exc

    job = JobWorkspace(
        job_id=job_id,
        jd_text=req.jd_text,
        created_at=datetime.now(timezone.utc),
        draft_profile=draft,
    )
    _store(request)[job_id] = job
    return _serialize_job(job)


@router.get("/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    return _serialize_job(_get_job(request, job_id))


@router.patch("/{job_id}/requirements")
def patch_requirements(job_id: str, req: PatchRequirementsRequest, request: Request) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.draft_profile is None:
        raise HTTPException(status_code=409, detail="No active draft; create a revision first.")

    by_id = {u.requirement_id: u for u in req.updates}
    updated: list[Requirement] = []
    seen: set[str] = set()
    for item in job.draft_profile.requirements:
        patch = by_id.get(item.requirement_id)
        if patch is None:
            updated.append(item)
            continue
        seen.add(item.requirement_id)
        changes = patch.model_dump(exclude={"requirement_id"}, exclude_none=True)
        updated.append(item.model_copy(update=changes))

    missing = set(by_id) - seen
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown requirement_id(s): {sorted(missing)}")

    job.draft_profile = job.draft_profile.model_copy(update={"requirements": tuple(updated)})
    return _serialize_job(job)


@router.post("/{job_id}/requirements/approve")
def approve_requirements(job_id: str, req: ApproveRequirementsRequest, request: Request) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.draft_profile is None:
        raise HTTPException(status_code=409, detail="No draft profile to approve.")
    if not job.draft_profile.active_requirements:
        raise HTTPException(status_code=400, detail="At least one requirement must be active before approval.")

    approved = job.draft_profile.model_copy(
        update={
            "approved": True,
            "approved_by": req.actor,
            "approved_at": datetime.now(timezone.utc),
        }
    )
    job.active_profile = approved
    job.draft_profile = None
    return _serialize_job(job)


@router.post("/{job_id}/requirements/revision")
def create_revision(job_id: str, request: Request) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.draft_profile is not None:
        return _serialize_job(job)
    if job.active_profile is None:
        raise HTTPException(status_code=409, detail="No approved profile to revise.")

    next_v = job.active_profile.version + 1
    job.draft_profile = job.active_profile.model_copy(
        update={
            "profile_id": f"{job.job_id}-v{next_v}",
            "version": next_v,
            "approved": False,
            "approved_by": None,
            "approved_at": None,
        }
    )
    return _serialize_job(job)


@router.post("/{job_id}/resumes")
async def upload_resumes(
    job_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.active_profile is None:
        raise HTTPException(status_code=409, detail="Approve requirements before uploading resumes.")

    added: list[dict[str, Any]] = []
    for upload in files:
        suffix = Path(upload.filename or "resume.pdf").suffix or ".pdf"
        content = await upload.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            result = normalize_resume(tmp_path, show_progress=False)
        finally:
            tmp_path.unlink(missing_ok=True)
        rec = _ingest_pipeline_result(job, upload.filename or tmp_path.name, result)
        added.append(_serialize_resume(rec))

    return {"added": added, "job": _serialize_job(job)}


@router.post("/{job_id}/resumes/load-sample")
def load_sample_resumes(job_id: str, request: Request) -> dict[str, Any]:
    """Load the synthetic sample pool into this job for offline UI testing."""
    job = _get_job(request, job_id)
    if job.active_profile is None:
        raise HTTPException(status_code=409, detail="Approve requirements before loading resumes.")

    now = datetime.now(timezone.utc)
    for prof in sample_pool():
        flagged = bool(prof.security_flags)
        rec = ResumeRecord(
            candidate_id=prof.candidate_id,
            filename=f"{prof.candidate_id}_sample.pdf",
            status="review_required" if flagged else "ready_to_score",
            uploaded_at=now,
            verified_profile=prof,
            pipeline_status="normalized_with_flag" if flagged else "normalized",
            hold_reason="Security flag detected in sample resume; HR clearance required" if flagged else None,
            is_sample=True,
        )
        job.resumes[prof.candidate_id] = rec

    return _serialize_job(job)


@router.post("/{job_id}/resumes/{candidate_id}/review")
def review_resume(
    job_id: str,
    candidate_id: str,
    req: ReviewResumeRequest,
    request: Request,
    ledger: LedgerDep,
) -> dict[str, Any]:
    job = _get_job(request, job_id)
    rec = job.resumes.get(candidate_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown candidate_id: {candidate_id}")
    if rec.status == "quarantined":
        raise HTTPException(status_code=409, detail="Quarantined files cannot be cleared for scoring.")

    if req.decision == "clear_for_scoring":
        if rec.verified_profile is None:
            raise HTTPException(status_code=409, detail="Cannot clear a resume without a Verified Profile.")
        rec.status = "ready_to_score"
        rec.hold_reason = None
    else:
        rec.status = "review_required"
        rec.hold_reason = req.reason

    entry = {
        "decision": req.decision,
        "actor": req.actor,
        "reason": req.reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    rec.review_history.append(entry)
    ledger.record_hr_override(candidate_id, req.actor, req.decision, req.reason)
    return {"resume": _serialize_resume(rec), "job": _serialize_job(job)}


@router.post("/{job_id}/score")
def score_job(
    job_id: str,
    req: ScoreJobRequest,
    request: Request,
    ledger: LedgerDep,
) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.active_profile is None:
        raise HTTPException(status_code=409, detail="No approved requirement profile to score against.")

    ready = [r for r in job.resumes.values() if r.status == "ready_to_score" and r.verified_profile is not None]
    if not ready:
        raise HTTPException(status_code=409, detail="No resumes are ready to score.")

    unranked = [
        {"candidate_id": r.candidate_id, "status": r.status, "reason": r.hold_reason or r.status}
        for r in job.resumes.values()
        if r.status != "ready_to_score"
    ]

    matcher: MatchingAgent = sample_matcher() if all(r.is_sample for r in ready) else LLMMatcher(model=req.model)
    cfg = ScoringConfig(exclude_flag_risks=("medium", "high") if req.exclude_flagged else ())
    ranking = score_pool(
        [r.verified_profile for r in ready if r.verified_profile is not None],
        job.active_profile,
        matcher,
        cfg,
        shortlist_size=req.shortlist_size,
    )
    ledger.record_pool_scoring(
        ranking,
        actor=req.actor,
        parameters={"job_id": job_id, "exclude_flagged": req.exclude_flagged, "shortlist_size": req.shortlist_size},
    )
    job.latest_ranking = ranking
    job.last_scored_at = datetime.now(timezone.utc)
    return {
        "ranking": ranking.model_dump(mode="json"),
        "n_ready": len(ready),
        "unranked": unranked,
        "job": _serialize_job(job),
    }


@router.post("/{job_id}/shortlist/approve")
def approve_shortlist(
    job_id: str,
    req: ApproveShortlistRequest,
    request: Request,
    ledger: LedgerDep,
) -> dict[str, Any]:
    job = _get_job(request, job_id)
    if job.latest_ranking is None or job.active_profile is None:
        raise HTTPException(status_code=409, detail="Score the pool before approving a shortlist.")

    ranked_ids = [r.candidate_id for r in job.latest_ranking.results if r.status == "ok"]
    shortlist = list(ranked_ids[: req.shortlist_size])

    for ov in req.overrides:
        ledger.record_hr_override(ov.candidate_id, req.actor, f"shortlist_{ov.action}", ov.reason)
        if ov.action == "include" and ov.candidate_id not in shortlist:
            shortlist.append(ov.candidate_id)
        elif ov.action == "exclude" and ov.candidate_id in shortlist:
            shortlist.remove(ov.candidate_id)
        job.overrides.append(
            {
                "candidate_id": ov.candidate_id,
                "action": ov.action,
                "reason": ov.reason,
                "actor": req.actor,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    snapshot = {
        "snapshot_id": f"sl-{len(job.shortlist_snapshots) + 1}",
        "profile_version": job.active_profile.version,
        "shortlist_size": req.shortlist_size,
        "approved_by": req.actor,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "candidates": shortlist,
        "overrides": [o.model_dump(mode="json") for o in req.overrides],
    }
    job.shortlist_snapshots.append(snapshot)
    return {"snapshot": snapshot, "job": _serialize_job(job)}
