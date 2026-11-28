import hashlib
import hmac
import json
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from sqlite3 import Connection
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentic_talent_screener.api.workflow_models import (
    AttemptAction, Body, CreateRevision, NonBlank, Positive, ProposalEdit, Review, Revision, SaveDraft, ScoreRun,
)
from agentic_talent_screener.resume_intake.l0_forensics import MAX_FILE_BYTES
from agentic_talent_screener.workflow import intake, requirements, scoring, shortlists
from agentic_talent_screener.workflow.storage import WorkflowError, audit, digest, encode, events_for, require, utc_now

MAX_BATCH_BYTES = 32 * 1024 * 1024
MAX_BATCH_FILES = 32
MAX_REQUEST_BYTES = MAX_BATCH_BYTES + 512 * 1024  # Includes multipart framing.
JD_TYPES = {".txt": "text/plain", ".pdf": "application/pdf"}
CandidateStatus = Literal["processing", "ready_to_score", "review_required", "quarantined"]


class BoundedRequestRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def bounded(request: Request):
            if request.method not in {"POST", "PUT"}:
                return await handle(request)
            received = 0

            async def receive():
                nonlocal received
                message = await request.receive()
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    # FastAPI passes HTTPException through during request-body parsing.
                    raise HTTPException(413)
                return message

            size = request.headers.get("Content-Length")
            try:
                if size is not None and (not size.isdecimal() or len(size) > 20):
                    raise WorkflowError(422, "INVALID_CONTENT_LENGTH", "Provide a valid Content-Length.")
                if size is not None and int(size) > MAX_REQUEST_BYTES:
                    raise HTTPException(413)
                return await handle(Request(request.scope, receive=receive))
            except HTTPException as error:
                if error.status_code == 413:
                    raise WorkflowError(413, "UPLOAD_TOO_LARGE", "The request exceeds the allowed size.") from error
                if error.status_code == 400:
                    raise WorkflowError(400, "MALFORMED_REQUEST", "The request body could not be parsed.") from error
                raise

        return bounded


router = APIRouter(prefix="/jobs", tags=["job workflow"], route_class=BoundedRequestRoute)
bearer = HTTPBearer(auto_error=False)


def identify(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> str:
    if credentials and credentials.credentials.isascii():
        for actor, token in request.app.state.settings.workflow_tokens.items():
            if hmac.compare_digest(credentials.credentials, token.get_secret_value()):
                return actor
    raise WorkflowError(401, "AUTHENTICATION_REQUIRED", "Provide valid workflow bearer credentials.")


Actor = Annotated[str, Depends(identify)]


def active_actor(db: Connection, actor: str) -> None:
    row = db.execute("SELECT disabled_at FROM actors WHERE actor_id = ?", (actor,)).fetchone()
    if row and row["disabled_at"]:
        raise WorkflowError(403, "ACTOR_DISABLED", "This account is disabled.")


def job_grants(db: Connection, job_id: str, actor: str) -> set[str]:
    return {row[0] for row in db.execute(
        "SELECT permission FROM job_permissions WHERE job_id = ? AND actor_id = ?", (job_id, actor),
    )}


def authorize(db: Connection, job_id: str | None, actor: str, permission: str) -> set[str]:
    active_actor(db, actor)
    if job_id is None:
        return set()
    grants = job_grants(db, job_id, actor)
    if not grants:
        raise WorkflowError(404, "NOT_FOUND", "Resource not found.")
    if not allowed(grants, permission):
        raise WorkflowError(403, "PERMISSION_REQUIRED", "You do not have permission for this action.")
    return grants


def allowed(grants: set[str], permission: str) -> bool:
    return "owner" in grants or permission in grants


@contextmanager
def access(request, actor, permission="read", *, write=False):
    with request.app.state.workflow_store.transaction(write=write) as db:
        grants = authorize(db, request.path_params.get("job_id"), actor, permission)
        yield db, grants


def job_path(request, job_id: str) -> str:
    return f"{request.app.state.settings.api_v1_prefix}/jobs/{job_id}"


def response(data: dict, status: int = 200, location: str | None = None) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if location:
        headers["Location"] = location
    return JSONResponse(data, status_code=status, headers=headers)


def write(
    request: Request, actor: str, permission: str, payload: dict, handle: Callable,
    *, status: int, location: Callable[[dict], str] | None = None,
) -> JSONResponse:
    path = request.url.path
    key = request.headers.get("Idempotency-Key")
    is_post = request.method == "POST"
    if is_post and (key is None or not key.strip() or len(key) > 128):
        raise WorkflowError(422, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of up to 128 characters.")
    with access(request, actor, permission, write=True) as (db, _):
        db.execute(
            "INSERT OR IGNORE INTO actors (actor_id, issuer, subject) VALUES (?, 'workflow_bearer', ?)", (actor, actor),
        )
        if is_post:
            request_hash = digest(payload)
            replay = db.execute(
                """SELECT * FROM idempotency_keys WHERE (actor_id, method, resource_path, idempotency_key)
                = (?, 'POST', ?, ?)""", (actor, path, key),
            ).fetchone()
            if replay:
                authorize(db, replay["job_id"], actor, permission)
                if replay["request_hash"] != request_hash:
                    raise WorkflowError(409, "IDEMPOTENCY_KEY_CONFLICT", "Use a new key for different input.")
                return response(json.loads(replay["response_json"]), replay["response_status"], replay["location"])
        result = handle(db)
        target = location(result) if location else None
        if is_post:
            db.execute(
                "INSERT INTO idempotency_keys VALUES (?, 'POST', ?, ?, ?, ?, ?, ?, ?, ?)",
                (actor, path, key, request.path_params.get("job_id") or result.get("job_id"), request_hash,
                 status, target, encode(result), utc_now()),
            )
        return response(result, status, target)


class Page:
    def __init__(
        self, limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: Annotated[str | None, Query(pattern=r"^[0-9]{1,9}$")] = None,
    ):
        self.limit = limit
        self.offset = int(cursor or "0")

    def take(self, values: Iterable) -> tuple[list, str | None]:
        items = list(islice(values, self.offset, self.offset + self.limit + 1))
        following = str(self.offset + self.limit) if len(items) > self.limit else None
        return items[:self.limit], following


Pagination = Annotated[Page, Depends(Page)]


def page_response(page: Page, collection: Iterable, view: Callable = dict) -> JSONResponse:
    items, cursor = page.take(collection)
    return response({"items": [view(item) for item in items], "next_cursor": cursor})


def read_file(file: UploadFile, remaining: int, *, jd: bool = False) -> tuple[bytes, str]:
    bound = min(MAX_FILE_BYTES, remaining)
    if file.size is not None and file.size > bound:
        raise WorkflowError(413, "UPLOAD_TOO_LARGE", "The upload exceeds the allowed size.")
    data = file.file.read(bound + 1)
    if len(data) > bound:
        raise WorkflowError(413, "UPLOAD_TOO_LARGE", "The upload exceeds the allowed size.")
    extension = Path(file.filename or "").suffix.lower()
    media = (JD_TYPES if jd else intake.RESUME_TYPES).get(extension)
    if jd and not media:
        raise WorkflowError(415, "UNSUPPORTED_JD_TYPE", "Upload a TXT or text-based PDF job description.")
    return data, media or "application/octet-stream"


def uploaded(item: tuple[bytes, str]) -> dict:
    return {"sha256": hashlib.sha256(item[0]).hexdigest(), "media_type": item[1]}


def workspace(db, request, job_id, grants) -> dict:
    job = requirements.get_job(db, job_id)
    version = job["active_profile_version"]
    draft = requirements.draft_view(db, job_id)
    title = requirements.approved_view(db, job_id, version)["role_title"] if version else draft.get("role_title")
    latest = scoring.latest_ranking(db, job_id)
    current_id = scoring.current_ranking(db, job)
    proposal = shortlists.current_proposal(db, job_id, current_id) if current_id else None
    run = db.execute("SELECT run_id FROM scoring_runs WHERE job_id = ? ORDER BY created_at DESC LIMIT 1",
                     (job_id,)).fetchone()
    return {
        "job_id": job_id, "role_title": title,
        "draft": {"status": draft["status"], "href": f"{job_path(request, job_id)}/requirement-draft"},
        "active_profile": {"profile_id": job["profile_id"], "version": version} if version else None,
        "ranking_state": "current" if current_id else "rescore_required" if latest else "none",
        "current_ranking_id": current_id, "latest_ranking_id": latest["ranking_id"] if latest else None,
        "latest_scoring_run_id": run["run_id"] if run else None,
        "current_proposal_id": proposal["proposal_id"] if proposal else None,
        "last_approved_shortlist_id": job["last_approved_shortlist_id"],
        "actions": {
            "upload_resumes": intake.action(version is not None and allowed(grants, "upload"),
                                            "REQUIREMENTS_NOT_APPROVED" if version is None else "PERMISSION_REQUIRED"),
            "approve_requirements": intake.action(draft["status"] == "ready" and allowed(grants, "requirements_approve")),
            "create_revision": intake.action(version is not None and draft["status"] == "none"
                                             and allowed(grants, "requirements_edit")),
        },
    }


@router.get("")
def list_jobs(request: Request, actor: Actor, page: Pagination):
    with access(request, actor) as (db, _):
        jobs = db.execute(
            """SELECT j.job_id FROM jobs j WHERE EXISTS (SELECT 1 FROM job_permissions p
            WHERE (p.job_id, p.actor_id) = (j.job_id, ?) AND p.permission IN ('owner','read'))
            ORDER BY created_at DESC, job_id""", (actor,),
        )
        return page_response(page, jobs, lambda row: workspace(
            db, request, row["job_id"], job_grants(db, row["job_id"], actor),
        ))


@router.post("", status_code=202)
def create_job(request: Request, actor: Actor, file: Annotated[UploadFile, File()]):
    item = read_file(file, MAX_FILE_BYTES, jd=True)

    def create(db):
        job_id = requirements.create_job(db, actor, *item)
        return workspace(db, request, job_id, {"owner"})

    return write(request, actor, "read", uploaded(item), create, status=202,
                 location=lambda result: result["draft"]["href"])


@router.get("/{job_id}")
def get_job(request: Request, actor: Actor, job_id: str):
    with access(request, actor) as (db, grants):
        return response(workspace(db, request, job_id, grants))


@router.get("/{job_id}/requirement-draft")
def get_draft(request: Request, actor: Actor, job_id: str):
    with access(request, actor) as (db, _):
        return response(requirements.draft_view(db, job_id))


@router.get("/{job_id}/job-descriptions/{jd_id}")
def get_jd(request: Request, actor: Actor, job_id: str, jd_id: str):
    with access(request, actor) as (db, _):
        row = require(db.execute(
            "SELECT jd_id, media_type, extracted_text AS text FROM job_descriptions WHERE (job_id, jd_id) = (?, ?)",
            (job_id, jd_id),
        ).fetchone())
        return response(dict(row))


@router.post("/{job_id}/requirement-draft/retry", status_code=202)
def retry_draft(request: Request, actor: Actor, job_id: str, expected_revision: Annotated[NonBlank, Form()],
                file: Annotated[UploadFile | None, File()] = None):
    replacement = read_file(file, MAX_FILE_BYTES, jd=True) if file else None
    return write(request, actor, "requirements_edit",
                 {"expected_revision": expected_revision, "file": uploaded(replacement) if replacement else None},
                 lambda db: requirements.retry_draft(db, job_id, actor, expected_revision, replacement),
                 status=202, location=lambda _: f"{job_path(request, job_id)}/requirement-draft")


@router.put("/{job_id}/requirement-draft")
def update_draft(request: Request, actor: Actor, job_id: str, body: SaveDraft):
    data = body.model_dump()
    return write(request, actor, "requirements_edit", data,
                 lambda db: requirements.save_draft(db, job_id, actor, data), status=200)


@router.post("/{job_id}/requirement-draft/approve", status_code=201)
def approve_draft(request: Request, actor: Actor, job_id: str, body: Revision):
    return write(request, actor, "requirements_approve", body.model_dump(),
                 lambda db: requirements.approve_draft(db, job_id, actor, body.expected_revision),
                 status=201, location=lambda result: f"{job_path(request, job_id)}/requirement-profiles/{result['version']}")


@router.post("/{job_id}/requirement-draft", status_code=201)
def revise_profile(request: Request, actor: Actor, job_id: str, body: CreateRevision):
    return write(request, actor, "requirements_edit", body.model_dump(),
                 lambda db: requirements.create_revision(db, job_id, actor, body.base_version), status=201)


@router.get("/{job_id}/requirement-profiles/{version}")
def get_profile(request: Request, actor: Actor, job_id: str, version: Positive):
    with access(request, actor) as (db, _):
        return response(requirements.approved_view(db, job_id, version))


@router.post("/{job_id}/resume-batches", status_code=202)
def upload_resumes(request: Request, actor: Actor, job_id: str, files: Annotated[list[UploadFile], File()],
                   profile_version: Annotated[Positive, Form()], replacement_request_id: Annotated[NonBlank | None, Form()] = None):
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise WorkflowError(413, "BATCH_LIMIT", f"Upload at most {MAX_BATCH_FILES} files at a time.")
    items = []
    remaining = MAX_BATCH_BYTES
    for file in files:
        item = read_file(file, remaining)
        items.append(item)
        remaining -= len(item[0])
    data = {"profile_version": profile_version, "files": [uploaded(item) for item in items],
            "replacement_request_id": replacement_request_id}
    return write(request, actor, "upload", data, lambda db: intake.upload_batch(
        db, job_id, actor, profile_version, items, replacement_request_id,
    ), status=202, location=lambda result: f"{job_path(request, job_id)}/resume-batches/{result['batch_id']}")


@router.get("/{job_id}/resume-batches")
def list_batches(request: Request, actor: Actor, job_id: str, page: Pagination):
    with access(request, actor) as (db, _):
        rows = db.execute("SELECT batch_id FROM resume_batches WHERE job_id = ? ORDER BY uploaded_at DESC, batch_id",
                          (job_id,))

        def summary(row):
            batch = intake.batch_view(db, job_id, row["batch_id"])
            batch.pop("files")
            return {**batch, "href": f"{job_path(request, job_id)}/resume-batches/{row['batch_id']}"}

        return page_response(page, rows, summary)


@router.get("/{job_id}/resume-batches/{batch_id}")
def get_batch(request: Request, actor: Actor, job_id: str, batch_id: str):
    with access(request, actor) as (db, _):
        return response(intake.batch_view(db, job_id, batch_id))


@router.get("/{job_id}/candidates")
def list_candidates(request: Request, actor: Actor, job_id: str, page: Pagination, status: CandidateStatus | None = None):
    with access(request, actor) as (db, grants):
        rows = db.execute("SELECT candidate_id FROM candidates WHERE job_id = ? ORDER BY candidate_id", (job_id,))
        views = (intake.candidate_view(db, job_id, row["candidate_id"], can_review=allowed(grants, "review"),
                                       can_upload=allowed(grants, "upload")) for row in rows)
        return page_response(page, (view for view in views if status is None or view["status"] == status))


@router.get("/{job_id}/candidates/{candidate_id}")
def get_candidate(request: Request, actor: Actor, job_id: str, candidate_id: str):
    with access(request, actor) as (db, grants):
        return response(intake.candidate_view(db, job_id, candidate_id, can_review=allowed(grants, "review"),
                                              can_upload=allowed(grants, "upload")))


@router.get("/{job_id}/candidates/{candidate_id}/intake-attempts/{attempt_id}/audit")
def get_intake_audit(request: Request, actor: Actor, job_id: str, candidate_id: str, attempt_id: str):
    with access(request, actor, "audit") as (db, _):
        return response(intake.intake_packet(db, job_id, candidate_id, attempt_id, request.url.path.removesuffix("/audit")))


@router.get("/{job_id}/candidates/{candidate_id}/intake-attempts/{attempt_id}/excerpts/{span_id}")
def get_excerpt(request: Request, actor: Actor, job_id: str, candidate_id: str, attempt_id: str, span_id: str):
    with access(request, actor, "audit", write=True) as (db, _):
        return response(intake.excerpt_view(db, job_id, candidate_id, attempt_id, span_id, actor))


@router.post("/{job_id}/candidates/{candidate_id}/review-dispositions", status_code=201)
def review_attempt(request: Request, actor: Actor, job_id: str, candidate_id: str, body: Review):
    data = body.model_dump()
    return write(request, actor, "review", data, lambda db: intake.disposition(db, job_id, candidate_id, actor, data),
                 status=201)


@router.post("/{job_id}/candidates/{candidate_id}/intake-retries", status_code=202)
def retry_attempt(request: Request, actor: Actor, job_id: str, candidate_id: str, body: AttemptAction):
    data = body.model_dump()
    return write(request, actor, "upload", data, lambda db: intake.retry_intake(db, job_id, candidate_id, actor, data),
                 status=202, location=lambda _: f"{job_path(request, job_id)}/candidates/{candidate_id}")


@router.post("/{job_id}/candidates/{candidate_id}/replacement-requests", status_code=201)
def request_replacement(request: Request, actor: Actor, job_id: str, candidate_id: str, body: AttemptAction):
    data = body.model_dump()
    return write(request, actor, "upload", data,
                 lambda db: intake.request_replacement(db, job_id, candidate_id, actor, data), status=201)


@router.get("/{job_id}/score-plan")
def score_plan(request: Request, actor: Actor, job_id: str):
    with access(request, actor) as (db, grants):
        view = scoring.score_preview(db, job_id)
        if not allowed(grants, "score"):
            view.update(can_start=False, reason_code="PERMISSION_REQUIRED", plan_token=None)
        return response(view)


@router.post("/{job_id}/scoring-runs", status_code=202)
def start_scoring(request: Request, actor: Actor, job_id: str, body: ScoreRun):
    return write(request, actor, "score", body.model_dump(),
                 lambda db: scoring.accept_run(db, job_id, actor, body.plan_token),
                 status=202, location=lambda result: f"{job_path(request, job_id)}/scoring-runs/{result['run_id']}")


@router.get("/{job_id}/scoring-runs/{run_id}")
def get_run(request: Request, actor: Actor, job_id: str, run_id: str):
    with access(request, actor) as (db, _):
        return response(scoring.run_view(db, job_id, run_id))


@router.get("/{job_id}/rankings")
def list_rankings(request: Request, actor: Actor, job_id: str, page: Pagination):
    with access(request, actor) as (db, _):
        rows = db.execute("SELECT ranking_id FROM rankings WHERE job_id = ? ORDER BY published_at DESC, ranking_id", (job_id,))
        return page_response(page, rows, lambda row: scoring.ranking_view(db, job_id, row["ranking_id"]))


@router.get("/{job_id}/rankings/{ranking_id}")
def get_ranking(request: Request, actor: Actor, job_id: str, ranking_id: str):
    with access(request, actor) as (db, _):
        return response(scoring.ranking_view(db, job_id, ranking_id))


@router.get("/{job_id}/rankings/{ranking_id}/candidates")
def get_ranked_rows(request: Request, actor: Actor, job_id: str, ranking_id: str, page: Pagination,
                    status: Literal["ranked", "unranked"] | None = None):
    with access(request, actor) as (db, _):
        rows = scoring.ranking_rows(db, job_id, ranking_id, status=status)
        return page_response(page, rows, lambda row: scoring.ranking_candidate(row, job_path(request, job_id)))


@router.get("/{job_id}/rankings/{ranking_id}/candidates/{candidate_id}")
def get_ranked_candidate(request: Request, actor: Actor, job_id: str, ranking_id: str, candidate_id: str):
    with access(request, actor) as (db, grants):
        row = require(scoring.ranking_rows(db, job_id, ranking_id, candidate_id=candidate_id).fetchone())
        view = scoring.ranking_candidate(row, job_path(request, job_id), detail=True)
        scoring.add_evidence_locations(db, job_id, view)
        if allowed(grants, "audit"):
            view["audit_href"] = f"{view['candidate_href']}/intake-attempts/{view['attempt_id']}/audit"
        return response(view)


@router.get("/{job_id}/rankings/{ranking_id}/audit")
def get_ranking_audit(request: Request, actor: Actor, job_id: str, ranking_id: str):
    with access(request, actor, "audit") as (db, _):
        ranking = scoring.ranking_view(db, job_id, ranking_id)
        profile = requirements.approved_view(db, job_id, ranking["profile_version"])
        return response({
            "ranking": ranking, "run": scoring.run_view(db, job_id, ranking["run_id"]),
            "profile_approval": {key: profile[key] for key in ("profile_id", "version", "approved_by", "approved_at")},
            "events": events_for(db, job_id, "run_id", ranking["run_id"]),
            "candidates_href": f"{job_path(request, job_id)}/rankings/{ranking_id}/candidates",
        })


def current_ranking_id(db, job_id):
    return scoring.current_ranking(db, requirements.get_job(db, job_id))


@router.get("/{job_id}/shortlist-proposals/{proposal_id}")
def get_proposal(request: Request, actor: Actor, job_id: str, proposal_id: str):
    with access(request, actor) as (db, _):
        return response(shortlists.proposal_view(db, job_id, proposal_id, current_ranking_id(db, job_id)))


@router.put("/{job_id}/shortlist-proposals/{proposal_id}")
def update_proposal(request: Request, actor: Actor, job_id: str, proposal_id: str, body: ProposalEdit):
    data = body.model_dump()
    return write(request, actor, "shortlist_edit", data, lambda db: shortlists.save_proposal(
        db, job_id, proposal_id, actor, data, current_ranking_id(db, job_id),
    ), status=200)


@router.post("/{job_id}/shortlist-proposals/{proposal_id}/approve", status_code=201)
def approve_proposal(request: Request, actor: Actor, job_id: str, proposal_id: str, body: Revision):
    return write(request, actor, "shortlist_approve", body.model_dump(), lambda db: shortlists.approve_proposal(
        db, job_id, proposal_id, actor, body.expected_revision, current_ranking_id(db, job_id),
    ), status=201, location=lambda result: f"{job_path(request, job_id)}/shortlists/{result['shortlist_id']}")


@router.post("/{job_id}/rankings/{ranking_id}/shortlist-proposals", status_code=201)
def new_proposal(request: Request, actor: Actor, job_id: str, ranking_id: str, body: Body):
    def create(db):
        if current_ranking_id(db, job_id) != ranking_id:
            raise WorkflowError(409, "RANKING_NOT_CURRENT", "Choose the current ranking.")
        proposal_id = shortlists.create_proposal(db, job_id, ranking_id, actor)
        return shortlists.proposal_view(db, job_id, proposal_id, ranking_id)

    return write(request, actor, "shortlist_edit", body.model_dump(), create,
                 status=201, location=lambda result: f"{job_path(request, job_id)}/shortlist-proposals/{result['proposal_id']}")


@router.get("/{job_id}/shortlists")
def list_shortlists(request: Request, actor: Actor, job_id: str, page: Pagination):
    with access(request, actor) as (db, _):
        rows = db.execute("SELECT shortlist_id FROM shortlist_approvals WHERE job_id = ? ORDER BY approved_at DESC, shortlist_id",
                          (job_id,))
        return page_response(page, rows, lambda row: shortlists.approved_shortlist(db, job_id, row["shortlist_id"]))


@router.get("/{job_id}/shortlists/{shortlist_id}")
def get_shortlist(request: Request, actor: Actor, job_id: str, shortlist_id: str):
    with access(request, actor) as (db, grants):
        view = shortlists.approved_shortlist(db, job_id, shortlist_id)
        return response({**view, "export_formats": ["json"] if allowed(grants, "export") else []})


@router.get("/{job_id}/shortlists/{shortlist_id}/audit")
def get_shortlist_audit(request: Request, actor: Actor, job_id: str, shortlist_id: str):
    with access(request, actor, "audit") as (db, _):
        view = shortlists.approved_shortlist(db, job_id, shortlist_id)
        edits = db.execute(
            """SELECT * FROM shortlist_proposal_revisions WHERE (job_id, proposal_id) = (?, ?) ORDER BY revision_no""",
            (job_id, view["proposal_id"]),
        )
        history = []
        for row in edits:
            edit = dict(row)
            edit["overrides"] = json.loads(edit.pop("overrides_json"))
            edit["prior_override_decisions"] = json.loads(edit.pop("prior_override_decisions_json"))
            history.append(edit)
        events = {
            event["event_no"]: event for field, value in (("proposal_id", view["proposal_id"]), ("shortlist_id", shortlist_id))
            for event in events_for(db, job_id, field, value)
        }
        return response({
            "shortlist": view, "proposal_history": history, "events": [events[n] for n in sorted(events)],
            "ranking_audit_href": f"{job_path(request, job_id)}/rankings/{view['ranking_id']}/audit",
        })


@router.get("/{job_id}/shortlists/{shortlist_id}/export")
def export_shortlist(request: Request, actor: Actor, job_id: str, shortlist_id: str, format: Literal["json"] = "json"):
    with access(request, actor, "export", write=True) as (db, _):
        view = shortlists.approved_shortlist(db, job_id, shortlist_id)
        fields = ("job_id", "shortlist_id", "ranking_id", "profile_id", "profile_version", "size",
                  "approved_member_ids", "approved_at")
        audit(db, job_id, actor, "shortlist_exported", {"shortlist_id": shortlist_id}, {"format": format})
        return JSONResponse({key: view[key] for key in fields}, headers={
            "Content-Disposition": f'attachment; filename="shortlist-{shortlist_id}.json"', "Cache-Control": "no-store",
        })
