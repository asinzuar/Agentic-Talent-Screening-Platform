import hashlib
import json
import logging
import sqlite3

import pymupdf
from pydantic import ValidationError

from agentic_talent_screener.job_analysis import draft_requirement_profile
from agentic_talent_screener.ranking.schemas import Requirement, RequirementProfile
from agentic_talent_screener.workflow.storage import (
    SYSTEM_ACTOR, WorkflowError, audit, encode, expect_revision, new_id, require, utc_now,
)

MAX_JD_CHARACTERS = 100_000
MAX_JD_PAGES = 50
LOGGER = logging.getLogger(__name__)


def get_job(db: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    return require(db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())


def open_draft(db: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return db.execute(
        "SELECT * FROM requirement_drafts WHERE job_id = ? AND state <> 'approved'", (job_id,)
    ).fetchone()


def draft_revision(db: sqlite3.Connection, draft: sqlite3.Row) -> sqlite3.Row:
    return require(db.execute(
        "SELECT * FROM requirement_draft_revisions WHERE (job_id, draft_id, revision_no) = (?, ?, ?)",
        (draft["job_id"], draft["draft_id"], draft["current_revision_no"]),
    ).fetchone())


def draft_view(db: sqlite3.Connection, job_id: str) -> dict:
    draft = open_draft(db, job_id)
    if draft is None:
        return {"status": "none"}
    result = {
        "status": draft["state"], "stage": draft["stage"], "jd_id": draft["jd_id"],
        "revision": draft["revision_token"], "base_version": draft["base_version"],
    }
    if draft["state"] == "ready":
        revision = draft_revision(db, draft)
        result.update(json.loads(revision["content_json"]), role_title=revision["role_title"])
        result.pop("schema_version")
    if draft["error_json"]:
        result["error"] = json.loads(draft["error_json"])
        result["actions"] = {
            "retry": {"allowed": result["error"]["retryable"], "reason_code": (
                None if result["error"]["retryable"] else "NEW_JD_REQUIRED"
            )},
            "replace_jd": {"allowed": True, "reason_code": None},
        }
    return result


def approved_view(db: sqlite3.Connection, job_id: str, version: int) -> dict:
    row = require(db.execute(
        """SELECT p.approved_by, p.approved_at, r.jd_id, r.role_title, r.content_json, j.profile_id
        FROM requirement_profiles p JOIN jobs j USING (job_id)
        JOIN requirement_draft_revisions r USING (job_id, draft_id, revision_no)
        WHERE p.job_id = ? AND p.version = ?""",
        (job_id, version),
    ).fetchone())
    content = json.loads(row["content_json"])
    content.pop("schema_version")
    return {
        "profile_id": row["profile_id"], "version": version, "jd_id": row["jd_id"],
        "role_title": row["role_title"], "approved": True, "approved_by": row["approved_by"],
        "approved_at": row["approved_at"], **content,
    }


def scoring_profile(db: sqlite3.Connection, job_id: str, version: int) -> RequirementProfile:
    profile = approved_view(db, job_id, version)
    profile.pop("jd_id")
    profile.pop("flag_reviews")
    for criterion in profile["requirements"]:
        criterion["flags"] = [flag["message"] for flag in criterion["flags"]]
    return RequirementProfile.model_validate(profile)


def add_jd(db: sqlite3.Connection, job_id: str, actor: str, file: bytes, media: str) -> str:
    jd_id = new_id()
    db.execute(
        """INSERT INTO job_descriptions
        (job_id, jd_id, media_type, size_bytes, sha256, source_bytes, uploaded_by, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, jd_id, media, len(file), hashlib.sha256(file).hexdigest(), file, actor, utc_now()),
    )
    return jd_id


def create_job(db: sqlite3.Connection, actor: str, file: bytes, media: str) -> str:
    job_id, draft_id = new_id(), new_id()
    db.execute(
        "INSERT INTO jobs(job_id, profile_id, created_by, created_at) VALUES (?, ?, ?, ?)",
        (job_id, new_id(), actor, utc_now()),
    )
    db.execute("INSERT INTO job_permissions VALUES (?, ?, 'owner')", (job_id, actor))
    jd_id = add_jd(db, job_id, actor, file, media)
    db.execute(
        """INSERT INTO requirement_drafts
        (job_id, draft_id, jd_id, state, revision_token, created_by, created_at)
        VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
        (job_id, draft_id, jd_id, new_id(), actor, utc_now()),
    )
    audit(db, job_id, actor, "job_created", {"draft_id": draft_id, "jd_id": jd_id})
    return job_id


def retry_draft(db, job_id, actor, revision, replacement: tuple[bytes, str] | None) -> dict:
    draft = require(open_draft(db, job_id))
    expect_revision(draft["revision_token"], revision)
    if draft["state"] != "failed" or draft["base_version"] is not None:
        raise WorkflowError(409, "DRAFT_NOT_RETRYABLE", "Only a failed initial draft can be retried.")
    jd_id = draft["jd_id"]
    if replacement:
        jd_id = add_jd(db, job_id, actor, *replacement)
    elif not json.loads(draft["error_json"])["retryable"]:
        raise WorkflowError(409, "NEW_JD_REQUIRED", "Upload a replacement job description.")
    db.execute(
        """UPDATE requirement_drafts SET jd_id = ?, state = 'queued', stage = NULL, error_json = NULL,
        revision_token = ? WHERE (job_id, draft_id) = (?, ?)""",
        (jd_id, new_id(), job_id, draft["draft_id"]),
    )
    audit(db, job_id, actor, "draft_retry_requested", {"draft_id": draft["draft_id"], "jd_id": jd_id})
    return draft_view(db, job_id)


def append_revision(db, draft, actor, title, content, event, details=None) -> None:
    number = (draft["current_revision_no"] or 0) + 1
    db.execute(
        "INSERT INTO requirement_draft_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (draft["job_id"], draft["draft_id"], number, draft["jd_id"], title, encode(content), actor, utc_now()),
    )
    db.execute(
        """UPDATE requirement_drafts SET current_revision_no = ?, revision_token = ?, state = 'ready',
        stage = NULL, lease_token = NULL, lease_until = NULL WHERE (job_id, draft_id) = (?, ?)""",
        (number, new_id(), draft["job_id"], draft["draft_id"]),
    )
    audit(db, draft["job_id"], actor, event, {"draft_id": draft["draft_id"], "revision_no": number}, details)


def create_revision(db, job_id, actor, base_version: int) -> dict:
    if get_job(db, job_id)["active_profile_version"] != base_version or open_draft(db, job_id):
        raise WorkflowError(409, "STALE_DRAFT_BASE", "The base must be active and there must be no open draft.")
    profile = approved_view(db, job_id, base_version)
    draft_id = new_id()
    db.execute(
        """INSERT INTO requirement_drafts
        (job_id, draft_id, jd_id, base_version, state, revision_token, created_by, created_at)
        VALUES (?, ?, ?, ?, 'ready', ?, ?, ?)""",
        (job_id, draft_id, profile["jd_id"], base_version, new_id(), actor, utc_now()),
    )
    content = {key: profile[key] for key in ("requirements", "not_used_for_scoring", "flag_reviews")}
    append_revision(db, open_draft(db, job_id), actor, profile["role_title"],
                    {"schema_version": 1, **content}, "draft_revision_opened")
    return draft_view(db, job_id)


def save_draft(db, job_id, actor, body: dict) -> dict:
    draft = require(open_draft(db, job_id))
    expect_revision(draft["revision_token"], body["expected_revision"])
    if draft["state"] != "ready":
        raise WorkflowError(409, "DRAFT_NOT_READY", "Wait until a draft is ready.")
    old = json.loads(draft_revision(db, draft)["content_json"])
    existing = {criterion["requirement_id"]: criterion for criterion in old["requirements"]}
    jd = db.execute(
        "SELECT extracted_text FROM job_descriptions WHERE (job_id, jd_id) = (?, ?)", (job_id, draft["jd_id"]),
    ).fetchone()["extracted_text"]
    activations, criteria, used_ids, used_options = set(), [], set(), set()
    requested_activations = set(body["activate_requirement_ids"])
    try:
        for source in body["requirements"]:
            values = dict(source)
            requirement_id = values.get("requirement_id")
            previous = existing.get(requirement_id)
            if requirement_id is not None and previous is None:
                raise ValueError("Unknown requirement ID.")
            values["requirement_id"] = requirement_id or f"r-{new_id()}"
            values["provenance"] = previous["provenance"] if previous else "hr_added"
            if previous and source.get("provenance", previous["provenance"]) != previous["provenance"]:
                raise ValueError("Requirement provenance cannot be reassigned.")
            if previous and previous["status"] == "inactive" and values.get("status", "inactive") == "active":
                activations.add(requirement_id)
            flags = previous["flags"] if previous else []
            values["flags"] = [flag["message"] for flag in flags]
            for group in ("options", "substitutes"):
                values[group] = [
                    {**item, "option_id": item.get("option_id") or f"o-{new_id()}"}
                    for item in values.get(group, [])
                ]
            criterion = Requirement.model_validate(values)
            if criterion.requirement_id in used_ids or any(item.option_id in used_options for item in criterion.criteria):
                raise ValueError("Requirement and option IDs must be unique.")
            option_ids = {item.option_id for item in criterion.criteria}
            if len(option_ids) != len(criterion.criteria):
                raise ValueError("Option IDs must be unique.")
            used_ids.add(criterion.requirement_id)
            used_options.update(option_ids)
            if criterion.quote is not None and (not criterion.quote or criterion.quote not in jd):
                raise ValueError("JD quotes must be verbatim excerpts.")
            if criterion.min_evidence == 0:
                raise ValueError("Minimum evidence must be E1–E4 or null.")
            criteria.append({**criterion.model_dump(mode="json"), "flags": flags})
        if activations != requested_activations:
            raise ValueError("Explicitly confirm every criterion that is being activated.")
        if any(excerpt not in jd for excerpt in body["not_used_for_scoring"]):
            raise ValueError("Non-scoring passages must be verbatim JD excerpts.")
        flag_ids = {flag["flag_id"] for criterion in criteria for flag in criterion["flags"]}
        reviews = body["flag_reviews"]
        review_ids = {review["flag_id"] for review in reviews}
        if len(reviews) != len(review_ids) or not review_ids <= flag_ids:
            raise ValueError("Flag dispositions must reference distinct flags on retained criteria.")
    except ValidationError as error:
        raise WorkflowError(422, "INVALID_REQUIREMENTS", "A criterion or option has invalid fields.") from error
    except (TypeError, ValueError, AttributeError) as error:
        message = str(error) if type(error) is ValueError else "A criterion or option has invalid fields."
        raise WorkflowError(422, "INVALID_REQUIREMENTS", message) from error

    append_revision(
        db, draft, actor, body["role_title"],
        {"schema_version": 1, "requirements": criteria, "not_used_for_scoring": body["not_used_for_scoring"],
         "flag_reviews": reviews},
        "draft_saved", {"activated": sorted(activations), "removed": sorted(set(existing) - used_ids)},
    )
    return draft_view(db, job_id)


def approve_draft(db, job_id, actor, revision: str) -> dict:
    draft = require(open_draft(db, job_id))
    expect_revision(draft["revision_token"], revision)
    if draft["state"] != "ready":
        raise WorkflowError(409, "DRAFT_NOT_READY", "Wait until a draft is ready.")
    content = json.loads(draft_revision(db, draft)["content_json"])
    if not any(item["status"] == "active" for item in content["requirements"]):
        raise WorkflowError(422, "NO_ACTIVE_REQUIREMENTS", "Keep at least one active criterion.")
    flags = {flag["flag_id"] for item in content["requirements"] for flag in item["flags"]}
    if flags != {review["flag_id"] for review in content["flag_reviews"]}:
        raise WorkflowError(422, "UNREVIEWED_FLAGS", "Record a disposition for each remaining reviewer flag.")
    version = (get_job(db, job_id)["active_profile_version"] or 0) + 1
    db.execute(
        "INSERT INTO requirement_profiles VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, version, draft["draft_id"], draft["current_revision_no"], actor, utc_now()),
    )
    db.execute("UPDATE requirement_drafts SET state = 'approved' WHERE (job_id, draft_id) = (?, ?)",
               (job_id, draft["draft_id"]))
    db.execute("UPDATE shortlist_proposals SET state = 'superseded' WHERE job_id = ? AND state = 'draft'", (job_id,))
    db.execute("UPDATE jobs SET active_profile_version = ?, current_ranking_id = NULL WHERE job_id = ?",
               (version, job_id))
    audit(db, job_id, actor, "profile_approved", {"profile_version": version, "draft_id": draft["draft_id"]},
          {"revision_no": draft["current_revision_no"]})
    return approved_view(db, job_id, version)


def extract_jd(data: bytes, media: str) -> str:
    try:
        if media == "application/pdf":
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if not document.is_pdf or document.needs_pass or len(document) > MAX_JD_PAGES:
                    raise ValueError
                pages, length = [], 0
                for page in document:
                    source = page.get_text()
                    length += len(source) + 1
                    if length > MAX_JD_CHARACTERS:
                        raise ValueError
                    pages.append(source)
                text = "\n".join(pages)
        else:
            text = data.decode("utf-8-sig")
        if not text.strip() or len(text) > MAX_JD_CHARACTERS:
            raise ValueError
        return text
    except (ValueError, RuntimeError) as error:
        raise WorkflowError(422, "JD_UNREADABLE", "Upload readable text or a text-based PDF within the limits.") from error


def process_draft(task) -> None:
    with task.store.transaction() as db:
        draft = task.current(db)
        jd = require(db.execute("SELECT * FROM job_descriptions WHERE (job_id, jd_id) = (?, ?)",
                                (draft["job_id"], draft["jd_id"])).fetchone())
        job = get_job(db, draft["job_id"])
    task.update(stage="extract_jd")
    text = jd["extracted_text"] or extract_jd(jd["source_bytes"] or b"", jd["media_type"])
    with task.store.transaction(write=True) as db:
        task.current(db)
        db.execute("UPDATE job_descriptions SET extracted_text = ? WHERE (job_id, jd_id) = (?, ?) AND extracted_text IS NULL",
                   (text, draft["job_id"], jd["jd_id"]))
    task.update(stage="draft_requirements")
    profile = draft_requirement_profile(text, profile_id=job["profile_id"], reasoning=False)
    content = profile.model_dump(mode="json", include={"requirements", "not_used_for_scoring"})
    for criterion in content["requirements"]:
        criterion["flags"] = [{"flag_id": new_id(), "message": message} for message in criterion["flags"]]
    with task.store.transaction(write=True) as db:
        current = task.current(db)
        append_revision(db, current, SYSTEM_ACTOR, profile.role_title,
                        {"schema_version": 1, "flag_reviews": [], **content}, "draft_generated")
    LOGGER.info("Requirements draft ready: %s criteria=%d active=%d", task.log_context,
                len(profile.requirements), len(profile.active_requirements))
