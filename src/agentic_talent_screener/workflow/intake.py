import hashlib
import json
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_talent_screener.llm import default_llm_model
from agentic_talent_screener.resume_intake.contracts import SourceSpan
from agentic_talent_screener.resume_intake.l1_injection import SecurityAssessment
from agentic_talent_screener.resume_intake.pii_redaction import redact_source_spans
from agentic_talent_screener.resume_intake.pipeline import normalize_resume
from agentic_talent_screener.resume_intake.ranking_adapter import to_ranking_profile
from agentic_talent_screener.workflow.requirements import get_job
from agentic_talent_screener.workflow.storage import (
    SYSTEM_ACTOR, WorkflowError, audit, encode, expect_revision, new_id, require, utc_now,
)

RESUME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}
MEDIA_EXTENSIONS = {media: extension for extension, media in RESUME_TYPES.items()}
RISK = {"low": 0, "medium": 1, "high": 2}
LOGGER = logging.getLogger(__name__)


def candidate_row(db, job_id, candidate_id):
    return require(db.execute("SELECT * FROM candidates WHERE (job_id, candidate_id) = (?, ?)",
                              (job_id, candidate_id)).fetchone())


def attempt_row(db, job_id, candidate_id, attempt_id):
    return require(db.execute(
        "SELECT * FROM intake_attempts WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)",
        (job_id, candidate_id, attempt_id),
    ).fetchone())


def flags_for(db, job_id, candidate_id, attempt_id) -> list[dict]:
    return [dict(row) for row in db.execute(
        """SELECT finding_id, layer, category, attack_family, risk, detector, action_code
        FROM intake_findings WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?) ORDER BY finding_id""",
        (job_id, candidate_id, attempt_id),
    )]


def eligibility(db, attempt) -> tuple[str, bool, dict | None]:
    keys = (attempt["job_id"], attempt["candidate_id"], attempt["attempt_id"])
    usable = db.execute(
        "SELECT 1 FROM verified_profiles WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)", keys,
    ).fetchone() is not None
    latest = db.execute(
        """SELECT disposition_id, action, reason, actor_id, recorded_at FROM intake_review_dispositions
        WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?) ORDER BY decision_no DESC LIMIT 1""",
        keys,
    ).fetchone()
    disposition = dict(latest) if latest else None
    if attempt["state"] != "finished":
        return "processing", usable, disposition
    if attempt["outcome"] == "quarantined":
        return "quarantined", usable, disposition
    if attempt["outcome"] in {"verified", "verified_with_flag"} and usable and (
        not attempt["requires_review"] or (disposition and disposition["action"] == "clear_for_scoring")
    ):
        return "ready_to_score", usable, disposition
    return "review_required", usable, disposition


def attempt_summary(db, attempt) -> dict:
    status, _, _ = eligibility(db, attempt)
    provenance = json.loads(attempt["provenance_json"] or "{}")
    return {
        "candidate_id": attempt["candidate_id"], "attempt_id": attempt["attempt_id"], "status": status,
        "stage": attempt["stage"] if status == "processing" else None,
        "overall_risk": attempt["overall_risk"], "issue_codes": provenance.get("issue_codes", []),
        "hold_code": attempt["hold_code"] if status == "review_required" else None,
    }


def action(allowed: bool, reason: str = "ACTION_UNAVAILABLE") -> dict:
    return {"allowed": bool(allowed), "reason_code": None if allowed else reason}


def candidate_view(db, job_id, candidate_id, *, can_review: bool = True, can_upload: bool = True) -> dict:
    candidate = candidate_row(db, job_id, candidate_id)
    attempt = attempt_row(db, job_id, candidate_id, candidate["current_attempt_id"])
    status, usable, disposition = eligibility(db, attempt)
    open_request = db.execute(
        "SELECT request_id FROM replacement_requests WHERE job_id = ? AND candidate_id = ? AND state = 'open'",
        (job_id, candidate_id),
    ).fetchone()
    held = status == "review_required"
    return {
        **attempt_summary(db, attempt), "revision": candidate["revision_token"],
        "latest_disposition": disposition, "security_flags": flags_for(db, job_id, candidate_id, attempt["attempt_id"]),
        "replacement_request_id": open_request["request_id"] if open_request else None,
        "actions": {
            "clear_for_scoring": action(can_review and held and usable, "PROFILE_NOT_CLEARABLE"),
            "keep_on_hold": action(can_review and held, "HOLD_REQUIRED"),
            "retry": action(can_upload and held and not usable and attempt["retryable"], "INTAKE_NOT_RETRYABLE"),
            "request_replacement": action(can_upload and not open_request
                                          and (status == "quarantined" or (held and not usable))),
        },
    }


def queue_attempt(db, job_id, candidate_id, attempt_id, document_id, actor, kind, prior=None) -> None:
    db.execute(
        """INSERT INTO intake_attempts (job_id, candidate_id, attempt_id, document_id, prior_attempt_id,
        trigger_kind, requested_by, state, stage, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', ?)""",
        (job_id, candidate_id, attempt_id, document_id, prior, kind, actor, utc_now()),
    )


def upload_batch(db, job_id, actor, version, files, replacement_id=None) -> dict:
    active = get_job(db, job_id)["active_profile_version"]
    if active is None:
        raise WorkflowError(409, "REQUIREMENTS_NOT_APPROVED", "Approve requirements before uploading resumes.")
    if active != version:
        raise WorkflowError(409, "STALE_PROFILE_VERSION", "Confirm the active requirements version.")
    replacement = None
    if replacement_id is not None:
        if len(files) != 1:
            raise WorkflowError(422, "ONE_REPLACEMENT_FILE", "Upload exactly one replacement file.")
        replacement = require(db.execute(
            """SELECT r.* FROM replacement_requests r JOIN candidates c USING (job_id, candidate_id)
            WHERE r.job_id = ? AND r.request_id = ?""", (job_id, replacement_id),
        ).fetchone())
        candidate = candidate_row(db, job_id, replacement["candidate_id"])
        if replacement["state"] != "open" or candidate["current_attempt_id"] != replacement["attempt_id"]:
            raise WorkflowError(409, "STALE_REPLACEMENT_REQUEST", "The replacement request is no longer open.")
    batch_id, created = new_id(), utc_now()
    db.execute("INSERT INTO resume_batches VALUES (?, ?, ?, ?, ?)", (job_id, batch_id, version, actor, created))
    last = db.execute("SELECT max(candidate_id) FROM candidates WHERE job_id = ?", (job_id,)).fetchone()[0]
    next_number = int(last[2:]) + 1 if last else 1
    for position, (data, media) in enumerate(files, start=1):
        attempt_id, document_id = new_id(), new_id()
        if replacement:
            candidate_id = replacement["candidate_id"]
        else:
            if next_number > 9999:
                raise WorkflowError(409, "POOL_LIMIT", "This job has no unused blind candidate IDs.")
            candidate_id = f"C-{next_number:04d}"
            next_number += 1
            db.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                       (job_id, candidate_id, attempt_id, new_id(), created))
        db.execute("INSERT INTO resume_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (job_id, candidate_id, document_id, batch_id, position, hashlib.sha256(data).hexdigest(),
                    len(data), media, created))
        queue_attempt(db, job_id, candidate_id, attempt_id, document_id, actor,
                      "replacement" if replacement else "upload", replacement["attempt_id"] if replacement else None)
        db.execute("INSERT INTO resume_payloads VALUES (?, ?, ?, ?)", (job_id, candidate_id, document_id, data))
        if replacement:
            db.execute("UPDATE candidates SET current_attempt_id = ?, revision_token = ? WHERE (job_id, candidate_id) = (?, ?)",
                       (attempt_id, new_id(), job_id, candidate_id))
            db.execute(
                """UPDATE replacement_requests SET state = 'fulfilled', closed_at = ?, replacement_document_id = ?
                WHERE (job_id, request_id) = (?, ?)""", (created, document_id, job_id, replacement_id),
            )
    audit(db, job_id, actor, "resume_batch_uploaded", {"batch_id": batch_id},
          {"file_count": len(files), "replacement_request_id": replacement_id})
    return batch_view(db, job_id, batch_id)


def batch_view(db, job_id, batch_id) -> dict:
    batch = dict(require(db.execute("SELECT * FROM resume_batches WHERE (job_id, batch_id) = (?, ?)",
                                    (job_id, batch_id)).fetchone()))
    rows = db.execute(
        """SELECT a.*, d.upload_position FROM resume_documents d JOIN intake_attempts a
        USING (job_id, candidate_id, document_id)
        WHERE (d.job_id, d.batch_id) = (?, ?) AND a.trigger_kind IN ('upload','replacement')
        ORDER BY d.upload_position""", (job_id, batch_id),
    )
    batch["files"] = [{"upload_position": row["upload_position"], **attempt_summary(db, row)} for row in rows]
    batch["status_counts"] = {}
    for file in batch["files"]:
        status = file["status"]
        batch["status_counts"][status] = batch["status_counts"].get(status, 0) + 1
    return batch


def checked_attempt(db, job_id, candidate_id, body: dict):
    candidate = candidate_row(db, job_id, candidate_id)
    expect_revision(candidate["revision_token"], body["expected_revision"])
    if candidate["current_attempt_id"] != body["attempt_id"]:
        raise WorkflowError(409, "STALE_INTAKE_ATTEMPT", "Reload the current intake attempt.")
    return attempt_row(db, job_id, candidate_id, body["attempt_id"])


def disposition(db, job_id, candidate_id, actor, body: dict) -> dict:
    attempt = checked_attempt(db, job_id, candidate_id, body)
    status, usable, _ = eligibility(db, attempt)
    if status != "review_required" or (body["action"] == "clear_for_scoring" and not usable):
        raise WorkflowError(409, "PROFILE_NOT_CLEARABLE", "This disposition is unavailable for the current attempt.")
    attempt_id = attempt["attempt_id"]
    decision_no = db.execute(
        """SELECT coalesce(max(decision_no), 0) + 1 FROM intake_review_dispositions
        WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)""", (job_id, candidate_id, attempt_id),
    ).fetchone()[0]
    disposition_id = new_id()
    db.execute(
        "INSERT INTO intake_review_dispositions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, candidate_id, attempt_id, disposition_id, decision_no, body["action"], body["reason"], actor, utc_now()),
    )
    db.execute("UPDATE candidates SET revision_token = ? WHERE (job_id, candidate_id) = (?, ?)",
               (new_id(), job_id, candidate_id))
    audit(db, job_id, actor, "intake_review_recorded",
          {"candidate_id": candidate_id, "attempt_id": attempt_id, "disposition_id": disposition_id},
          {"action": body["action"]})
    return candidate_view(db, job_id, candidate_id)


def retry_intake(db, job_id, candidate_id, actor, body: dict) -> dict:
    attempt = checked_attempt(db, job_id, candidate_id, body)
    status, usable, _ = eligibility(db, attempt)
    if status != "review_required" or usable or not attempt["retryable"] or not db.execute(
        "SELECT 1 FROM resume_payloads WHERE (job_id, candidate_id, document_id) = (?, ?, ?)",
        (job_id, candidate_id, attempt["document_id"]),
    ).fetchone():
        raise WorkflowError(409, "INTAKE_NOT_RETRYABLE", "Request a replacement instead of retrying this file.")
    attempt_id = new_id()
    queue_attempt(db, job_id, candidate_id, attempt_id, attempt["document_id"], actor, "retry", attempt["attempt_id"])
    db.execute("UPDATE candidates SET current_attempt_id = ?, revision_token = ? WHERE (job_id, candidate_id) = (?, ?)",
               (attempt_id, new_id(), job_id, candidate_id))
    db.execute("""UPDATE replacement_requests SET state = 'superseded', closed_at = ?
               WHERE (job_id, candidate_id) = (?, ?) AND state = 'open'""", (utc_now(), job_id, candidate_id))
    audit(db, job_id, actor, "intake_retry_requested", {"candidate_id": candidate_id, "attempt_id": attempt_id})
    return candidate_view(db, job_id, candidate_id)


def request_replacement(db, job_id, candidate_id, actor, body: dict) -> dict:
    attempt = checked_attempt(db, job_id, candidate_id, body)
    status, usable, _ = eligibility(db, attempt)
    if status != "quarantined" and (status != "review_required" or usable):
        raise WorkflowError(409, "REPLACEMENT_UNAVAILABLE", "A replacement is unavailable for this attempt.")
    if db.execute("SELECT 1 FROM replacement_requests WHERE (job_id, candidate_id) = (?, ?) AND state = 'open'",
                  (job_id, candidate_id)).fetchone():
        raise WorkflowError(409, "REPLACEMENT_ALREADY_REQUESTED", "A replacement is already requested.")
    request_id, now = new_id(), utc_now()
    db.execute("INSERT INTO replacement_requests VALUES (?, ?, ?, ?, ?, ?, 'open', NULL, NULL)",
               (job_id, candidate_id, request_id, attempt["attempt_id"], actor, now))
    db.execute("UPDATE candidates SET revision_token = ? WHERE (job_id, candidate_id) = (?, ?)",
               (new_id(), job_id, candidate_id))
    audit(db, job_id, actor, "replacement_requested",
          {"candidate_id": candidate_id, "attempt_id": attempt["attempt_id"], "request_id": request_id})
    return {"request_id": request_id, "candidate_id": candidate_id, "attempt_id": attempt["attempt_id"],
            "state": "open", "requested_by": actor, "requested_at": now}


def intake_packet(db, job_id, candidate_id, attempt_id, prefix: str) -> dict:
    attempt = attempt_row(db, job_id, candidate_id, attempt_id)
    keys = (job_id, candidate_id, attempt_id)
    profile = db.execute("SELECT profile_json FROM verified_profiles WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)",
                         keys).fetchone()
    spans = {}
    for row in db.execute(
        """SELECT span_id, page, location_json, visibility, excluded_from_extraction, excluded_from_evidence,
        controlled_excerpt IS NOT NULL AS excerpt_available
        FROM intake_spans WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)""", keys,
    ):
        span = dict(row)
        span["location"] = json.loads(span.pop("location_json") or "null")
        if span.pop("excerpt_available"):
            span["excerpt_href"] = f"{prefix}/excerpts/{span['span_id']}"
        spans[span["span_id"]] = span
    findings = {finding["finding_id"]: {**finding, "spans": []} for finding in flags_for(db, *keys)}
    for row in db.execute(
        "SELECT finding_id, span_id FROM intake_finding_spans WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)",
        keys,
    ):
        findings[row["finding_id"]]["spans"].append(spans[row["span_id"]])
    dispositions = [dict(row) for row in db.execute(
        """SELECT disposition_id, action, reason, actor_id, recorded_at FROM intake_review_dispositions
        WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?) ORDER BY decision_no""", keys,
    )]
    sha = db.execute("SELECT sha256 FROM resume_documents WHERE (job_id, candidate_id, document_id) = (?, ?, ?)",
                     (job_id, candidate_id, attempt["document_id"])).fetchone()[0]
    return {
        **attempt_summary(db, attempt), "document_sha256": sha, "outcome": attempt["outcome"],
        "provenance": json.loads(attempt["provenance_json"] or "{}"), "error": json.loads(attempt["error_json"] or "null"),
        "verified_profile": json.loads(profile["profile_json"]) if profile else None,
        "findings": list(findings.values()),
        "exclusions": [span for span in spans.values() if span["excluded_from_evidence"]],
        "dispositions": dispositions,
    }


def excerpt_view(db, job_id, candidate_id, attempt_id, span_id, actor) -> dict:
    row = require(db.execute(
        """SELECT page, location_json, controlled_excerpt FROM intake_spans
        WHERE (job_id, candidate_id, attempt_id, span_id) = (?, ?, ?, ?) AND controlled_excerpt IS NOT NULL""",
        (job_id, candidate_id, attempt_id, span_id),
    ).fetchone())
    audit(db, job_id, actor, "excerpt_accessed",
          {"candidate_id": candidate_id, "attempt_id": attempt_id, "span_id": span_id})
    return {"span_id": span_id, "page": row["page"], "location": json.loads(row["location_json"] or "null"),
            "excerpt": row["controlled_excerpt"]}


def persist_findings(db, keys: tuple, report: dict, assessment: SecurityAssessment, excluded: set[str]) -> None:
    raw = report.get("details", {}).get("spans", [])
    spans, _ = redact_source_spans(tuple(SourceSpan.from_l0(span, None, None) for span in raw))
    safe_spans = {span.span_id: span for span in spans}
    for span in spans:
        omitted = span.span_id in excluded or span.visibility != "visible"
        # The excerpt is only retained for reviewed/excluded content. The routine profile uses L3 quotes.
        excerpt = (span.text[:300] or None) if omitted else None
        location = {"char_start": span.char_start, "char_end": span.char_end, "bbox": span.bbox}
        db.execute("INSERT INTO intake_spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (*keys, span.span_id, span.page, encode(location), span.visibility, int(omitted), int(omitted), excerpt))
    findings = []
    for issue in report.get("issues", []):
        category = issue["code"]
        family, locations = None, []
        if category in {"PDF_NONVISIBLE_TEXT", "DOCX_NONVISIBLE_TEXT"}:
            family, locations = "A3", [span.span_id for span in spans if span.visibility != "visible"]
        elif category == "UNICODE_ZERO_WIDTH_OR_FORMAT":
            family = "A4"
            locations = [span["span_id"] for span in raw if span.get("normalisation", {}).get("zero_width_or_format")]
        findings.append(("L0", category, family, issue["severity"], "document_forensics", "flag", locations))
    findings.extend(
        ("L1", finding.code, finding.attack_family, finding.risk, finding.detector,
         finding.recommended_action, finding.span_ids)
        for finding in assessment.findings
    )
    for i, (layer, category, family, risk, detector, action_code, locations) in enumerate(findings, start=1):
        finding_id = f"F-{i:05d}"
        db.execute("INSERT INTO intake_findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (*keys, finding_id, layer, category, family, risk, detector, action_code))
        db.executemany("INSERT INTO intake_finding_spans VALUES (?, ?, ?, ?, ?)",
                       [(*keys, finding_id, span_id) for span_id in set(locations) if span_id in safe_spans])


def process_intake(task) -> None:
    with task.store.transaction() as db:
        attempt = task.current(db)
        row = require(db.execute(
            """SELECT p.source_bytes, d.media_type FROM resume_payloads p JOIN resume_documents d
            USING (job_id, candidate_id, document_id) WHERE (p.job_id, p.candidate_id, p.document_id) = (?, ?, ?)""",
            (attempt["job_id"], attempt["candidate_id"], attempt["document_id"]),
        ).fetchone())
    report = {}
    model = default_llm_model()
    with TemporaryDirectory(prefix="ats-intake-") as directory:
        path = Path(directory) / ("resume" + MEDIA_EXTENSIONS.get(row["media_type"], ".unknown"))
        path.write_bytes(row["source_bytes"])
        result = normalize_resume(
            path, model=model, candidate_id=attempt["candidate_id"], show_progress=False,
            on_stage=lambda stage: task.update(stage=stage), on_forensics=report.update,
        )
    assessment = result.security_assessment or SecurityAssessment(
        risk="low", injection_risk_score=0, decision="allow",
    )
    overall = max(
        [assessment.risk, report.get("risk_level", "low"), *[issue["severity"] for issue in report.get("issues", [])]],
        key=lambda value: RISK.get(value, 0),
    )
    if overall not in RISK:
        overall = assessment.risk
    profile = None
    mandatory = result.requires_human_review or overall == "high"
    outcome = result.status if result.status == "quarantined" else "manual_review"
    if result.status in {"normalized", "normalized_with_flag"} and result.profile:
        outcome = "verified_with_flag" if mandatory or result.status == "normalized_with_flag" else "verified"
        handoff = to_ranking_profile(result.profile, assessment, forensic_report=report,
                                     intake_status=outcome, requires_human_review=mandatory)
        profile = handoff.profile if handoff.profile.facts else None
        if profile is None:
            outcome = "manual_review"
    issue_codes = sorted({item for item in result.events if re.fullmatch(r"[A-Z][A-Z0-9_]{2,95}", item)}
                         | {issue["code"] for issue in report.get("issues", [])})
    retryable = outcome == "manual_review" and any("UNAVAILABLE" in code for code in issue_codes)
    provenance = {"schema_version": 1, "pipeline_schema": result.schema_version, "model": model,
                  "forensics_schema": report.get("schema_version"), "l3_model": model,
                  "stopped_at": result.stopped_at, "issue_codes": issue_codes}
    if result.profile:
        provenance["verification_schema"] = result.profile.schema_version
        provenance["verification_events"] = [
            event.model_dump(mode="json", include={"claim_id", "code", "source_span_ids"})
            for event in result.profile.verification_events
        ]
        provenance["verification_methods"] = {fact.fact_id: fact.verification_method for fact in result.profile.facts}
    hold_code = "CLEARANCE_REQUIRED" if profile and mandatory else None
    if outcome == "manual_review":
        hold_code = "NO_USABLE_VERIFIED_PROFILE"
    with task.store.transaction(write=True) as db:
        task.current(db)
        keys = (attempt["job_id"], attempt["candidate_id"], attempt["attempt_id"])
        persist_findings(db, keys, report, assessment, set(result.sanitised_span_ids))
        if profile:
            db.execute("INSERT INTO verified_profiles VALUES (?, ?, ?, 1, ?, ?)",
                       (*keys, utc_now(), profile.model_dump_json()))
        if outcome == "quarantined":
            db.execute("DELETE FROM resume_payloads WHERE (job_id, candidate_id, document_id) = (?, ?, ?)",
                       (keys[0], keys[1], attempt["document_id"]))
        task.finish(db, state="finished", outcome=outcome, overall_risk=overall, requires_review=int(mandatory),
                    hold_code=hold_code, retryable=int(retryable), provenance_json=encode(provenance),
                    finished_at=utc_now())
        db.execute("UPDATE candidates SET revision_token = ? WHERE (job_id, candidate_id, current_attempt_id) = (?, ?, ?)",
                   (new_id(), *keys))
        audit(db, keys[0], SYSTEM_ACTOR, "intake_finished", {"candidate_id": keys[1], "attempt_id": keys[2]},
              {"outcome": outcome, "risk": overall, "issue_codes": issue_codes})
    forensics_codes = {issue["code"] for issue in report.get("issues", [])}
    review_codes = {event.code for event in result.event_details if event.severity != "info"}
    LOGGER.info(
        "Intake outcome: %s outcome=%s stopped_at=%s risk=%s hold=%s retryable=%s verified_facts=%d findings=%d issues=%s",
        task.log_context, outcome, result.stopped_at, overall, hold_code or "none", retryable,
        len(profile.facts) if profile else 0, len(report.get("issues", [])) + len(assessment.findings),
        ",".join(sorted(forensics_codes | review_codes)) or "none",
    )
