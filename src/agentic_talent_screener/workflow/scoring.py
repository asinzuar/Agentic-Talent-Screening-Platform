import hmac
import json
import logging
from collections import Counter

from agentic_talent_screener.file_logging import exception_site
from agentic_talent_screener.llm import default_llm_model, model_log_context
from agentic_talent_screener.ranking.config import DEFAULT_CONFIG, ScoringConfig
from agentic_talent_screener.ranking.matching import LLMMatcher
from agentic_talent_screener.ranking.schemas import CandidateMatches, VerifiedProfile
from agentic_talent_screener.ranking.scorer import score_candidate
from agentic_talent_screener.workflow.intake import attempt_row, candidate_row, eligibility, flags_for
from agentic_talent_screener.workflow.requirements import get_job, scoring_profile
from agentic_talent_screener.workflow.shortlists import create_proposal
from agentic_talent_screener.workflow.storage import (
    SYSTEM_ACTOR, WorkflowError, audit, digest, encode, new_id, require, utc_now,
)

SCORER_VERSION = "workflow-scorer-v2"
LOGGER = logging.getLogger(__name__)


def scoring_basis(version):
    if version is None:
        return None
    model = default_llm_model()
    values = {
        "profile_version": version, "matcher_version": LLMMatcher(model).version,
        "scorer_version": SCORER_VERSION, "config_version": DEFAULT_CONFIG.config_version,
        "settings": {"model": model, "config": DEFAULT_CONFIG.model_dump(mode="json")},
    }
    return {**values, "fingerprint": digest(values)}


def latest_ranking(db, job_id):
    return db.execute(
        """SELECT k.ranking_id, b.fingerprint FROM rankings k JOIN scoring_runs r USING (job_id, run_id)
        JOIN scoring_bases b USING (job_id, basis_id) WHERE k.job_id = ? ORDER BY k.published_at DESC LIMIT 1""",
        (job_id,),
    ).fetchone()


def current_ranking(db, job):
    if job["current_ranking_id"] is None:
        return None
    basis = scoring_basis(job["active_profile_version"])
    row = db.execute(
        """SELECT b.fingerprint FROM rankings k JOIN scoring_runs r USING (job_id, run_id)
        JOIN scoring_bases b USING (job_id, basis_id) WHERE (k.job_id, k.ranking_id) = (?, ?)""",
        (job["job_id"], job["current_ranking_id"]),
    ).fetchone()
    return job["current_ranking_id"] if basis and row["fingerprint"] == basis["fingerprint"] else None


def plan_inputs(db, job_id):
    job = get_job(db, job_id)
    version = job["active_profile_version"]
    basis = scoring_basis(version)
    latest = latest_ranking(db, job_id)
    incremental = basis is not None and latest is not None and latest["fingerprint"] == basis["fingerprint"]
    reusable = {}
    if incremental:
        reusable = {
            (row["candidate_id"], row["attempt_id"]): row["score_id"]
            for row in db.execute(
                """SELECT c.candidate_id, c.attempt_id, c.score_id FROM ranking_rows k
                JOIN scoring_run_candidates c USING (job_id, run_id, candidate_id)
                WHERE (k.job_id, k.ranking_id) = (?, ?) AND k.rank IS NOT NULL""", (job_id, latest["ranking_id"]),
            )
        }
    excluded = Counter()
    to_score = reuse = 0
    roster = []
    attempts = db.execute(
        """SELECT a.* FROM candidates c JOIN intake_attempts a ON (c.job_id,c.candidate_id,c.current_attempt_id)
        = (a.job_id,a.candidate_id,a.attempt_id) WHERE c.job_id = ? ORDER BY c.candidate_id""", (job_id,),
    )
    for attempt in attempts:
        status, _, disposition = eligibility(db, attempt)
        candidate_id, attempt_id = attempt["candidate_id"], attempt["attempt_id"]
        score_id = reusable.get((candidate_id, attempt_id)) if status == "ready_to_score" else None
        clearance_id = disposition["disposition_id"] if status == "ready_to_score" and attempt["requires_review"] else None
        if status != "ready_to_score":
            selected = "exclude"
            excluded[status] += 1
        elif score_id is not None:
            selected = "reuse"
            reuse += 1
        else:
            selected = "score"
            to_score += 1
        roster.append({
            "candidate_id": candidate_id, "attempt_id": attempt_id, "clearance_id": clearance_id,
            "planned_action": selected, "score_id": score_id, "excluded_reason": status if selected == "exclude" else None,
            "captured_risk": attempt["overall_risk"], "findings": flags_for(db, job_id, candidate_id, attempt_id),
        })
    reason = None
    if version is None:
        reason = "REQUIREMENTS_NOT_APPROVED"
    elif db.execute("SELECT 1 FROM scoring_runs WHERE job_id = ? AND state IN ('queued','running')", (job_id,)).fetchone():
        reason = "RUN_IN_PROGRESS"
    elif not to_score + reuse:
        reason = "NO_READY_CANDIDATES"
    elif not to_score and reuse == len(reusable):
        reason = "NO_NEW_WORK"
    mode = ("incremental" if incremental else "full") if basis else None
    title = scoring_profile(db, job_id, version).role_title if version else None
    summary = {
        "role_title": title, "profile_id": job["profile_id"] if version else None, "profile_version": version,
        "matcher_version": basis["matcher_version"] if basis else None, "config_version": basis["config_version"] if basis else None,
        "mode": mode, "pool_count": len(roster), "ready_count": to_score + reuse, "to_score_count": to_score,
        "reuse_count": reuse, "excluded_counts": dict(excluded), "can_start": reason is None, "reason_code": reason,
    }
    fingerprint = digest({"job_id": job_id, "basis": basis, "mode": mode, "roster": roster})
    return summary, basis, roster, fingerprint


def signed_plan(db, fingerprint: str) -> str:
    key = db.execute("SELECT value FROM workflow_secrets WHERE name = 'score_plan'").fetchone()[0]
    return hmac.digest(key, fingerprint.encode(), "sha256").hex()


def score_preview(db, job_id) -> dict:
    summary, _, _, fingerprint = plan_inputs(db, job_id)
    return {**summary, "plan_token": signed_plan(db, fingerprint) if summary["can_start"] else None}


def accept_run(db, job_id, actor, token: str) -> dict:
    summary, basis, roster, fingerprint = plan_inputs(db, job_id)
    if not summary["can_start"]:
        raise WorkflowError(409, summary["reason_code"], "Refresh the scoring plan before starting a run.")
    if not hmac.compare_digest(signed_plan(db, fingerprint), token):
        raise WorkflowError(409, "SCORE_PLAN_STALE", "Confirm the updated scoring plan.")
    stored = db.execute("SELECT basis_id FROM scoring_bases WHERE (job_id, fingerprint) = (?, ?)",
                        (job_id, basis["fingerprint"])).fetchone()
    basis_id = stored["basis_id"] if stored else new_id()
    if not stored:
        db.execute("INSERT INTO scoring_bases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (job_id, basis_id, basis["profile_version"], basis["matcher_version"], basis["scorer_version"],
                    basis["config_version"], encode(basis["settings"]), basis["fingerprint"]))
    run_id = new_id()
    db.execute(
        """INSERT INTO scoring_runs (job_id, run_id, basis_id, mode, plan_fingerprint, requested_by, state, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)""", (job_id, run_id, basis_id, summary["mode"], fingerprint, actor, utc_now()),
    )
    for item in roster:
        state = {"score": "pending", "reuse": "reused", "exclude": "excluded"}[item["planned_action"]]
        db.execute("INSERT INTO scoring_run_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                   (job_id, run_id, item["candidate_id"], basis_id, item["attempt_id"], item["clearance_id"],
                    item["planned_action"], item["excluded_reason"], item["captured_risk"], encode(item["findings"]),
                    state, item["score_id"]))
    audit(db, job_id, actor, "scoring_started", {"run_id": run_id}, {"mode": summary["mode"]})
    return run_view(db, job_id, run_id)


def run_view(db, job_id, run_id) -> dict:
    run = require(db.execute(
        """SELECT r.*, b.profile_version, b.matcher_version, b.config_version, j.profile_id, k.ranking_id
        FROM scoring_runs r JOIN scoring_bases b USING (job_id, basis_id) JOIN jobs j USING (job_id)
        LEFT JOIN rankings k USING (job_id, run_id) WHERE (r.job_id,r.run_id) = (?, ?)""", (job_id, run_id),
    ).fetchone())
    rows = db.execute("SELECT state, planned_action, excluded_reason FROM scoring_run_candidates WHERE (job_id,run_id)=(?,?)",
                      (job_id, run_id)).fetchall()
    planned = Counter(row["planned_action"] for row in rows)
    counts = Counter(row["state"] for row in rows)
    excluded = Counter(row["excluded_reason"] for row in rows if row["excluded_reason"])
    return {
        **{key: run[key] for key in ("job_id", "run_id", "mode", "profile_id", "profile_version", "config_version",
                                    "matcher_version", "ranking_id", "created_at", "started_at", "finished_at")},
        "status": run["state"], "pool_count": len(rows), "ready_count": planned["score"] + planned["reuse"],
        "to_score_count": planned["score"], "reuse_count": planned["reuse"], "excluded_counts": dict(excluded),
        "scored_count": counts["scored"], "reused_count": counts["reused"], "error_count": counts["failed"],
        "skipped_count": counts["skipped"], "error": json.loads(run["error_json"] or "null"),
    }


def still_eligible(db, item) -> bool:
    job_id, candidate_id, attempt_id = (item[key] for key in ("job_id", "candidate_id", "attempt_id"))
    if candidate_row(db, job_id, candidate_id)["current_attempt_id"] != attempt_id:
        return False
    attempt = attempt_row(db, job_id, candidate_id, attempt_id)
    status, _, disposition = eligibility(db, attempt)
    actual = disposition["disposition_id"] if disposition and attempt["requires_review"] else None
    return status == "ready_to_score" and actual == item["clearance_id"]


def persist_candidate(task, item, result=None, links=None, failure=None) -> None:
    with task.store.transaction(write=True) as db:
        task.current(db)
        job_id, run_id, candidate_id = (item[key] for key in ("job_id", "run_id", "candidate_id"))
        status = "failed" if failure else "scored"
        score_id = None
        if not still_eligible(db, item):
            status, failure = "skipped", "input_changed"
        elif result is not None:
            score_id = new_id()
            db.execute("INSERT INTO candidate_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (job_id, score_id, run_id, candidate_id, item["attempt_id"], item["basis_id"],
                        result.fit_score, result.tier, result.mandatory_coverage, None, encode(links),
                        encode({"schema_version": 1, "unassessed": result.unassessed,
                                "explanations": [exp.model_dump(mode="json") for exp in result.explanations]}), utc_now()))
        db.execute(
            """UPDATE scoring_run_candidates SET state = ?, score_id = ?, error_code = ?
            WHERE (job_id, run_id, candidate_id) = (?, ?, ?)""", (status, score_id, failure, job_id, run_id, candidate_id),
        )


def process_run(task) -> None:
    with task.store.transaction() as db:
        run = task.current(db)
        job_id, run_id = run["job_id"], run["run_id"]
        basis = db.execute("SELECT * FROM scoring_bases WHERE (job_id, basis_id) = (?, ?)",
                           (job_id, run["basis_id"])).fetchone()
        profile = scoring_profile(db, job_id, basis["profile_version"])
    settings = json.loads(basis["settings_json"])
    config = ScoringConfig.model_validate(settings["config"])
    matcher = LLMMatcher(settings["model"])
    if matcher.version != basis["matcher_version"] or basis["scorer_version"] != SCORER_VERSION:
        raise WorkflowError(409, "SCORING_BASIS_UNAVAILABLE", "Start a run with the current scoring configuration.")
    valid_options = {option.option_id for criterion in profile.active_requirements for option in criterion.criteria}
    while True:
        with task.store.transaction() as db:
            task.current(db)
            item = db.execute(
                """SELECT * FROM scoring_run_candidates WHERE (job_id, run_id) = (?, ?) AND state = 'pending'
                ORDER BY candidate_id LIMIT 1""", (job_id, run_id),
            ).fetchone()
            if item is None:
                break
            if still_eligible(db, item):
                payload = db.execute(
                    "SELECT profile_json FROM verified_profiles WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?)",
                    (job_id, item["candidate_id"], item["attempt_id"]),
                ).fetchone()[0]
            else:
                payload = None
        if payload is None:
            persist_candidate(task, item, failure="input_changed")
            LOGGER.info("Candidate scoring skipped: %s candidate_id=%s reason=input_changed",
                        task.log_context, item["candidate_id"])
            continue
        phase = "load_verified_profile"
        try:
            verified = VerifiedProfile.model_validate_json(payload)
            if verified.candidate_id != item["candidate_id"]:
                raise ValueError("Profile identity mismatch")
            phase = "matching"
            with model_log_context(f"{task.log_context} candidate_id={item['candidate_id']}"):
                matches = CandidateMatches.model_validate(matcher.match(verified.matcher_view(), profile))
            if matches.matcher_version != basis["matcher_version"]:
                raise ValueError("Matcher version mismatch")
            phase = "deterministic_scoring"
            result = score_candidate(verified, profile, matches, config)
            if result.status != "ok":
                raise ValueError("Scorer rejected the candidate result")
            fact_ids = {fact.fact_id for fact in verified.facts}
            links = [
                link.model_dump(mode="json", exclude={"rationale"}) for link in matches.matches
                if link.option_id in valid_options and link.fact_id in fact_ids
            ]
        except Exception as error:
            LOGGER.warning("Candidate scoring failed: %s candidate_id=%s phase=%s error=%s at=%s",
                           task.log_context, item["candidate_id"], phase, type(error).__name__, exception_site(error))
            persist_candidate(task, item, failure="scoring_failed")
        else:
            persist_candidate(task, item, result, links)
    publish_run(task, job_id, run_id, basis["fingerprint"])


def publish_run(task, job_id, run_id, fingerprint) -> None:
    with task.store.transaction(write=True) as db:
        task.current(db)
        for row in db.execute("""SELECT * FROM scoring_run_candidates WHERE (job_id, run_id) = (?, ?)
                              AND state IN ('scored','reused')""", (job_id, run_id)).fetchall():
            if not still_eligible(db, row):
                db.execute(
                    """UPDATE scoring_run_candidates SET state = 'skipped', score_id = NULL, error_code = 'input_changed'
                    WHERE (job_id, run_id, candidate_id) = (?, ?, ?)""", (job_id, run_id, row["candidate_id"]),
                )
        prior = latest_ranking(db, job_id)
        ranking_id, now = new_id(), utc_now()
        db.execute("INSERT INTO rankings VALUES (?, ?, ?, ?)", (job_id, ranking_id, run_id, now))
        ordered = db.execute(
            """SELECT c.candidate_id, c.state, c.excluded_reason, c.error_code
            FROM scoring_run_candidates c LEFT JOIN candidate_scores s USING (job_id, score_id)
            WHERE (c.job_id, c.run_id) = (?, ?)
            ORDER BY s.fit_score DESC, coalesce(s.mandatory_coverage, 0) DESC, c.candidate_id""", (job_id, run_id),
        )
        rank, had_errors = 0, False
        for row in ordered:
            ranked = row["state"] in {"scored", "reused"}
            rank += int(ranked)
            had_errors |= row["state"] == "failed"
            db.execute("INSERT INTO ranking_rows VALUES (?, ?, ?, ?, ?, ?)",
                       (job_id, ranking_id, row["candidate_id"], run_id, rank if ranked else None,
                        None if ranked else (row["excluded_reason"] or row["error_code"])))
        state = "completed_with_errors" if had_errors else "completed"
        task.finish(db, state=state, finished_at=now)
        job = get_job(db, job_id)
        active = scoring_basis(job["active_profile_version"])
        is_current = active is not None and active["fingerprint"] == fingerprint
        if is_current:
            db.execute("UPDATE shortlist_proposals SET state = 'superseded' WHERE job_id = ? AND state = 'draft'", (job_id,))
            db.execute("UPDATE jobs SET current_ranking_id = ? WHERE job_id = ?", (ranking_id, job_id))
            create_proposal(db, job_id, ranking_id, SYSTEM_ACTOR, previous_ranking_id=prior["ranking_id"] if prior else None)
        audit(db, job_id, SYSTEM_ACTOR, "ranking_published", {"run_id": run_id, "ranking_id": ranking_id},
              {"ranked_count": rank})
    LOGGER.info("Ranking published: %s ranking_id=%s state=%s ranked=%d current=%s",
                task.log_context, ranking_id, state, rank, is_current)


def ranking_view(db, job_id, ranking_id) -> dict:
    row = require(db.execute(
        """SELECT k.*, r.started_at, r.finished_at, r.requested_by, b.profile_version, b.matcher_version, b.config_version,
        j.profile_id, d.role_title FROM rankings k JOIN scoring_runs r USING (job_id, run_id)
        JOIN scoring_bases b USING (job_id, basis_id) JOIN jobs j USING (job_id)
        JOIN requirement_profiles p ON (p.job_id,p.version) = (b.job_id,b.profile_version)
        JOIN requirement_draft_revisions d USING (job_id, draft_id, revision_no)
        WHERE (k.job_id,k.ranking_id) = (?, ?)""", (job_id, ranking_id),
    ).fetchone())
    records = db.execute(
        """SELECT k.rank, k.unranked_reason, json_array_length(c.finding_summary_json) AS flag_count
        FROM ranking_rows k JOIN scoring_run_candidates c USING (job_id, run_id, candidate_id)
        WHERE (k.job_id, k.ranking_id) = (?, ?)""", (job_id, ranking_id),
    ).fetchall()
    unranked = Counter(record["unranked_reason"] for record in records if record["rank"] is None)
    return {
        **dict(row), "ranked_count": len(records) - sum(unranked.values()), "unranked_count": sum(unranked.values()),
        "unranked_counts": dict(unranked), "security_finding_count": sum(record["flag_count"] for record in records),
        "flagged_candidate_count": sum(record["flag_count"] > 0 for record in records),
    }


def ranking_rows(db, job_id, ranking_id, *, status=None, candidate_id=None):
    require(db.execute("SELECT 1 FROM rankings WHERE (job_id, ranking_id) = (?, ?)", (job_id, ranking_id)).fetchone())
    query = """SELECT k.candidate_id, k.rank, k.unranked_reason, c.attempt_id, c.captured_risk, c.finding_summary_json,
        s.fit_score, s.tier, s.mandatory_coverage, s.explanations_json, s.counterfactual_delta
        FROM ranking_rows k JOIN scoring_run_candidates c USING (job_id, run_id, candidate_id)
        LEFT JOIN candidate_scores s ON (c.job_id,c.score_id) = (s.job_id,s.score_id) AND k.rank IS NOT NULL
        WHERE (k.job_id,k.ranking_id) = (?, ?)"""
    params = [job_id, ranking_id]
    if status is not None:
        query += " AND k.rank IS NOT NULL" if status == "ranked" else " AND k.rank IS NULL"
    if candidate_id is not None:
        query += " AND k.candidate_id = ?"
        params.append(candidate_id)
    return db.execute(query + " ORDER BY k.rank IS NULL, k.rank, k.candidate_id", params)


def ranking_candidate(row, prefix, *, detail=False) -> dict:
    result = {
        "candidate_id": row["candidate_id"], "attempt_id": row["attempt_id"], "overall_risk": row["captured_risk"],
        "security_flags": json.loads(row["finding_summary_json"]),
        "candidate_href": f"{prefix}/candidates/{row['candidate_id']}",
    }
    if row["rank"] is None:
        result.update(status="unranked", reason_code=row["unranked_reason"])
    else:
        evidence = json.loads(row["explanations_json"])
        result.update(status="ranked", rank=row["rank"], fit_score=row["fit_score"], tier=row["tier"],
                      unassessed=evidence["unassessed"])
        if detail:
            result["mandatory_coverage"] = row["mandatory_coverage"]
            for explanation in evidence["explanations"]:
                explanation["assessment"] = ("not_assessed" if explanation["missing"] else
                                             "met" if explanation["met"] else "not_met")
                if explanation["missing"]:
                    explanation["credit"] = None
            result["assessments"] = evidence["explanations"]
            if row["counterfactual_delta"] is not None:
                result["counterfactual_delta"] = row["counterfactual_delta"]
    return result


def add_evidence_locations(db, job_id, candidate: dict) -> None:
    if candidate["status"] != "ranked":
        return
    spans = {
        row["span_id"]: (row["page"], json.loads(row["location_json"] or "null"))
        for row in db.execute(
            """SELECT span_id, page, location_json FROM intake_spans
            WHERE (job_id, candidate_id, attempt_id) = (?, ?, ?) AND excluded_from_evidence = 0""",
            (job_id, candidate["candidate_id"], candidate["attempt_id"]),
        )
    }
    for requirement in candidate["assessments"]:
        criteria = requirement["criteria"]
        if requirement["deciding"] is not None:
            criteria = [*criteria, requirement["deciding"]]
        for criterion in criteria:
            duration = criterion.get("duration")
            citations = [criterion, *duration["spans"]] if duration else [criterion]
            for citation in citations:
                span_id = citation["location"]
                if span_id in spans:
                    page, source_location = spans[span_id]
                    citation.update(page=page, source_location=source_location)
