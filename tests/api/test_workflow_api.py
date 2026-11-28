import logging
import sys
from threading import Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentic_talent_screener import job_analysis, llm
from agentic_talent_screener.api import create_app
from agentic_talent_screener.api.config import Settings
from agentic_talent_screener.ranking.config import ScoringConfig
from agentic_talent_screener.ranking.schemas import CandidateMatches, RequirementProfile
from agentic_talent_screener.resume_intake.pipeline import PipelineResult
from agentic_talent_screener.workflow import intake, requirements, scoring, worker
from agentic_talent_screener.workflow.storage import Store
from agentic_talent_screener.workflow.worker import LeaseLost, claim, run_once

JOBS = "/api/v1/jobs"


def post(client, path, status=202, **kwargs):
    headers = {"Idempotency-Key": uuid4().hex, **kwargs.pop("headers", {})}
    result = client.post(path, headers=headers, **kwargs)
    assert result.status_code == status, result.text
    return result


def get(client, path):
    result = client.get(path)
    assert result.status_code == 200, result.text
    return result.json()


def save(client, path, data):
    result = client.put(path, json=data)
    assert result.status_code == 200, result.text
    return result.json()


@pytest.fixture
def clients(tmp_path, monkeypatch):
    settings = Settings(
        workflow_db=tmp_path / "workflow.sqlite3",
        log_dir=tmp_path / "logs",
        workflow_tokens={"hr": "fixture-token", "other": "fixture-other"},
    )
    client = TestClient(create_app(settings), headers={"Authorization": "Bearer fixture-token"})
    store = Store(settings.workflow_db)

    def draft(text, *, profile_id, reasoning):
        assert "Python" in text
        assert reasoning is False
        return RequirementProfile.model_validate({
            "profile_id": profile_id, "version": 1, "role_title": "Python engineer", "requirements": [{
                "requirement_id": "r1", "statement": "Python", "kind": "skill",
                "options": [{"option_id": "r1.o1", "text": "Python"}], "priority": "M", "weight": 5,
                "provenance": "jd_qualification", "quote": "Python", "flags": ["Check scope"],
            }, {
                "requirement_id": "r2", "statement": "Testing", "kind": "skill",
                "options": [{"option_id": "r2.o1", "text": "Testing"}], "priority": "P", "weight": 2,
                "provenance": "model_suggestion", "rationale": "Testing helps maintain the service.",
            }],
        })

    def normalize(path, *, model, candidate_id, show_progress, on_stage, on_forensics):
        case = path.read_text()
        on_stage("document_safety")
        if case == "invalid":
            on_forensics({"issues": [{"code": "UNSUPPORTED_FILE_TYPE", "severity": "high"}]})
            return PipelineResult(status="quarantined", stopped_at="L0_DOC_FORENSICS", resume_sha256="x")
        excluded = "Never use this excluded span as evidence"
        if case in ("short_duration", "enough_duration"):
            amount = "5+ years" if case == "short_duration" else "12+ years"
            duration_quotes = (amount, "of experience in", "Python")
            duration_spans = [
                {"span_id": f"p1:s{i:04d}", "quote": quote}
                for i, quote in enumerate(duration_quotes, start=1)
            ]
            on_forensics({"details": {"spans": [
                {"span_id": span["span_id"], "page": 1, "visibility": "visible", "text": span["quote"]}
                for span in duration_spans
            ]}})
            return PipelineResult.model_validate({
                "status": "normalized", "stopped_at": "L3_EVIDENCE_VERIFIER", "resume_sha256": "x",
                "profile": {
                    "blind_candidate_id": candidate_id, "extraction_confidence": 1,
                    "facts": [{"fact_id": "f1", "kind": "experience", "statement": " ".join(duration_quotes),
                               "evidence_level": "E3", "source_spans": duration_spans}],
                },
            })
        on_forensics({
            "details": {"spans": [
                {"span_id": "p1:s0001", "page": 1, "visibility": "visible", "text": "Built Python tools"},
                {"span_id": "p1:s0002", "page": 1, "visibility": "hidden" if case == "hold" else "visible", "text": excluded},
            ]},
            "issues": [{"code": "PDF_NONVISIBLE_TEXT", "severity": "medium"}] if case == "hold" else [],
        })
        for stage in ("injection_detection", "extraction", "evidence_verification"):
            on_stage(stage)
        return PipelineResult.model_validate({
            "status": "normalized_with_flag" if case == "hold" else "normalized",
            "stopped_at": "L3_EVIDENCE_VERIFIER", "resume_sha256": "x",
            "requires_human_review": case == "hold",
            "sanitised_span_ids": ["p1:s0002"] if case == "hold" else [],
            "profile": {
                "blind_candidate_id": candidate_id, "extraction_confidence": 1,
                "facts": [{"fact_id": "f1", "kind": "skill", "statement": "Python", "evidence_level": "E4" if case == "hold" else "E3",
                           "source_spans": [{"span_id": "p1:s0001", "quote": "Built Python tools"}]}],
            },
        })

    class Matcher:
        def __init__(self, model):
            self.version = f"fixture-matcher:{model}"

        def match(self, candidate, profile):
            assert "quote" not in candidate.facts[0].model_dump()
            if candidate.candidate_id == "C-0003":
                raise RuntimeError("Provider diagnostic must stay private")
            return CandidateMatches.model_validate({
                "candidate_id": candidate.candidate_id, "profile_id": profile.profile_id,
                "profile_version": profile.version, "matcher_version": self.version,
                "matches": [{"option_id": "r1.o1", "fact_id": "f1", "relation": "equivalent"}],
            })

    monkeypatch.setattr(requirements, "draft_requirement_profile", draft)
    monkeypatch.setattr(intake, "normalize_resume", normalize)
    monkeypatch.setattr(scoring, "LLMMatcher", Matcher)
    return client, store, settings


def create(client, store):
    files = {"file": ("job.txt", b"Python developer")}
    headers = {"Idempotency-Key": "new-job"}
    created = post(client, JOBS, files=files, headers=headers)
    assert post(client, JOBS, files=files, headers=headers).json() == created.json()
    post(client, JOBS, status=409, files={"file": ("job.txt", b"Another role")}, headers=headers)
    job = f"{JOBS}/{created.json()['job_id']}"
    assert get(client, created.headers["location"])["status"] == "queued"
    assert run_once(store)
    draft = get(client, f"{job}/requirement-draft")
    assert draft["status"] == "ready"
    assert get(client, f"{job}/job-descriptions/{draft['jd_id']}")["text"] == "Python developer"
    return job, draft


def approve(client, job, draft):
    return post(client, f"{job}/requirement-draft/approve", status=201,
                json={"expected_revision": draft["revision"]}).json()


def acknowledged(client, job, draft):
    flags = [flag for criterion in draft["requirements"] for flag in criterion["flags"]]
    return save(client, f"{job}/requirement-draft", {
        "expected_revision": draft["revision"], "role_title": draft["role_title"],
        "requirements": draft["requirements"], "flag_reviews": [
            {"flag_id": flag["flag_id"], "disposition": "acknowledged"} for flag in flags
        ],
    })


def upload(client, job, *cases, **fields):
    return post(client, f"{job}/resume-batches", data={"profile_version": 1, **fields},
                files=[("files", ("resume.txt", case.encode())) for case in cases])


def score(client, job):
    plan = get(client, f"{job}/score-plan")
    return post(client, f"{job}/scoring-runs", json={"plan_token": plan["plan_token"]}).headers["location"]


def test_workflow_preserves_ranking_and_shortlist_history(clients):
    client, store, settings = clients
    job, draft = create(client, store)
    post(client, f"{job}/resume-batches", status=409, data={"profile_version": 1},
         files={"files": ("resume.txt", b"clean")})
    post(client, f"{job}/requirement-draft/approve", status=422, json={"expected_revision": draft["revision"]})
    assert approve(client, job, acknowledged(client, job, draft))["version"] == 1

    batch = upload(client, job, "clean", "hold", "clean")
    assert [item["status"] for item in batch.json()["files"]] == ["processing"] * 3
    assert run_once(store)
    plan = get(client, f"{job}/score-plan")
    assert (plan["mode"], plan["to_score_count"], plan["excluded_counts"]) == ("full", 1, {"processing": 2})
    first_run = score(client, job)
    assert get(client, first_run)["status"] == "queued"
    assert all(run_once(store) for _ in range(3))
    first = get(client, job)
    rank_v1 = f"{job}/rankings/{first['current_ranking_id']}"
    assert get(client, rank_v1)["unranked_counts"] == {"processing": 2}

    candidate = f"{job}/candidates/C-0002"
    held = get(client, candidate)
    assert held["status"] == "review_required"
    attempt = f"{candidate}/intake-attempts/{held['attempt_id']}"
    packet = get(client, f"{attempt}/audit")
    assert packet["findings"][0]["spans"][0]["span_id"] == "p1:s0002"
    assert "excluded span" in get(client, packet["findings"][0]["spans"][0]["excerpt_href"])["excerpt"]
    post(client, f"{candidate}/review-dispositions", status=201, json={
        "expected_revision": held["revision"], "attempt_id": held["attempt_id"],
        "action": "clear_for_scoring", "reason": "Reviewed the retained evidence.",
    })
    assert get(client, candidate)["status"] == "ready_to_score"

    proposal = f"{job}/shortlist-proposals/{first['current_proposal_id']}"
    saved = save(client, proposal, {
        "expected_revision": get(client, proposal)["revision"], "size": 1,
        "overrides": [{"candidate_id": "C-0001", "action": "include", "reason": "Keep this verified applicant."}],
    })
    approval = post(client, f"{proposal}/approve", status=201, json={"expected_revision": saved["revision"]})
    approved_href = approval.headers["location"]
    assert approval.json()["approved_member_ids"] == ["C-0001"]
    assert get(client, f"{approved_href}/export?format=json")["ranking_id"] == first["current_ranking_id"]

    plan = get(client, f"{job}/score-plan")
    assert (plan["mode"], plan["to_score_count"], plan["reuse_count"]) == ("incremental", 2, 1)
    second_run = score(client, job)
    assert run_once(store)
    assert get(client, second_run)["status"] == "completed_with_errors"
    restarted = TestClient(create_app(settings), headers=client.headers)
    second = get(restarted, job)
    assert second["last_approved_shortlist_id"] == approval.json()["shortlist_id"]
    new_rank = f"{job}/rankings/{second['current_ranking_id']}"
    rows = get(restarted, f"{new_rank}/candidates")["items"]
    assert [row["candidate_id"] for row in rows] == ["C-0002", "C-0001", "C-0003"]
    assert rows[-1]["reason_code"] == "scoring_failed" and "fit_score" not in rows[-1]
    evidence = get(client, f"{new_rank}/candidates/C-0002")
    assert evidence["assessments"][0]["assessment"] == "met" and "counterfactual_delta" not in evidence
    assert evidence["assessments"][0]["deciding"]["page"] == 1
    assert get(client, rank_v1)["unranked_count"] == 2

    next_proposal = f"{job}/shortlist-proposals/{second['current_proposal_id']}"
    view = get(client, next_proposal)
    assert view["effective_member_ids"] == ["C-0002"]
    assert view["comparison"]["computed_top_n"]["entered"] == [
        {"candidate_id": "C-0002", "old_rank": None, "new_rank": 1}
    ]
    post(client, f"{next_proposal}/approve", status=409, json={"expected_revision": view["revision"]})
    view = save(client, next_proposal, {
        "expected_revision": view["revision"], "size": 1,
        "prior_override_decisions": [{"override_id": view["prior_overrides"][0]["override_id"], "decision": "drop"}],
    })
    approved = post(client, f"{next_proposal}/approve", status=201, json={"expected_revision": view["revision"]}).json()
    assert approved["supersedes_shortlist_id"] == approval.json()["shortlist_id"]
    assert approved["approved_member_ids"] == ["C-0002"]
    assert get(client, approved_href)["approved_member_ids"] == ["C-0001"]

    draft = post(client, f"{job}/requirement-draft", status=201, json={"base_version": 1}).json()
    assert get(client, job)["current_ranking_id"] == second["current_ranking_id"]
    assert approve(client, job, draft)["version"] == 2
    assert get(client, job)["ranking_state"] == "rescore_required"
    plan = get(client, f"{job}/score-plan")
    assert plan["mode"] == "full" and plan["reuse_count"] == 0
    assert get(client, new_rank)["profile_version"] == 1


@pytest.mark.parametrize("gate_floor", [0.5, 1.0])
def test_duration_changes_mandatory_coverage_and_ranking_evidence(clients, monkeypatch, gate_floor):
    # With a floor of 1 the scores tie and mandatory coverage still decides the order.
    monkeypatch.setattr(scoring, "DEFAULT_CONFIG", ScoringConfig(gate_floor=gate_floor))
    client, store, _ = clients
    job, draft = create(client, store)
    draft["requirements"][0].update(kind="experience", min_years=12, min_evidence=3)
    approve(client, job, acknowledged(client, job, draft))

    upload(client, job, "short_duration", "enough_duration")
    assert run_once(store) and run_once(store)
    run = score(client, job)
    assert run_once(store)
    ranking_id = get(client, run)["ranking_id"]
    path = f"{job}/rankings/{ranking_id}/candidates"
    short = get(client, f"{path}/C-0001")
    enough = get(client, f"{path}/C-0002")
    assert (short["rank"], enough["rank"]) == (2, 1)
    assert (short["mandatory_coverage"], enough["mandatory_coverage"]) == (0, 1)
    assert (short["fit_score"], enough["fit_score"]) == (0.75 * gate_floor, 0.75)
    assessment = short["assessments"][0]
    assert assessment["assessment"] == "not_met" and assessment["evidence_met"]
    assert assessment["gap"] == "12-year minimum not established"
    assert assessment["credit"] == enough["assessments"][0]["credit"] == 0.75
    duration = assessment["deciding"]["duration"]
    assert duration["reported"] == "5+ years" and duration["lower_bound_months"] == 60
    assert duration["upper_bound_months"] is None
    assert [(span["quote"], span["page"]) for span in duration["spans"]] == [
        ("5+ years", 1), ("of experience in", 1), ("Python", 1),
    ]


def test_quarantine_replacement_and_job_authorization(clients):
    client, store, settings = clients
    job, draft = create(client, store)
    approve(client, job, acknowledged(client, job, draft))
    other = TestClient(create_app(settings), headers={"Authorization": "Bearer fixture-other"})
    assert other.get(job).status_code == 404
    assert other.get(JOBS).json()["items"] == []
    assert other.get(job, headers={"Authorization": "Bearer invalid"}).status_code == 401
    batch = upload(client, job, "invalid")
    assert run_once(store)
    candidate = f"{job}/candidates/C-0001"
    row = get(client, candidate)
    assert row["status"] == "quarantined"
    body = {"expected_revision": row["revision"], "attempt_id": row["attempt_id"]}
    post(client, f"{candidate}/intake-retries", status=409, json=body)
    post(client, f"{candidate}/review-dispositions", status=409,
         json={**body, "action": "clear_for_scoring", "reason": "Attempted clearance."})
    replacement = post(client, f"{candidate}/replacement-requests", status=201, json=body).json()
    upload(client, job, "clean", replacement_request_id=replacement["request_id"])
    assert run_once(store)
    assert get(client, candidate)["status"] == "ready_to_score"
    assert get(client, batch.headers["location"])["files"][0]["status"] == "quarantined"
    post(client, f"{job}/resume-batches", status=409, data={
        "profile_version": 1, "replacement_request_id": replacement["request_id"],
    }, files={"files": ("resume.txt", b"clean")})


def test_expired_task_lease_is_recovered_and_fenced(clients):
    client, store, settings = clients
    created = post(client, JOBS, files={"file": ("job.txt", b"Python developer")})
    task, _ = claim(store)
    with store.transaction(write=True) as db:
        db.execute("UPDATE requirement_drafts SET lease_until = '2000-01-01' WHERE job_id = ?",
                   (created.json()["job_id"],))
    assert run_once(Store(settings.workflow_db))
    with pytest.raises(LeaseLost):
        task.update(stage="draft_requirements")
    assert not run_once(store)
    assert get(client, created.headers["location"])["status"] == "ready"


def test_worker_reports_progress_while_waiting_on_the_agent(clients, monkeypatch, caplog):
    client, _, settings = clients
    created = post(client, JOBS, files={"file": ("job.txt", b"Python developer")})
    seen = Event()
    draft = requirements.draft_requirement_profile
    log = worker.LOGGER.info
    progress_messages = []

    def record(message, *args):
        log(message, *args)
        formatted = message % args
        if formatted.startswith("Workflow task progress:") and "stage=draft_requirements" in formatted:
            progress_messages.append(formatted)
            if len(progress_messages) == 2:
                seen.set()

    def slow_draft(*args, **kwargs):
        assert seen.wait(timeout=3), "A model call should not block worker progress logs."
        return draft(*args, **kwargs)

    monkeypatch.setattr(worker, "PROGRESS_INTERVAL", 0.02)
    monkeypatch.setattr(worker.LOGGER, "info", record)
    monkeypatch.setattr(requirements, "draft_requirement_profile", slow_draft)
    monkeypatch.setattr(worker, "get_settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["worker", "--once"])
    with caplog.at_level(logging.INFO, logger=worker.LOGGER.name):
        worker.main()

    assert get(client, created.headers["location"])["status"] == "ready"
    assert all(f"job_id={created.json()['job_id']}" in message for message in progress_messages)
    assert "Workflow task finished:" in caplog.text
    assert "Python developer" not in caplog.text
    file_log = (settings.log_dir / "worker.log").read_text()
    assert all(message in file_log for message in progress_messages)
    assert "Workflow task finished:" in file_log
    assert "Python developer" not in file_log


def test_worker_file_logs_explain_model_failure_without_request_or_response_text(clients, monkeypatch, caplog):
    client, _, settings = clients
    created = post(client, JOBS, files={"file": ("job.txt", b"Python developer")})
    response = llm.requests.Response()
    response.status_code = 200
    monkeypatch.setattr(response, "json", lambda: {
        "choices": [{"message": {"content": None, "reasoning": "private model output"}, "finish_reason": "length"}],
    })
    monkeypatch.chdir(settings.workflow_db.parent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "private fixture credential")
    captured = []

    def post_model(*args, **kwargs):
        captured.append(kwargs["json"]["reasoning"])
        return response

    monkeypatch.setattr(llm.requests, "post", post_model)
    monkeypatch.setattr(requirements, "draft_requirement_profile", job_analysis.draft_requirement_profile)
    monkeypatch.setattr(worker, "get_settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["worker", "--once"])
    with caplog.at_level(logging.INFO):
        worker.main()

    failed = get(client, created.headers["location"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "PROCESSING_FAILED"
    assert captured == [{"enabled": False}]
    file_log = (settings.log_dir / "worker.log").read_text()
    assert "stage=draft_requirements" in file_log
    assert "purpose=job_analysis_draft" in file_log
    assert "reasoning_enabled=False" in file_log
    assert f"job_id={created.json()['job_id']}" in file_log
    assert "http_status=200" in file_log and "reason=missing_content" in file_log
    assert "finish_reason=length" in file_log and "error=RuntimeError at=llm.py:" in file_log
    for sensitive in ("Python developer", "private model output", "private fixture credential"):
        assert sensitive not in file_log and sensitive not in caplog.text


def test_failed_jd_can_be_replaced_and_stale_plan_requires_confirmation(clients):
    client, store, _ = clients
    oversized = client.post(JOBS, headers={"Content-Length": str(64 * 1024 * 1024)})
    assert oversized.status_code == 413
    assert get(client, JOBS)["items"] == []

    created = post(client, JOBS, files={"file": ("bad.txt", b" ")})
    assert run_once(store)
    draft_path = created.headers["location"]
    failed = get(client, draft_path)
    assert failed["status"] == "failed" and failed["error"]["code"] == "JD_UNREADABLE"
    post(client, f"{draft_path}/retry", status=409, data={"expected_revision": failed["revision"]})
    post(client, f"{draft_path}/retry", data={"expected_revision": failed["revision"]},
         files={"file": ("replacement.txt", b"Python engineer")})
    assert run_once(store)
    recovered = get(client, draft_path)
    assert recovered["jd_id"] != failed["jd_id"]
    job = draft_path.removesuffix("/requirement-draft")
    approve(client, job, acknowledged(client, job, recovered))
    upload(client, job, "clean", "clean")
    assert run_once(store)
    preview = get(client, f"{job}/score-plan")
    assert preview["to_score_count"] == 1
    assert run_once(store)
    stale = post(client, f"{job}/scoring-runs", status=409, json={"plan_token": preview["plan_token"]})
    assert stale.json()["code"] == "SCORE_PLAN_STALE"
    assert get(client, f"{job}/score-plan")["to_score_count"] == 2
