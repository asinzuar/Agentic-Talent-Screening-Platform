from fastapi.testclient import TestClient

from agentic_talent_screener.api.app import create_app


def test_full_job_workspace_flow_with_sample_pool() -> None:
    client = TestClient(create_app())

    # 1. Create job (sample mode = offline, deterministic)
    resp = client.post("/api/v1/jobs", json={"jd_text": "Senior ML Engineer JD", "use_sample": True})
    assert resp.status_code == 200
    job = resp.json()
    job_id = job["job_id"]
    assert job["resume_uploads_unlocked"] is False
    assert job["draft_profile"]["approved"] is False

    # 2. Resume loading is blocked before approval
    blocked = client.post(f"/api/v1/jobs/{job_id}/resumes/load-sample")
    assert blocked.status_code == 409

    # 3. Approve & lock v1
    approved = client.post(f"/api/v1/jobs/{job_id}/requirements/approve", json={"actor": "hm-01"})
    assert approved.status_code == 200
    assert approved.json()["resume_uploads_unlocked"] is True
    assert approved.json()["active_profile"]["version"] == 1

    # 4. Load sample resumes (flagged ones go to review_required)
    loaded = client.post(f"/api/v1/jobs/{job_id}/resumes/load-sample")
    assert loaded.status_code == 200
    counts = loaded.json()["resume_counts"]
    assert counts["ready_to_score"] >= 1
    assert counts["review_required"] >= 1

    # 5. Clear one held resume for scoring
    held = [r for r in loaded.json()["resumes"] if r["status"] == "review_required"][0]
    cleared = client.post(
        f"/api/v1/jobs/{job_id}/resumes/{held['candidate_id']}/review",
        json={"decision": "clear_for_scoring", "actor": "hr-01", "reason": "Reviewed spans; safe to score"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["resume"]["status"] == "ready_to_score"

    # 6. Score ready resumes
    scored = client.post(f"/api/v1/jobs/{job_id}/score", json={"shortlist_size": 2, "actor": "hr-01"})
    assert scored.status_code == 200
    assert scored.json()["ranking"]["n_scored"] >= 2

    # 7. Approve shortlist with an override
    results = scored.json()["ranking"]["results"]
    last_cid = results[-1]["candidate_id"]
    sl = client.post(
        f"/api/v1/jobs/{job_id}/shortlist/approve",
        json={
            "actor": "hm-01",
            "shortlist_size": 2,
            "overrides": [{"candidate_id": last_cid, "action": "include", "reason": "Strong domain fit"}],
        },
    )
    assert sl.status_code == 200
    assert last_cid in sl.json()["snapshot"]["candidates"]
