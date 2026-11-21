import pytest
from fastapi.testclient import TestClient

from agentic_talent_screener.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _rank(client, **body):
    res = client.post("/api/v1/rank", json=body)
    assert res.status_code == 200, res.text
    return {r["candidate_id"]: r for r in res.json()["results"]}


def test_api_rank_l4_toggle_changes_injected_candidate(client):
    on = _rank(client, exclude_flagged=True)
    off = _rank(client, exclude_flagged=False)
    assert off["C-0417"]["fit_score"] > on["C-0417"]["fit_score"] + 0.3
    assert off["C-0417"]["rank"] < on["C-0417"]["rank"]
    assert on["C-0417"]["counterfactual_delta"] > 0.1


def test_api_rank_shortlist(client):
    res = client.post("/api/v1/rank", json={"shortlist_size": 2})
    assert res.status_code == 200
    body = res.json()
    assert body["shortlist_size"] == 2
    assert [r["candidate_id"] for r in body["results"] if r["shortlisted"]] == ["C-0932", "C-0101"]
    assert client.post("/api/v1/rank", json={"shortlist_size": 0}).status_code == 422
    audit = client.get("/api/v1/audit").json()["events"][-1]
    assert audit["details"]["shortlist_size"] == 2


def test_api_rank_rejects_unknown_fields(client):
    res = client.post("/api/v1/rank", json={"weights": {"skill_match": 0.9}})
    assert res.status_code == 422


def test_api_override_and_audit(client):
    ok = client.post("/api/v1/override", json={
        "candidate_id": "C-0417", "actor": "hr-01", "action": "exclude", "reason": "confirmed injection"})
    assert ok.status_code == 200 and ok.json()["event_type"] == "hr_override"
    for reason in ("", "   "):
        bad = client.post("/api/v1/override", json={
            "candidate_id": "C-0417", "actor": "hr-01", "action": "exclude", "reason": reason})
        assert bad.status_code in (400, 422)
    _rank(client)
    audit = client.get("/api/v1/audit").json()
    assert audit["chain_valid"] is True
    assert [e["event_type"] for e in audit["events"]] == [
        "profile_approved", "hr_override", "pool_scored"]


def test_api_override_unknown_candidate_404(client):
    res = client.post("/api/v1/override", json={
        "candidate_id": "C-9999", "actor": "hr-01", "action": "exclude", "reason": "x"})
    assert res.status_code == 404


def test_api_benchmark_get_is_not_logged_post_is(client):
    assert client.get("/api/v1/benchmark").status_code == 200
    assert client.post("/api/v1/benchmark", json={}).status_code == 200
    events = [e["event_type"] for e in client.get("/api/v1/audit").json()["events"]]
    assert events == ["profile_approved", "benchmark_run"]


def test_api_audit_export_is_jsonl(client):
    res = client.get("/api/v1/audit/export")
    assert res.status_code == 200
    assert "attachment" in res.headers["content-disposition"]
    assert res.text.count("\n") == 1  # seeded approval event only


def test_ledger_is_per_app():
    a, b = TestClient(create_app()), TestClient(create_app())
    _rank(a)
    assert len(b.get("/api/v1/audit").json()["events"]) == 1


def test_api_sample_and_index(client):
    s = client.get("/api/v1/sample").json()
    assert set(s) == {"config", "profile", "ranking"}
    index = client.get("/")
    assert index.status_code == 200 and "text/html" in index.headers["content-type"]
