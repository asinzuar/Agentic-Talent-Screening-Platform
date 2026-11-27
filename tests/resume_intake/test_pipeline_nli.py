"""Pipeline integration with controlled L1/L2 and optional real local L3 inference."""
from __future__ import annotations

import os
import re

import pytest

from agentic_talent_screener import llm
from agentic_talent_screener.resume_intake import hooks, pipeline
from agentic_talent_screener.resume_intake.l1_injection import SemanticAssessment
from agentic_talent_screener.resume_intake.nli import DEFAULT_NLI_MODEL, NliUnavailable


@pytest.fixture
def controlled_resume(tmp_path, monkeypatch):
    """Keep the actual L0, redaction, schema validation, and L3 pipeline path."""
    resume = tmp_path / "synthetic_resume.txt"
    resume.write_text(
        "Synthetic Candidate\nsynthetic.contact@example.test\nEXPERIENCE\n"
        "Built a data pipeline using Python.\n"
        "The Atlas project reduced latency by exactly 40 percent.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hooks, "assess_semantic_injection_risk",
        lambda *_args, **_kwargs: SemanticAssessment(risk="low", decision="allow", findings=[]),
    )

    def extract(messages, _model):
        text = "\n".join(message["content"] for message in messages)
        spans = re.findall(r"\[[^\]:]+:SPAN:(p\d+:s\d{4})\] ([^\n]*)", text)
        python_span = next(sid for sid, body in spans if "Built a data pipeline using Python." in body)
        metric_span = next(sid for sid, body in spans if "reduced latency by exactly 40 percent" in body)
        return {
            "identity_envelope": {},
            "proposed_claims": [
                {
                    "claim_id": "c001", "kind": "experience",
                    "value": "Used Python to build a data pipeline.",
                    "source_span_ids": [python_span],
                },
                {
                    "claim_id": "c002", "kind": "project",
                    "value": "The Atlas project reduced latency by exactly 80 percent.",
                    "source_span_ids": [metric_span],
                },
            ],
            "extraction_quality": {
                "confidence": 0.9, "missing_fields": [], "needs_human_review": False,
            },
        }

    monkeypatch.setattr(pipeline, "call_model", extract)
    return resume


def assert_verified_profile(result, model):
    assert result.status == "normalized"
    assert result.profile is not None
    assert [fact.fact_id for fact in result.profile.facts] == ["c001"]
    assert result.profile.facts[0].source_spans[0].quote == "Built a data pipeline using Python."
    events = {event.claim_id: event for event in result.profile.verification_events}
    assert events["c001"].detail == "nli: entailed (nli_semantic)"
    assert events["c001"].claim_summary == "Used Python to build a data pipeline."
    assert events["c001"].source_span_ids == [result.profile.facts[0].source_spans[0].span_id]
    assert result.profile.facts[0].verification_method == "nli_semantic"
    assert events["c002"].code == "CLAIM_NOT_ENTAILED"
    assert f"L3_NLI_MODEL:{model}" in result.events
    detail = next(item for item in result.event_details if item.code == "L3_NLI_VERIFICATION_COMPLETED")
    assert model in detail.message


def test_pipeline_routes_l2_claims_through_the_selected_nli_model(controlled_resume, monkeypatch):
    class ControlledVerifier:
        def __init__(self, model):
            self.model = model

        def verify(self, claim, evidence):
            assert "synthetic.contact@example.test" not in evidence
            return "contradicted" if "80 percent" in claim else "entailed"

    monkeypatch.setattr(pipeline, "OllamaVerifier", ControlledVerifier)
    monkeypatch.setattr(llm.requests, "post", lambda *_args, **_kwargs: pytest.fail("Unexpected network request"))
    result = pipeline.normalize_resume(
        controlled_resume, model="controlled-hosted-agents",
        l3_verifier="ollama", l3_model="trained-test-model", show_progress=False,
    )
    assert_verified_profile(result, "trained-test-model")


def test_pipeline_defaults_to_the_local_qwen_verifier(controlled_resume, monkeypatch):
    class ControlledVerifier:
        def __init__(self, model):
            self.model = model

        def verify(self, claim, evidence):
            return "contradicted" if "80 percent" in claim else "entailed"

    monkeypatch.setattr(pipeline, "OllamaVerifier", ControlledVerifier)
    result = pipeline.normalize_resume(controlled_resume, model="controlled-hosted-agents", show_progress=False)
    assert_verified_profile(result, DEFAULT_NLI_MODEL)


def test_pipeline_routes_nli_outage_to_literal_fallback_then_manual_review_when_no_facts_survive(controlled_resume):
    class UnavailableVerifier:
        model = "unavailable-test-model"

        def verify(self, claim, evidence):
            raise NliUnavailable("L3_NLI_UNAVAILABLE: model missing or unreachable.")

    result = pipeline.normalize_resume(
        controlled_resume, model="controlled-hosted-agents",
        support_verifier=UnavailableVerifier(), show_progress=False,
    )
    assert result.status == "manual_review"
    assert result.stopped_at == "L3_EVIDENCE_VERIFIER"
    assert result.requires_human_review is True
    assert result.profile is None
    assert "L2_L3_TARGETED_RETRY" not in result.events
    assert "L3_NLI_MODEL:unavailable-test-model" in result.events
    detail = next(item for item in result.event_details if item.code == "L3_NLI_LITERAL_FALLBACK")
    assert detail.severity == "action_required"
    assert "literal support" in detail.message


def test_pipeline_preserves_qwen_facts_then_flags_literal_fallback(controlled_resume):
    class FailsAfterFirstVerifier:
        model = "fails-after-first-test-model"

        def __init__(self):
            self.calls = 0

        def verify(self, claim, evidence):
            self.calls += 1
            if self.calls == 1:
                return "entailed"
            raise NliUnavailable("L3_NLI_UNAVAILABLE: test outage")

    result = pipeline.normalize_resume(
        controlled_resume, model="controlled-hosted-agents",
        support_verifier=FailsAfterFirstVerifier(), show_progress=False,
    )

    assert result.status == "normalized_with_flag"
    assert result.requires_human_review is True
    assert result.profile is not None
    assert [(fact.fact_id, fact.verification_method) for fact in result.profile.facts] == [
        ("c001", "nli_semantic"),
    ]
    assert result.profile.l3_fallback_used is True
    assert "L3_NLI_LITERAL_FALLBACK" in result.events
    assert "L2_L3_TARGETED_RETRY" not in result.events


@pytest.mark.skipif(
    os.environ.get("ATS_RUN_L3_LIVE") != "1",
    reason="Set ATS_RUN_L3_LIVE=1 to call the real local Ollama model.",
)
def test_pipeline_with_real_trained_l3_model(controlled_resume, monkeypatch):
    model = os.environ.get("ATS_L3_LIVE_MODEL", DEFAULT_NLI_MODEL)
    real_post = llm.requests.post
    calls = []

    def guarded_post(url, **kwargs):
        assert url.endswith("/api/chat"), "This test must not make paid/hosted model requests."
        payload = kwargs["json"]
        assert payload["model"] == model
        assert "synthetic.contact@example.test" not in str(payload["messages"])
        calls.append(url)
        return real_post(url, **kwargs)

    monkeypatch.setattr(llm.requests, "post", guarded_post)
    result = pipeline.normalize_resume(
        controlled_resume, model="controlled-hosted-agents",
        l3_verifier="ollama", l3_model=model, show_progress=False,
    )
    assert calls
    assert_verified_profile(result, model)
    print(f"Pipeline model: {model}; status: {result.status}; retained c001; dropped unsupported c002.")
