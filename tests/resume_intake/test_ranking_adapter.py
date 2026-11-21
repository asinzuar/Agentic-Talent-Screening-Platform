from __future__ import annotations

from agentic_talent_screener.resume_intake.l1_injection import SecurityAssessment
from agentic_talent_screener.resume_intake.l3_evidence_verifier import VerifiedCandidateProfile, VerifiedFact
from agentic_talent_screener.resume_intake.ranking_adapter import to_ranking_profile


def test_only_allowlisted_verified_skill_facts_cross_to_ranking() -> None:
    intake = VerifiedCandidateProfile(
        blind_candidate_id="C-0001",
        extraction_confidence=0.9,
        facts=[
            VerifiedFact(
                claim_id="c1", kind="skill", value="Python", source_span_ids=["p1:s0001"],
                evidence_level="E1",
            ),
            VerifiedFact(
                claim_id="c2", kind="education", value="Bachelor degree", source_span_ids=["p1:s0002"],
                evidence_level="E2",
            ),
        ],
    )
    security = SecurityAssessment(
        injection_risk_score=0, risk="low", decision="allow", findings=[]
    )

    handoff = to_ranking_profile(intake, security, {"python": "esco:python"})

    assert handoff.profile.candidate_id == "C-0001"
    assert handoff.profile.claims[0].skill_id == "esco:python"
    assert handoff.profile.claims[0].spans[0].location == "p1:s0001"
    assert handoff.profile.roles is None
    assert handoff.dropped_fact_ids == ("c2",)
