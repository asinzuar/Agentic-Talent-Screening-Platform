from agentic_talent_screener.resume_intake.claim_coverage import assess_coverage, build_coverage_anchors
from agentic_talent_screener.resume_intake.contracts import SourceSpan
from agentic_talent_screener.resume_intake.l3_evidence_verifier import (
    VerifiedCandidateProfile,
    VerifiedFact,
    VerifiedSpan,
)


def _span(span_id: str, text: str, section: str) -> SourceSpan:
    return SourceSpan(
        span_id=span_id, text=text, normalised_text=text, visible_text=text,
        section=section, visibility="visible",
    )


def _fact(kind: str, span_id: str) -> VerifiedFact:
    return VerifiedFact(
        fact_id="c001", kind=kind, statement="Senior SDET, 2020 - 2024" if kind == "experience" else "B.Tech Computer Science",
        source_spans=[VerifiedSpan(span_id=span_id, quote="source", section=kind)],
        dates=[], evidence_level="E2",
    )


def test_coverage_requires_role_and_degree_anchors_to_reach_verified_facts():
    anchors = build_coverage_anchors((
        _span("p1:s0001", "Senior SDET | 2020 - 2024", "experience"),
        _span("p1:s0002", "B.Tech Computer Science, 2016", "education"),
    ))
    assert [(item.anchor_id, item.kind, item.span_id) for item in anchors] == [
        ("experience-001", "experience", "p1:s0001"),
        ("education-001", "education", "p1:s0002"),
    ]

    only_degree = VerifiedCandidateProfile(
        blind_candidate_id="C-0001", facts=[_fact("education", "p1:s0002")],
        extraction_confidence=0.9,
    )
    report = assess_coverage(anchors, only_degree)
    assert report.missing_anchor_ids == ("experience-001",)
    assert report.complete is False


def test_coverage_does_not_treat_an_uncited_or_wrong_kind_fact_as_coverage():
    anchors = build_coverage_anchors((_span("p1:s0001", "Senior SDET | 2020 - 2024", "experience"),))
    wrong_kind = VerifiedCandidateProfile(
        blind_candidate_id="C-0001", facts=[_fact("education", "p1:s0001")],
        extraction_confidence=0.9,
    )
    assert assess_coverage(anchors, wrong_kind).missing_anchor_ids == ("experience-001",)


def test_coverage_detects_a_role_when_dates_are_in_a_separate_or_missing_span():
    anchors = build_coverage_anchors((
        _span("p1:s0001", "Software Engineer", "work experience"),
        _span("p1:s0002", "2020 - 2024", "work experience"),
    ))
    assert [(item.kind, item.span_id) for item in anchors] == [("experience", "p1:s0001")]
