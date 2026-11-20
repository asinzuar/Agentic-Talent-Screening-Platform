"""Deterministic L3 verification of L2's span-cited proposed claims."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .l2_resume_agent import AgentOutput, ProposedClaim


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationEvent(StrictModel):
    claim_id: str
    code: Literal[
        "SPAN_NOT_FOUND",
        "SPAN_NOT_VISIBLE",
        "SPAN_EXCLUDED",
        "CLAIM_NOT_ENTAILED",
        "CLAIM_VERIFIED",
    ]
    detail: str


class VerifiedFact(StrictModel):
    claim_id: str
    kind: Literal["skill", "certification", "experience", "education", "project"]
    value: str
    source_span_ids: list[str] = Field(min_length=1)
    source_sections: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    evidence_level: Literal["E1", "E2", "E3", "E4"]


class VerifiedCandidateProfile(StrictModel):
    """PII-blind facts that are the only L2-derived data allowed downstream."""

    schema_version: str = "1.0"
    blind_candidate_id: str
    facts: list[VerifiedFact] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    verification_events: list[VerificationEvent] = Field(default_factory=list)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _entails(claim: ProposedClaim, source_text: str) -> bool:
    """Conservative deterministic entailment until an evaluated NLI adapter exists."""
    return _normalise(claim.value) in _normalise(source_text)


def _evidence_level(claim: ProposedClaim, source_text: str) -> Literal["E1", "E2", "E3", "E4"]:
    if claim.kind == "skill":
        return "E1"
    if claim.kind in {"certification", "education"}:
        return "E2"
    concrete_outcome = re.search(
        r"(?:\b\d+(?:\.\d+)?\s*%\b|\b(?:reduced|improved|increased|saved|delivered|built|led|owned|designed)\b)",
        source_text,
        flags=re.IGNORECASE,
    )
    return "E4" if concrete_outcome else "E3"


def verify_agent_output(
    agent_output: AgentOutput,
    forensic_report: dict[str, Any],
    excluded_span_ids: set[str],
    blind_candidate_id: str,
) -> VerifiedCandidateProfile:
    """Drop unsupported claims and emit non-scoring events for every decision."""
    spans = {
        span["span_id"]: span
        for span in forensic_report.get("details", {}).get("spans", [])
        if span.get("span_id")
    }
    facts: list[VerifiedFact] = []
    events: list[VerificationEvent] = []
    for claim in agent_output.proposed_claims:
        cited = [spans.get(span_id) for span_id in claim.source_span_ids]
        missing = [span_id for span_id, span in zip(claim.source_span_ids, cited) if span is None]
        if missing:
            events.append(VerificationEvent(claim_id=claim.claim_id, code="SPAN_NOT_FOUND", detail=",".join(missing)))
            continue
        if any(span["span_id"] in excluded_span_ids for span in cited):
            events.append(VerificationEvent(claim_id=claim.claim_id, code="SPAN_EXCLUDED", detail="cited span was sanitised"))
            continue
        if any(span.get("visibility") != "visible" for span in cited):
            events.append(VerificationEvent(claim_id=claim.claim_id, code="SPAN_NOT_VISIBLE", detail="cited span is not visible"))
            continue
        source_text = "\n".join(
            span.get("normalisation", {}).get("text", span.get("visible_text", "")) for span in cited
        )
        if not _entails(claim, source_text):
            events.append(VerificationEvent(claim_id=claim.claim_id, code="CLAIM_NOT_ENTAILED", detail="value not present in cited text"))
            continue
        sections = sorted({str(span.get("section", "unknown")) for span in cited})
        dates = sorted(set(re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2})?\b", source_text)))
        facts.append(
            VerifiedFact(
                claim_id=claim.claim_id,
                kind=claim.kind,
                value=claim.value,
                source_span_ids=claim.source_span_ids,
                source_sections=sections,
                dates=dates,
                evidence_level=_evidence_level(claim, source_text),
            )
        )
        events.append(VerificationEvent(claim_id=claim.claim_id, code="CLAIM_VERIFIED", detail="deterministic support"))
    return VerifiedCandidateProfile(
        blind_candidate_id=blind_candidate_id,
        facts=facts,
        extraction_confidence=agent_output.extraction_quality.confidence,
        verification_events=events,
    )
