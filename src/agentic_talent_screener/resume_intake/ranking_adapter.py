"""Explicit, PII-blind handoff from L3 facts to the deterministic ranking contract.

This adapter is the only supported bridge between the untrusted intake lane and
``ranking.VerifiedProfile`` (RESUME_INTAKE_SPEC §3.4, SPEC §13.2).  Facts of all five kinds
cross as free text, exactly as verified, with their bounded and PII-screened quotes; there is
no skill catalogue (SPEC §6.3) and nothing is canonicalised at the boundary.  The quotes are
for HR review and the Explanation Agent: ranking hands the Matching Agent a quote-free
``MatcherView``.  Security findings from L0 and L1 cross as flags that carry span IDs only,
never the flagged text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from agentic_talent_screener.ranking.schemas import (
    ALL_FACT_KINDS,
    EvidenceLevel,
    Fact,
    RiskLevel,
    SecurityFlag,
    Span,
    VerifiedProfile,
)

from .l1_injection import SecurityAssessment
from .l3_evidence_verifier import VerifiedCandidateProfile
from .l0_forensics import _is_decorative_text, _unicode_payload_only


_PII_PATTERN = re.compile(r"(?:\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://|\b\d{3}[-. )]+\d{3}[-. ]\d{4}\b)", re.I)
_RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# L0 findings that cite spans: issue code -> (attack family, which spans the finding covers).
_L0_FINDINGS: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    "PDF_NONVISIBLE_TEXT": (
        "A3", lambda span: span.get("visibility") != "visible"
        and not _is_decorative_text(span.get("normalisation", {}).get("text", "")),
    ),
    "DOCX_NONVISIBLE_TEXT": (
        "A3", lambda span: span.get("visibility") != "visible"
        and not _is_decorative_text(span.get("normalisation", {}).get("text", "")),
    ),
    "UNICODE_ZERO_WIDTH_OR_FORMAT": (
        "A4", lambda span: _unicode_payload_only(
            span.get("normalisation", {}).get("text", ""),
            span.get("normalisation", {}).get("zero_width_or_format", []),
        ),
    ),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RankingHandoff(_Strict):
    """Result plus a non-sensitive account of facts excluded at the boundary."""

    profile: VerifiedProfile
    accepted_fact_ids: tuple[str, ...] = ()
    dropped_fact_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _l0_flags(forensic_report: dict[str, Any]) -> list[SecurityFlag]:
    """One flag per L0 finding, listing every span the finding covers (IDs only, no text)."""
    spans = forensic_report.get("details", {}).get("spans", [])
    flags: list[SecurityFlag] = []
    for issue in forensic_report.get("issues", []):
        rule = _L0_FINDINGS.get(issue.get("code", ""))
        if rule is None:
            continue
        family, covers = rule
        locations = tuple(span["span_id"] for span in spans if span.get("span_id") and covers(span))
        if locations:
            flags.append(
                SecurityFlag(
                    flag_id="pending", attack_family=family, layer="L0",
                    risk=issue.get("severity", "medium"), locations=locations, note=issue["code"],
                )
            )
    return flags


def _l1_flags(security: SecurityAssessment) -> list[SecurityFlag]:
    """One flag per grounded L1 finding, with the finding's own severity and every span."""
    return [
        SecurityFlag(
            flag_id="pending", attack_family=finding.attack_family, layer="L1", risk=finding.risk,
            locations=tuple(finding.span_ids), note=finding.code,
        )
        for finding in security.findings
        if finding.span_ids
    ]


def to_ranking_profile(
    profile: VerifiedCandidateProfile,
    security: SecurityAssessment,
    *,
    forensic_report: dict[str, Any] | None = None,
    intake_status: str | None = None,
    requires_human_review: bool = False,
) -> RankingHandoff:
    """Create a ranking profile from visible, L3-verified facts of every kind.

    ``forensic_report`` supplies the L0 findings (hidden text, Unicode obfuscation); without it
    only L1 findings become flags.  ``intake_status`` defaults to ``verified_with_flag`` when
    there is any flag or a non-low risk, else ``verified``.  ``overall_risk`` is the highest of
    the L1 assessment and every flag, kept apart from each flag's own risk.
    """
    if not re.fullmatch(r"C-\d{4}", profile.blind_candidate_id):
        raise ValueError("ranking handoff requires a blind candidate ID in C-#### format")
    # Preserve the L3 availability-fallback review requirement even if a future
    # caller forgets to propagate PipelineResult.requires_human_review.
    requires_human_review = requires_human_review or profile.l3_fallback_used

    facts: list[Fact] = []
    accepted: list[str] = []
    dropped: list[str] = []
    for fact in profile.facts:
        texts = (fact.statement, *(span.quote for span in fact.source_spans))
        if any(_PII_PATTERN.search(text) for text in texts):
            dropped.append(fact.fact_id)
            continue
        facts.append(
            Fact(
                fact_id=fact.fact_id,
                kind=fact.kind,
                statement=fact.statement,
                evidence_level=EvidenceLevel[fact.evidence_level],
                dates=tuple(fact.dates),
                spans=tuple(
                    Span(location=span.span_id, quote=span.quote, visible=span.visibility == "visible")
                    for span in fact.source_spans
                ),
            )
        )
        accepted.append(fact.fact_id)

    flags = [*(_l0_flags(forensic_report) if forensic_report else ()), *_l1_flags(security)]
    flags = [flag.model_copy(update={"flag_id": f"intake-{i}"}) for i, flag in enumerate(flags, start=1)]
    overall: RiskLevel = max(
        (security.risk, *(flag.risk for flag in flags)), key=lambda risk: _RISK_ORDER[risk]
    )
    status = intake_status or ("verified_with_flag" if flags or overall != "low" or requires_human_review else "verified")
    if status not in {"verified", "verified_with_flag"}:
        raise ValueError("only verified or verified_with_flag intake results may be handed to ranking")

    return RankingHandoff(
        profile=VerifiedProfile(
            candidate_id=profile.blind_candidate_id,
            facts=tuple(facts),
            extracted_kinds=ALL_FACT_KINDS,
            intake_status=status,
            overall_risk=overall,
            security_flags=tuple(flags),
        ),
        accepted_fact_ids=tuple(accepted),
        dropped_fact_ids=tuple(dropped),
        notes=(
            "Verified facts cross as free text with PII-screened quotes; flags carry span IDs only.",
            *(("L3 used literal fallback after local NLI processing failed; human review is required.",)
              if requires_human_review else ()),
        ),
    )
