"""Explicit, PII-blind handoff from L3 facts to the deterministic ranking contract.

This adapter is the only supported bridge between the untrusted intake lane and
``ranking.VerifiedProfile``.  Skills cross as free text, exactly as verified; there
is no skill catalogue (SPEC §6.3) and nothing is canonicalised at the boundary.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from agentic_talent_screener.ranking.schemas import (
    EvidenceLevel,
    SecurityFlag,
    SkillClaim,
    Span,
    VerifiedProfile,
)

from .l1_injection import SecurityAssessment
from .l3_evidence_verifier import VerifiedCandidateProfile


_PII_PATTERN = re.compile(r"(?:\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://|\b\d{3}[-. )]+\d{3}[-. ]\d{4}\b)", re.I)
_ATTACK_FAMILY = {
    "INSTRUCTION_OVERRIDE": "A1", "ROLE_PROMPT_REFERENCE": "A2",
    "RULE_BYPASS": "A1", "HIRING_OUTCOME_DIRECTIVE": "A1",
    "SCORE_MANIPULATION": "A1", "STATUS_OVERRIDE": "A1",
    "PERSONA_HIJACK": "A2", "SECURITY_BYPASS": "A1",
    "DELIMITER_SPOOFING": "A2", "TOOL_EXECUTION_HIJACK": "A9",
    "DATA_EXFILTRATION": "A9", "CONTEXT_RESET": "A1",
    "DIRECT_AGENT_ADDRESS": "A8", "OBFUSCATED_PAYLOAD": "A4",
    "RECOMMENDATION_MANIPULATION": "A8", "ATTENTION_MANIPULATION": "A8",
    "CONTEXT_DOMINATION": "A8", "EVALUATION_BASIS_MANIPULATION": "A8",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RankingHandoff(_Strict):
    """Result plus a non-sensitive account of facts excluded at the boundary."""

    profile: VerifiedProfile
    accepted_fact_ids: tuple[str, ...] = ()
    dropped_fact_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def to_ranking_profile(
    profile: VerifiedCandidateProfile,
    security: SecurityAssessment,
) -> RankingHandoff:
    """Create a ranking profile from visible, L3-verified *skill* facts only.

    Roles, projects, education and location remain ``None`` until their fields have
    equivalent deterministic intake contracts.  Ranking then handles them as missing
    components rather than silently treating them as zero.
    """
    if not re.fullmatch(r"C-\d{4}", profile.blind_candidate_id):
        raise ValueError("ranking handoff requires a blind candidate ID in C-#### format")

    claims: list[SkillClaim] = []
    accepted: list[str] = []
    dropped: list[str] = []
    for fact in profile.facts:
        if fact.kind != "skill" or _PII_PATTERN.search(fact.value):
            dropped.append(fact.claim_id)
            continue
        claims.append(
            SkillClaim(
                id=fact.claim_id,
                skill=fact.value,
                evidence_level=EvidenceLevel[fact.evidence_level],
                spans=tuple(Span(location=span_id, quote=fact.value) for span_id in fact.source_span_ids),
            )
        )
        accepted.append(fact.claim_id)

    risk = security.risk
    flags: list[SecurityFlag] = []
    for index, finding in enumerate(security.findings, start=1):
        if not finding.span_ids:
            continue
        code = finding.code.removeprefix("L1B_")
        family = _ATTACK_FAMILY.get(code, "A8")
        flags.append(
            SecurityFlag(
                flag_id=f"intake-{index}", attack_family=family, layer="L1", risk=risk,
                location=finding.span_ids[0], note=finding.code,
            )
        )
    return RankingHandoff(
        profile=VerifiedProfile(candidate_id=profile.blind_candidate_id, claims=tuple(claims), security_flags=tuple(flags)),
        accepted_fact_ids=tuple(accepted), dropped_fact_ids=tuple(dropped),
        notes=("Only L3-verified skill facts cross this boundary, as free text.",),
    )
