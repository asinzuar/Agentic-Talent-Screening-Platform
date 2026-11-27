"""Deterministic L3 verification of L2's span-cited proposed claims."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .l1_injection import assess_injection_risk
from .l2_resume_agent import INSTRUCTION_ECHO_PATTERNS, AgentOutput, ProposedClaim
from .contracts import SourceSpan
from .pii_redaction import redact_source_spans, redact_text
from .nli import NliUnavailable, SupportVerifier

# A quote is the claim's own sentence or bullet, never a page or section (SPEC §13.2).
MAX_QUOTE_CHARS = 300
REDACTION = "[redacted]"
VerificationMethod = Literal["deterministic_literal", "nli_literal", "nli_semantic", "literal_fallback"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationEvent(StrictModel):
    """An L3 decision with safe context for the human audit lane.

    ``claim_summary`` is deliberately absent when exposing it would reintroduce
    PII or untrusted instruction text. It is never a ranking input.
    """

    claim_id: str
    claim_summary: str | None = Field(default=None, max_length=MAX_QUOTE_CHARS)
    source_span_ids: list[str] = Field(default_factory=list, max_length=8)
    code: Literal[
        "SPAN_NOT_FOUND",
        "SPAN_NOT_VISIBLE",
        "SPAN_EXCLUDED",
        "CLAIM_NOT_ENTAILED",
        "CLAIM_PII_REDACTED",
        "QUOTE_REPLACED_BY_STATEMENT",
        "CLAIM_VERIFIED",
    ]
    detail: str


NON_REJECTION_CODES = frozenset({"CLAIM_VERIFIED", "QUOTE_REPLACED_BY_STATEMENT"})
"""Events that don't mean a claim was dropped, so they never count towards L3_DROPPED or a retry."""


class VerifiedSpan(StrictModel):
    """What a fact carries across the trust boundary for one cited span."""

    span_id: str
    quote: str
    section: str = "other"
    visibility: Literal["visible"] = "visible"


class VerifiedFact(StrictModel):
    fact_id: str
    kind: Literal["skill", "certification", "experience", "education", "project"]
    statement: str
    source_spans: list[VerifiedSpan] = Field(min_length=1)
    dates: list[str] = Field(default_factory=list)
    evidence_level: Literal["E1", "E2", "E3", "E4"]
    # Audit-only provenance. It is not a scoring input and carries no resume text.
    verification_method: VerificationMethod = "deterministic_literal"
    verification_status: Literal["verified"] = "verified"


class VerifiedCandidateProfile(StrictModel):
    """PII-blind facts that are the only L2-derived data allowed downstream."""

    schema_version: str = "2.2"
    blind_candidate_id: str
    facts: list[VerifiedFact] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    verification_events: list[VerificationEvent] = Field(default_factory=list)
    # A local-NLI processing failure caused remaining eligible claims to use literal support.
    # The profile may be ranked only with an explicit human-review flag.
    l3_fallback_used: bool = False
    l3_fallback_reason: str | None = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _entails(claim: ProposedClaim, source_text: str) -> bool:
    """Conservative deterministic entailment until an evaluated NLI adapter exists."""
    return _normalise(claim.value) in _normalise(source_text)


_PII_VALUE = re.compile(
    r"(?:\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://|\b\d{3}[-. )]+\d{3}[-. ]\d{4}\b)",
    re.I,
)
_REDACT_VALUE = re.compile(
    r"(?:\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?:https?://|www\.)\S+|\b\d{3}[-. )]+\d{3}[-. ]\d{4}\b)",
    re.I,
)
_INSTITUTION = re.compile(r"\b(?:university|college|institute|school)\b", re.I)


def _identity_values(agent_output: AgentOutput) -> list[str]:
    envelope = agent_output.identity_envelope
    values = (envelope.full_name, envelope.email, envelope.phone, envelope.location, *envelope.links)
    return [value for value in values if value and len(_normalise(value)) >= 3]


def _contains_disallowed_pii(claim: ProposedClaim, agent_output: AgentOutput) -> bool:
    """Keep identity-envelope values and institution names out of verified facts."""
    normalised_value = _normalise(claim.value)
    copied_identity = any(
        _normalise(value) in normalised_value for value in _identity_values(agent_output)
    )
    return copied_identity or bool(_PII_VALUE.search(claim.value)) or (
        claim.kind == "education" and bool(_INSTITUTION.search(claim.value))
    )


def _redact(text: str, identity_values: list[str]) -> str:
    """Remove the values a statement may not carry (identity envelope, emails, phones, URLs)."""
    for value in identity_values:
        text = re.sub(re.escape(value), REDACTION, text, flags=re.I)
    return _REDACT_VALUE.sub(REDACTION, text)


def _bounded_quote(text: str, statement: str, cap: int = MAX_QUOTE_CHARS) -> str:
    """Cut a span's text to ``cap`` characters, centred on the statement when it is inside."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= cap:
        return text
    index = text.lower().find(_normalise(statement))
    start = 0 if index < 0 else max(0, index - max(0, cap - len(statement)) // 2)
    start = min(start, len(text) - cap)
    end = start + cap
    if start > 0 and text[start - 1] != " " and " " in text[start:end]:
        start += text[start:end].index(" ") + 1
    if end < len(text) and text[end] != " " and " " in text[start:end]:
        end = start + text[start:end].rindex(" ")
    cut = text[start:end].strip()
    return f"{'…' if start > 0 else ''}{cut}{'…' if end < len(text) else ''}"


def _quote_is_unsafe(quote: str) -> bool:
    """The deterministic L2 output filter plus the L1 injection rules, applied to a quote."""
    return any(pattern.search(quote) for pattern in INSTRUCTION_ECHO_PATTERNS) or bool(
        assess_injection_risk(quote).findings
    )


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


def _reviewable_claim_summary(
    claim: ProposedClaim,
    agent_output: AgentOutput,
    identity_values: list[str],
) -> str | None:
    """Return bounded claim text only when it is safe for a human-facing event."""
    if _contains_disallowed_pii(claim, agent_output):
        return None
    summary, contact = redact_text(claim.value, header_context=True)
    summary = _redact(summary, identity_values).strip()
    if contact or not summary or _quote_is_unsafe(summary):
        return None
    return _bounded_quote(summary, summary)


def verify_agent_output(
    agent_output: AgentOutput,
    forensic_report: dict[str, Any],
    excluded_span_ids: set[str],
    blind_candidate_id: str,
    *,
    support_verifier: SupportVerifier | None = None,
    model_safe_spans: tuple[SourceSpan, ...] | None = None,
) -> VerifiedCandidateProfile:
    """Drop unsupported claims and emit non-scoring events for every decision."""
    spans = {
        span["span_id"]: span
        for span in forensic_report.get("details", {}).get("spans", [])
        if span.get("span_id")
    }
    identity_values = _identity_values(agent_output)
    safe = {}
    if support_verifier is not None:
        # Standalone callers get the same PII policy as the pipeline, not a
        # shortcut that sends the canonical audit text to a model.
        if model_safe_spans is None:
            model_safe_spans, _ = redact_source_spans(tuple(
                SourceSpan.from_l0(span, None, None) for span in spans.values()
            ))
        safe = {span.span_id: span.normalised_text for span in model_safe_spans}
    facts: list[VerifiedFact] = []
    events: list[VerificationEvent] = []
    fallback_reason: str | None = None

    def event(claim: ProposedClaim, code: str, detail: str) -> VerificationEvent:
        return VerificationEvent(
            claim_id=claim.claim_id,
            claim_summary=_reviewable_claim_summary(claim, agent_output, identity_values),
            source_span_ids=list(claim.source_span_ids),
            code=code,
            detail=detail,
        )

    def append_fact(
        claim: ProposedClaim,
        cited: list[dict[str, Any]],
        quotes: list[str],
        method: VerificationMethod,
    ) -> None:
        source_text = "\n".join(quotes)
        facts.append(VerifiedFact(
            fact_id=claim.claim_id,
            kind=claim.kind,
            statement=claim.value,
            source_spans=[VerifiedSpan(span_id=span["span_id"], quote=quote,
                                      section=str(span.get("section", "other")))
                          for span, quote in zip(cited, quotes)],
            dates=sorted(set(re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2})?\b", source_text))),
            evidence_level=_evidence_level(claim, source_text),
            verification_method=method,
        ))
    for claim in agent_output.proposed_claims:
        if _contains_disallowed_pii(claim, agent_output):
            events.append(event(claim, "CLAIM_PII_REDACTED", "claim contains identity-bearing content"))
            continue
        cited = [spans.get(span_id) for span_id in claim.source_span_ids]
        missing = [span_id for span_id, span in zip(claim.source_span_ids, cited) if span is None]
        if missing:
            events.append(event(claim, "SPAN_NOT_FOUND", ",".join(missing)))
            continue
        if any(span["span_id"] in excluded_span_ids for span in cited):
            events.append(event(claim, "SPAN_EXCLUDED", "cited span was sanitised"))
            continue
        if any(span.get("visibility") != "visible" for span in cited):
            events.append(event(claim, "SPAN_NOT_VISIBLE", "cited span is not visible"))
            continue
        if support_verifier is not None:
            # Reject contact-bearing claims; never repair a claim into a new fact.
            _, contact = redact_text(claim.value, header_context=True)
            if contact:
                events.append(event(claim, "CLAIM_PII_REDACTED", "claim contains contact PII"))
                continue
            texts = [_redact(redact_text(safe.get(span["span_id"], ""), header_context=True)[0], identity_values)
                     for span in cited]
            quotes = [_bounded_quote(text, claim.value) for text in texts]
            if any(not q.strip() or _quote_is_unsafe(q) for q in quotes):
                events.append(event(claim, "CLAIM_NOT_ENTAILED", "nli: no safe bounded evidence quote"))
                continue
            literal = _entails(claim, "\n".join(quotes))
            if fallback_reason is None:
                try:
                    verdict = support_verifier.verify(claim.value, "\n".join(quotes))
                except NliUnavailable as error:
                    # Preserve earlier NLI decisions; all remaining claims use the safe literal fallback.
                    fallback_reason = str(error)
                else:
                    if verdict != "entailed":
                        events.append(event(claim, "CLAIM_NOT_ENTAILED", f"nli: {verdict}"))
                        continue
                    method: VerificationMethod = "nli_literal" if literal else "nli_semantic"
                    append_fact(claim, cited, quotes, method)
                    events.append(event(claim, "CLAIM_VERIFIED", f"nli: entailed ({method})"))
                    continue
            if not literal:
                events.append(event(claim, "CLAIM_NOT_ENTAILED",
                                    "literal fallback: value not present in sanitised cited text"))
                continue
            append_fact(claim, cited, quotes, "literal_fallback")
            events.append(event(claim, "CLAIM_VERIFIED", "literal fallback after nli processing failure"))
            continue
        span_texts = [
            span.get("normalisation", {}).get("text", span.get("visible_text", "")) for span in cited
        ]
        if not _entails(claim, "\n".join(span_texts)):
            events.append(event(claim, "CLAIM_NOT_ENTAILED", "value not present in cited text"))
            continue
        # Quotes come from the cited span text, never from the claim, and lose the values a
        # statement may not carry. A claim whose support redaction removes is dropped.
        redacted_texts = [_redact(text, identity_values) for text in span_texts]
        source_text = "\n".join(redacted_texts)
        if not _entails(claim, source_text):
            events.append(event(claim, "CLAIM_PII_REDACTED",
                                "redacting identity values in the cited text removed the claim's support"))
            continue
        verified_spans: list[VerifiedSpan] = []
        replaced = False
        for span, text in zip(cited, redacted_texts):
            quote = _bounded_quote(text, claim.value)
            if _quote_is_unsafe(quote):
                # The fact still crosses, but without the context around the statement.
                quote, replaced = claim.value[:MAX_QUOTE_CHARS], True
            verified_spans.append(VerifiedSpan(
                span_id=span["span_id"], quote=quote, section=str(span.get("section", "other")),
            ))
        if replaced:
            events.append(event(claim, "QUOTE_REPLACED_BY_STATEMENT",
                                "cited text tripped the quote filter; the statement stands in for the quote"))
        dates = sorted(set(re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2})?\b", source_text)))
        facts.append(VerifiedFact(
            fact_id=claim.claim_id,
            kind=claim.kind,
            statement=claim.value,
            source_spans=verified_spans,
            dates=dates,
            evidence_level=_evidence_level(claim, source_text),
            verification_method="deterministic_literal",
        ))
        events.append(event(claim, "CLAIM_VERIFIED", "deterministic support"))
    return VerifiedCandidateProfile(
        blind_candidate_id=blind_candidate_id,
        facts=facts,
        extraction_confidence=agent_output.extraction_quality.confidence,
        verification_events=events,
        l3_fallback_used=fallback_reason is not None,
        l3_fallback_reason=fallback_reason,
    )
