"""Fixed, typed security hooks for the resume-intake pipeline.

These hooks are internal orchestration points, not an extension/plugin API.
Their order is security-sensitive and owned by the application.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import ValidationError

from .contracts import DetectionObservation, GroundedFinding, IntakeSecurityDecision, SourceSpan
from .grounding import ground_observations
from .risk import decide_security
from .l0_forensics import scan_file
from agentic_talent_screener.llm import default_llm_model

from .l1_injection import (
    SecurityAssessment,
    SecurityFinding,
    assess_injection_risk,
    assess_semantic_injection_risk,
    extract_resume_text,
)
from .l2_resume_agent import AgentOutput, IdentityEnvelope, validate_agent_output
from .l3_evidence_verifier import NON_REJECTION_CODES, VerifiedCandidateProfile, verify_agent_output
from .pii_redaction import PIIRedaction, redact_source_spans
from .nli import SupportVerifier, NliUnavailable



HookAction = Literal["continue", "manual_review", "quarantine"]
HookStage = Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"]
L0_CONTENT_SIGNAL_CODES = {
    "PDF_NONVISIBLE_TEXT", "DOCX_NONVISIBLE_TEXT", "UNICODE_ZERO_WIDTH_OR_FORMAT", "UNICODE_MIXED_SCRIPT",
}


@dataclass
class IntakeContext:
    """State shared only by the fixed, ordered intake hooks for one resume."""

    resume_path: Path
    model: str = field(default_factory=default_llm_model)
    l1b_model: str = field(default_factory=default_llm_model)
    forensic_report: dict[str, Any] | None = None
    raw_text: str | None = None
    sanitised_text: str | None = None
    security_assessment: SecurityAssessment | None = None
    source_spans: tuple[SourceSpan, ...] = ()
    model_safe_spans: tuple[SourceSpan, ...] = ()
    pii_redactions: tuple[PIIRedaction, ...] = ()
    observations: tuple[DetectionObservation, ...] = ()
    grounded_findings: tuple[GroundedFinding, ...] = ()
    security_decision: IntakeSecurityDecision | None = None
    excluded_span_ids: set[str] = field(default_factory=set)
    l0_excluded_span_ids: set[str] = field(default_factory=set)
    security_flagged: bool = False
    requires_human_review: bool = False
    raw_agent_output: dict[str, Any] | None = None
    agent_output: AgentOutput | None = None
    l2_data_marker: str | None = None
    blind_candidate_id: str | None = None
    identity_envelope: IdentityEnvelope | None = None
    verified_profile: VerifiedCandidateProfile | None = None
    support_verifier: SupportVerifier | None = None
    l3_verifier_model: str | None = None
    l3_fallback_used: bool = False
    l3_fallback_reason: str | None = None
    retry_failed_first_pass_preserved: bool = False
    events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HookOutcome:
    """A typed routing decision returned by one hook."""

    hook_name: str
    stage: HookStage
    action: HookAction = "continue"
    events: tuple[str, ...] = ()


IntakeHook = Callable[[IntakeContext], HookOutcome]


def run_hooks(context: IntakeContext, hooks: tuple[IntakeHook, ...]) -> HookOutcome:
    """Run hooks in their declared order and stop at the first terminal action."""
    last = HookOutcome("hook_pipeline", "L0_DOC_FORENSICS")
    for hook in hooks:
        outcome = hook(context)
        context.events.extend(outcome.events)
        last = outcome
        if outcome.action != "continue":
            return outcome
    return last


def _source_spans(report: dict[str, Any]) -> tuple[SourceSpan, ...]:
    raw_spans = report.get("details", {}).get("spans", [])
    return tuple(
        SourceSpan.from_l0(
            raw,
            raw_spans[index - 1]["span_id"] if index else None,
            raw_spans[index + 1]["span_id"] if index + 1 < len(raw_spans) else None,
        )
        for index, raw in enumerate(raw_spans)
    )


def _ensure_model_safe_spans(context: IntakeContext) -> None:
    """Create the contact-PII-redacted representation used at every LLM boundary."""
    if context.model_safe_spans or not context.source_spans:
        return
    context.model_safe_spans, context.pii_redactions = redact_source_spans(context.source_spans)


def _l0_suspicious_span_ids(context: IntakeContext) -> set[str]:
    if context.forensic_report is None:
        return set()
    return {
        span["span_id"]
        for span in context.forensic_report.get("details", {}).get("spans", [])
        if span.get("visibility") != "visible" or span.get("normalisation", {}).get("zero_width_or_format")
    }


def _rebuild_sanitised_text(context: IntakeContext) -> None:
    """Exclude L0/L1 flagged spans before L2 receives text."""
    if context.forensic_report is None:
        return
    parts = [
        span.get("normalisation", {}).get("text", span.get("visible_text", ""))
        for span in context.forensic_report.get("details", {}).get("spans", [])
        if span.get("visibility") == "visible" and span.get("span_id") not in context.excluded_span_ids
    ]
    clean = "\n".join(part for part in parts if part).strip()
    context.sanitised_text = clean


def l0_document_forensics_hook(context: IntakeContext) -> HookOutcome:
    """Admit a bounded document and retain L0's clean text representation."""
    report = scan_file(context.resume_path)
    context.forensic_report = report
    context.source_spans = _source_spans(report)
    _ensure_model_safe_spans(context)
    context.sanitised_text = report.get("details", {}).get("sanitised_text")
    context.excluded_span_ids.update(_l0_suspicious_span_ids(context))
    context.l0_excluded_span_ids = set(context.excluded_span_ids)
    content_only_review = (
        report["decision"] == "manual_review"
        and bool(report.get("issues", []))
        and all(issue["code"] in L0_CONTENT_SIGNAL_CODES for issue in report.get("issues", []))
    )
    if report["decision"] == "allow" or content_only_review:
        events = [
            "L0_FORENSICS_COMPLETED_WITH_EXCLUSIONS"
            if context.l0_excluded_span_ids else "L0_FORENSICS_COMPLETED_CLEAN"
        ]
        if context.pii_redactions:
            events.append("PII_REDACTED_BEFORE_MODEL")
        if context.excluded_span_ids:
            context.security_flagged = True
        return HookOutcome("l0_document_forensics", "L0_DOC_FORENSICS", events=tuple(events))
    return HookOutcome(
        "l0_document_forensics", "L0_DOC_FORENSICS", action=report["decision"],
        events=("L0_FORENSICS_REJECTED", report["decision"]),
    )


def l1_text_extraction_hook(context: IntakeContext) -> HookOutcome:
    """Extract bounded raw text while preserving L0's clean text for L2."""
    try:
        context.raw_text = extract_resume_text(context.resume_path)
    except ValueError as error:
        return HookOutcome("l1_text_extraction", "L1_SECURITY_AGENT", action="manual_review", events=("L1_TEXT_EXTRACTION_REVIEW", str(error)))
    if not context.sanitised_text:
        context.sanitised_text = context.raw_text
    return HookOutcome("l1_text_extraction", "L1_SECURITY_AGENT", events=("L1_TEXT_EXTRACTION_PASSED",))


def l1_deterministic_detection_hook(context: IntakeContext) -> HookOutcome:
    """Run deterministic rules; the LLM judge still runs next for every resume."""
    if context.raw_text is None:
        raise RuntimeError("L1 deterministic hook requires extracted resume text.")
    if not context.source_spans and context.forensic_report is not None:
        context.source_spans = _source_spans(context.forensic_report)
    _ensure_model_safe_spans(context)
    assessment = assess_injection_risk(context.raw_text)
    context.observations = tuple(
        DetectionObservation(
            observation_id=f"rule-{index}", detector="l1_rule", category=finding.code,
            severity="high" if finding.score >= 60 else "medium", excerpt=finding.excerpt,
        )
        for index, finding in enumerate(assessment.findings, start=1)
    )
    context.grounded_findings = ground_observations(context.observations, context.source_spans)
    grounded_by_id = {finding.observation_id: finding for finding in context.grounded_findings}
    findings = [
        finding.model_copy(update={
            "span_ids": list(grounded_by_id[f"rule-{index}"].span_ids),
            "detector": "l1_rule",
            "recommended_action": "manual_review" if assessment.risk == "high" else "sanitise",
        })
        for index, finding in enumerate(assessment.findings, start=1)
    ]
    deterministic = (
        assessment.deterministic_assessment.model_copy(update={"findings": findings})
        if assessment.deterministic_assessment else None
    )
    context.security_assessment = assessment.model_copy(
        update={"findings": findings, "deterministic_assessment": deterministic}
    )
    if not context.observations:
        return HookOutcome("l1_deterministic_detection", "L1_SECURITY_AGENT", events=("L1A_LOW_RISK",))
    return HookOutcome(
        "l1_deterministic_detection",
        "L1_SECURITY_AGENT",
        events=(("L1A_HIGH_RISK",) if assessment.risk == "high" else ("L1A_GREY_ZONE",)),
    )


def l1_semantic_detection_hook(context: IntakeContext) -> HookOutcome:
    """Run the LLM judge for every L0-approved resume and merge it with L1a."""
    if context.raw_text is None or context.security_assessment is None:
        raise RuntimeError("L1 semantic hook requires extracted text and deterministic assessment.")
    try:
        semantic = assess_semantic_injection_risk(
            context.raw_text,
            context.l1b_model,
            context.security_assessment.findings,
            context.model_safe_spans,
        )
    except (ConnectionError, RuntimeError, ValueError) as error:
        event = "L1B_SCHEMA_INVALID" if str(error).startswith("L1B_SCHEMA_INVALID:") else "L1B_MODEL_UNAVAILABLE_OR_INVALID_JSON"
        decision = decide_security(
            context.observations, context.grounded_findings, context.excluded_span_ids,
            context.security_assessment.injection_risk_score,
        )
        context.security_decision = decision
        context.excluded_span_ids = set(decision.sanitised_span_ids)
        context.security_flagged = bool(context.observations) or bool(context.excluded_span_ids)
        context.requires_human_review = decision.requires_human_review
        _rebuild_sanitised_text(context)
        if context.observations and decision.continue_to_l2:
            return HookOutcome(
                "l1_semantic_detection",
                "L1_SECURITY_AGENT",
                events=(event, str(error), "L1B_DETERMINISTIC_FALLBACK"),
            )
        return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", action="manual_review", events=(event, str(error)))
    semantic_observations = tuple(
        DetectionObservation(
            observation_id=f"semantic-{index}", detector="l1_semantic", category=finding.category,
            severity=finding.severity, excerpt=finding.evidence,
            proposed_span_ids=tuple(finding.source_span_ids), rationale=finding.reason,
        )
        for index, finding in enumerate(semantic.findings, start=1)
    )
    observations = (*context.observations, *semantic_observations)
    deterministic_grounded = ground_observations(context.observations, context.source_spans)
    semantic_grounded = ground_observations(semantic_observations, context.model_safe_spans)
    grounded = (*deterministic_grounded, *semantic_grounded)
    decision = decide_security(
        observations, grounded, context.excluded_span_ids,
        context.security_assessment.injection_risk_score,
    )
    context.observations = observations
    context.grounded_findings = grounded
    context.security_decision = decision
    context.excluded_span_ids = set(decision.sanitised_span_ids)
    _rebuild_sanitised_text(context)
    grounded_by_id = {finding.observation_id: finding for finding in grounded}
    findings: list[SecurityFinding] = []
    for observation in observations:
        grounded_finding = grounded_by_id[observation.observation_id]
        findings.append(SecurityFinding(
            code=observation.category if observation.detector == "l1_rule" else f"L1B_{observation.category.upper()}",
            score=100 if observation.severity == "high" else 50 if observation.severity == "medium" else 0,
            excerpt=observation.excerpt, span_ids=list(grounded_finding.span_ids),
            detector="l1_rule" if observation.detector == "l1_rule" else f"llm:{context.l1b_model}",
            recommended_action="manual_review" if observation.severity == "high" else "sanitise",
        ))
    deterministic_findings = [finding for finding in findings if finding.detector == "l1_rule"]
    deterministic = context.security_assessment.deterministic_assessment
    context.security_assessment = SecurityAssessment(
        injection_risk_score=max(context.security_assessment.injection_risk_score, {"low": 0, "medium": 50, "high": 100}[semantic.risk]),
        risk=decision.risk,
        decision="manual_review" if decision.requires_human_review else "allow",
        findings=findings,
        deterministic_assessment=deterministic.model_copy(update={"findings": deterministic_findings}) if deterministic else None,
        semantic_assessment=semantic,
    )
    context.security_flagged = bool(findings) or bool(context.excluded_span_ids)
    context.requires_human_review = decision.requires_human_review
    ungrounded_semantic = any(
        item.observation_id.startswith("semantic-") and item.grounding_status == "ungrounded"
        for item in grounded
    )
    events = ["L1B_COMPLETE"]
    if ungrounded_semantic:
        events.append("L1B_FINDING_DROPPED_UNGROUNDED")
    if not decision.continue_to_l2:
        return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", action="manual_review", events=tuple(events))
    if decision.risk == "high":
        events.append("L1B_HIGH_RISK")
    return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", events=tuple(events))


def l1_sanitisation_hook(context: IntakeContext) -> HookOutcome:
    """Build the clean L2 input after all L1 findings have been collected."""
    _rebuild_sanitised_text(context)
    if not context.sanitised_text:
        return HookOutcome("l1_sanitisation", "L1_SECURITY_AGENT", action="manual_review", events=("L1_SANITISATION_EMPTY",))
    if context.excluded_span_ids:
        context.security_flagged = True
        return HookOutcome("l1_sanitisation", "L1_SECURITY_AGENT", events=("L1_SPANS_SANITISED",))
    return HookOutcome("l1_sanitisation", "L1_SECURITY_AGENT")


def l2_output_schema_hook(context: IntakeContext) -> HookOutcome:
    """Validate L2 output before it can become a pipeline profile."""
    if context.raw_agent_output is None:
        raise RuntimeError("L2 output hook requires a model response.")
    if context.l2_data_marker is None:
        raise RuntimeError("L2 output hook requires the request data marker.")
    try:
        repair_events: list[str] = []
        context.agent_output = validate_agent_output(
            context.raw_agent_output, context.l2_data_marker, repair_events
        )
        context.identity_envelope = context.agent_output.identity_envelope
    except (ValidationError, ValueError) as error:
        extra_fields = sorted(set(context.raw_agent_output) - set(AgentOutput.model_fields))
        policy_error = str(error).startswith("L2_OUTPUT_POLICY_VIOLATION:")
        events = ["L2_OUTPUT_POLICY_VIOLATION" if policy_error else "L2_SCHEMA_VALIDATION_FAILED"]
        events.append(f"L2_VALIDATION_DETAIL:{error}")
        if extra_fields:
            events.append("L2_UNEXPECTED_FIELDS:" + ",".join(extra_fields))
        # A malformed model response is not an unsafe candidate file.  It is a
        # recoverable processing failure and must never become an auto-rejection.
        return HookOutcome("l2_output_schema", "L2_RESUME_AGENT", action="manual_review", events=tuple(events))
    return HookOutcome(
        "l2_output_schema",
        "L2_RESUME_AGENT",
        events=("L2_SCHEMA_VALIDATION_PASSED", *repair_events),
    )


def l3_evidence_verification_hook(context: IntakeContext) -> HookOutcome:
    """Verify L2 claim citations; unverified claims never enter the profile."""
    if context.agent_output is None or context.forensic_report is None or context.blind_candidate_id is None:
        raise RuntimeError("L3 verification requires validated L2 output, L0 report, and blind ID.")
    kwargs = ({"support_verifier": context.support_verifier, "model_safe_spans": context.model_safe_spans or None}
              if context.support_verifier is not None else {})
    backend_events: tuple[str, ...] = ()
    if context.support_verifier is not None:
        context.l3_verifier_model = getattr(context.support_verifier, "model", "injected verifier")
        backend_events = (f"L3_NLI_MODEL:{context.l3_verifier_model}",)
    try:
        context.verified_profile = verify_agent_output(
            context.agent_output, context.forensic_report, context.excluded_span_ids,
            context.blind_candidate_id, **kwargs,
        )
    except NliUnavailable as error:
        context.verified_profile = None
        context.requires_human_review = True
        return HookOutcome("l3_evidence_verification", "L3_EVIDENCE_VERIFIER", action="manual_review",
                           events=(*backend_events, "L3_NLI_PROCESSING_FAILED", str(error)))
    context.l3_fallback_used = context.verified_profile.l3_fallback_used
    context.l3_fallback_reason = context.verified_profile.l3_fallback_reason
    if context.l3_fallback_used:
        # Earlier NLI decisions remain valid; later facts have literal-only provenance and require review.
        context.requires_human_review = True
        backend_events += ("L3_NLI_LITERAL_FALLBACK",)
    rejected = sum(
        event.code not in NON_REJECTION_CODES for event in context.verified_profile.verification_events
    )
    return HookOutcome(
        "l3_evidence_verification",
        "L3_EVIDENCE_VERIFIER",
        events=(*backend_events, f"L3_VERIFIED:{len(context.verified_profile.facts)}", f"L3_DROPPED:{rejected}"),
    )


# Keep the execution order explicit. These are intentionally not user-supplied plugins.
PRE_LLM_HOOKS: tuple[IntakeHook, ...] = (
    l0_document_forensics_hook,
    l1_text_extraction_hook,
    l1_deterministic_detection_hook,
    l1_semantic_detection_hook,
    l1_sanitisation_hook,
)
POST_L2_HOOKS: tuple[IntakeHook, ...] = (l2_output_schema_hook, l3_evidence_verification_hook)
