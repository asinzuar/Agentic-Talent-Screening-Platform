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

from .l0_forensics import scan_file
from .l1_injection import (
    DEFAULT_SEMANTIC_MODEL,
    SecurityAssessment,
    SecurityFinding,
    assess_injection_risk,
    assess_semantic_injection_risk,
    extract_resume_text,
    merge_semantic_assessment,
)
from .l2_resume_agent import AgentOutput, DEFAULT_MODEL, IdentityEnvelope, validate_agent_output
from .l3_evidence_verifier import VerifiedCandidateProfile, verify_agent_output


HookAction = Literal["continue", "manual_review", "quarantine"]
HookStage = Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"]
L0_CONTENT_SIGNAL_CODES = {
    "PDF_NONVISIBLE_TEXT", "DOCX_NONVISIBLE_TEXT", "UNICODE_ZERO_WIDTH_OR_FORMAT", "UNICODE_MIXED_SCRIPT",
}


@dataclass
class IntakeContext:
    """State shared only by the fixed, ordered intake hooks for one resume."""

    resume_path: Path
    model: str = DEFAULT_MODEL
    l1b_model: str = DEFAULT_SEMANTIC_MODEL
    forensic_report: dict[str, Any] | None = None
    raw_text: str | None = None
    sanitised_text: str | None = None
    security_assessment: SecurityAssessment | None = None
    excluded_span_ids: set[str] = field(default_factory=set)
    security_flagged: bool = False
    requires_human_review: bool = False
    raw_agent_output: dict[str, Any] | None = None
    agent_output: AgentOutput | None = None
    l2_data_marker: str | None = None
    blind_candidate_id: str | None = None
    identity_envelope: IdentityEnvelope | None = None
    verified_profile: VerifiedCandidateProfile | None = None
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


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def _span_ids_for_excerpt(context: IntakeContext, excerpt: str) -> list[str]:
    if context.forensic_report is None:
        return []
    needle = _normalise(excerpt)
    if not needle:
        return []
    return [
        span["span_id"]
        for span in context.forensic_report.get("details", {}).get("spans", [])
        if needle in _normalise(span.get("text", "")) or _normalise(span.get("text", "")) in needle
    ]


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
    context.sanitised_text = report.get("details", {}).get("sanitised_text")
    context.excluded_span_ids.update(_l0_suspicious_span_ids(context))
    content_only_review = (
        report["decision"] == "manual_review"
        and bool(report.get("issues", []))
        and all(issue["code"] in L0_CONTENT_SIGNAL_CODES for issue in report.get("issues", []))
    )
    if report["decision"] == "allow" or content_only_review:
        events = ["L0_FORENSICS_PASSED"]
        if context.excluded_span_ids:
            context.security_flagged = True
            events.append("L0_SPANS_EXCLUDED")
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
    """Run deterministic rules; Qwen still runs next for every resume."""
    if context.raw_text is None:
        raise RuntimeError("L1 deterministic hook requires extracted resume text.")
    assessment = assess_injection_risk(context.raw_text)
    findings: list[SecurityFinding] = []
    for finding in assessment.findings:
        span_ids = _span_ids_for_excerpt(context, finding.excerpt)
        findings.append(finding.model_copy(update={
            "span_ids": span_ids,
            "detector": "l1_rule",
            "recommended_action": "manual_review" if assessment.risk == "high" else "sanitise",
        }))
        context.excluded_span_ids.update(span_ids)
    context.security_assessment = assessment.model_copy(update={"findings": findings})
    if not findings:
        return HookOutcome("l1_deterministic_detection", "L1_SECURITY_AGENT", events=("L1A_LOW_RISK",))
    context.security_flagged = True
    if assessment.risk == "high":
        context.requires_human_review = True
    return HookOutcome(
        "l1_deterministic_detection",
        "L1_SECURITY_AGENT",
        events=(("L1A_HIGH_RISK",) if assessment.risk == "high" else ("L1A_GREY_ZONE",)),
    )


def l1_semantic_detection_hook(context: IntakeContext) -> HookOutcome:
    """Run Qwen for every L0-approved resume and merge it with L1a."""
    if context.raw_text is None or context.security_assessment is None:
        raise RuntimeError("L1 semantic hook requires extracted text and deterministic assessment.")
    try:
        semantic = assess_semantic_injection_risk(context.raw_text, context.l1b_model)
    except (ConnectionError, RuntimeError, ValueError) as error:
        event = "L1B_SCHEMA_INVALID" if str(error).startswith("L1B_SCHEMA_INVALID:") else "L1B_MODEL_UNAVAILABLE_OR_INVALID_JSON"
        # A high/medium deterministic result is an approved safe fallback:
        # retain the evidence, remove it from L2 input, and require HR review.
        if context.security_assessment.risk in {"medium", "high"}:
            context.security_flagged = True
            context.requires_human_review = True
            _rebuild_sanitised_text(context)
            return HookOutcome(
                "l1_semantic_detection",
                "L1_SECURITY_AGENT",
                events=(event, str(error), "L1B_DETERMINISTIC_FALLBACK", "L1_SPANS_SANITISED"),
            )
        return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", action="manual_review", events=(event, str(error)))
    combined = merge_semantic_assessment(context.security_assessment, semantic)
    findings: list[SecurityFinding] = []
    for finding in combined.findings:
        span_ids = _span_ids_for_excerpt(context, finding.excerpt)
        if finding.detector.startswith("qwen:") and not span_ids:
            return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", action="manual_review", events=("L1B_EVIDENCE_NOT_IN_L0_SPANS",))
        findings.append(finding.model_copy(update={"span_ids": span_ids}))
        context.excluded_span_ids.update(span_ids)
    context.security_assessment = combined.model_copy(update={"findings": findings})
    if combined.risk == "high":
        context.requires_human_review = True
        _rebuild_sanitised_text(context)
        return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", events=("L1B_HIGH_RISK", "L1_SPANS_SANITISED"))
    return HookOutcome("l1_semantic_detection", "L1_SECURITY_AGENT", events=("L1B_COMPLETE",))


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
        context.agent_output = validate_agent_output(context.raw_agent_output, context.l2_data_marker)
        context.identity_envelope = context.agent_output.identity_envelope
    except (ValidationError, ValueError) as error:
        extra_fields = sorted(set(context.raw_agent_output) - set(AgentOutput.model_fields))
        policy_error = str(error).startswith("L2_OUTPUT_POLICY_VIOLATION:")
        events = ["L2_OUTPUT_POLICY_VIOLATION" if policy_error else "L2_SCHEMA_VALIDATION_FAILED"]
        if extra_fields:
            events.append("L2_UNEXPECTED_FIELDS:" + ",".join(extra_fields))
        return HookOutcome("l2_output_schema", "L2_RESUME_AGENT", action="quarantine", events=tuple(events))
    return HookOutcome("l2_output_schema", "L2_RESUME_AGENT", events=("L2_SCHEMA_VALIDATION_PASSED",))


def l3_evidence_verification_hook(context: IntakeContext) -> HookOutcome:
    """Verify L2 claim citations; unverified claims never enter the profile."""
    if context.agent_output is None or context.forensic_report is None or context.blind_candidate_id is None:
        raise RuntimeError("L3 verification requires validated L2 output, L0 report, and blind ID.")
    context.verified_profile = verify_agent_output(
        context.agent_output,
        context.forensic_report,
        context.excluded_span_ids,
        context.blind_candidate_id,
    )
    rejected = sum(event.code != "CLAIM_VERIFIED" for event in context.verified_profile.verification_events)
    return HookOutcome(
        "l3_evidence_verification",
        "L3_EVIDENCE_VERIFIER",
        events=(f"L3_VERIFIED:{len(context.verified_profile.facts)}", f"L3_DROPPED:{rejected}"),
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
