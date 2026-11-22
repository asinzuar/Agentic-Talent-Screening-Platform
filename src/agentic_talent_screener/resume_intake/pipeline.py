"""Orchestrate resume intake through fixed pre-LLM and post-L2 hooks."""

from __future__ import annotations

import argparse
import sys
from uuid import uuid4
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_talent_screener.llm import default_llm_model

from .hooks import IntakeContext, POST_L2_HOOKS, PRE_LLM_HOOKS, run_hooks
from .contracts import IntakeSecurityDecision, SanitisedDocumentView
from .l1_injection import SecurityAssessment
from .l2_resume_agent import AgentOutput, build_l2_input, build_messages, call_model
from .l3_evidence_verifier import NON_REJECTION_CODES, VerifiedCandidateProfile


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntakeEventDetail(StrictModel):
    """Human-readable explanation of an intake outcome; raw codes remain in ``events``."""

    code: str
    stage: str
    severity: Literal["info", "warning", "action_required"]
    message: str
    span_ids: list[str] = Field(default_factory=list)


class PipelineResult(StrictModel):
    """A safe terminal result for every expected pipeline outcome."""

    schema_version: str = "1.0"
    status: Literal["normalized", "normalized_with_flag", "l2_extracted", "quarantined", "manual_review"]
    stopped_at: Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"]
    resume_sha256: str
    events: list[str] = Field(default_factory=list)
    event_details: list[IntakeEventDetail] = Field(default_factory=list)
    sanitised_span_ids: list[str] = Field(default_factory=list)
    security_assessment: SecurityAssessment | None = None
    security_decision: IntakeSecurityDecision | None = None
    requires_human_review: bool = False
    # Development-only output for L2 quality evaluation. It is unverified and must never
    # cross the trust boundary into matching, ranking, or recruiter decision support.
    l2_output: AgentOutput | None = None
    profile: VerifiedCandidateProfile | None = None


def _progress(message: str, enabled: bool) -> None:
    """Show human-readable progress without corrupting JSON stdout output."""
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _pipeline_result(
    context: IntakeContext,
    *,
    status: Literal["normalized", "normalized_with_flag", "l2_extracted", "quarantine", "quarantined", "manual_review"],
    stopped_at: Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"],
    l2_output: AgentOutput | None = None,
    profile: VerifiedCandidateProfile | None = None,
) -> PipelineResult:
    if context.forensic_report is None:
        raise RuntimeError("Intake cannot return a result before L0 produces a forensic report.")
    # Hooks use the operational verb ``quarantine``; the external result uses
    # the past-tense terminal status ``quarantined``.
    normalized_status = "quarantined" if status == "quarantine" else status
    details: list[IntakeEventDetail] = []
    if context.forensic_report is not None:
        if context.l0_excluded_span_ids:
            details.append(IntakeEventDetail(
                code="L0_FORENSICS_COMPLETED_WITH_EXCLUSIONS", stage="L0 document forensics",
                severity="warning",
                message="File processing completed, but non-visible or suspicious text was excluded from later stages.",
                span_ids=sorted(context.l0_excluded_span_ids),
            ))
        else:
            details.append(IntakeEventDetail(
                code="L0_FORENSICS_COMPLETED_CLEAN", stage="L0 document forensics", severity="info",
                message="File processing completed with no L0 spans excluded.",
            ))
    if context.pii_redactions:
        details.append(IntakeEventDetail(
            code="PII_REDACTED_BEFORE_MODEL", stage="L0 data minimisation", severity="info",
            message=("Contact PII was replaced with redaction tokens before any LLM call; "
                     "canonical evidence remains restricted to the audit lane."),
            span_ids=sorted({item.span_id for item in context.pii_redactions if item.span_id}),
        ))
    if context.security_decision is not None:
        decision = context.security_decision
        details.append(IntakeEventDetail(
            code=("L1_HIGH_RISK_SAFE_REMAINDER" if decision.risk == "high" and decision.continue_to_l2
                  else "L1_REVIEW_REQUIRED_NO_SAFE_REMAINDER" if not decision.continue_to_l2
                  else "L1_SECURITY_ANALYSIS_COMPLETED"),
            stage="L1 security analysis",
            severity="action_required" if decision.requires_human_review else "warning" if decision.risk == "medium" else "info",
            message=("High-risk content was excluded; extraction continued on the safe remainder and requires human review."
                     if decision.risk == "high" and decision.continue_to_l2
                     else "Security observations could not be grounded to safe actionable spans; human review is required."
                     if not decision.continue_to_l2
                     else "Security analysis completed; only the listed spans were excluded before extraction."),
            span_ids=list(decision.sanitised_span_ids),
        ))
    if context.agent_output is not None:
        details.append(IntakeEventDetail(
            code="L2_OUTPUT_VALIDATED", stage="L2 quarantined extraction", severity="info",
            message="L2 returned a schema-validated, unverified claims record. It is not eligible for ranking until L3 verification.",
        ))
    if normalized_status == "manual_review":
        details.append(IntakeEventDetail(
            code="MANUAL_REVIEW_REQUIRED", stage="intake routing", severity="action_required",
            message="The system did not make a hiring decision. An authorised reviewer must resolve this intake outcome.",
        ))
    return PipelineResult(
        status=normalized_status,
        stopped_at=stopped_at,
        resume_sha256=context.forensic_report["file"]["sha256"],
        events=context.events,
        event_details=details,
        sanitised_span_ids=sorted(context.excluded_span_ids),
        security_assessment=context.security_assessment,
        security_decision=context.security_decision,
        requires_human_review=context.requires_human_review,
        l2_output=l2_output,
        profile=profile,
    )


def normalize_resume(
    resume_path: str | Path,
    model: str | None = None,
    *,
    l1b_model: str | None = None,
    show_progress: bool = True,
    stop_after_l2: bool = False,
) -> PipelineResult:
    """Run intake hooks and optionally stop after schema-validated L2 for development evaluation."""
    model = model or default_llm_model()
    context = IntakeContext(resume_path=Path(resume_path), model=model, l1b_model=l1b_model or model)

    _progress("[hooks] Running ordered L0/L1 pre-LLM hooks...", show_progress)
    outcome = run_hooks(context, PRE_LLM_HOOKS)
    if outcome.action != "continue":
        _progress(f"[{outcome.stage}] Stopped by {outcome.hook_name}: {outcome.action}.", show_progress)
        return _pipeline_result(context, status=outcome.action, stopped_at=outcome.stage)

    if context.forensic_report is None:
        raise RuntimeError("Pre-LLM hooks completed without an L0 forensic report.")
    if context.security_decision is None:
        raise RuntimeError("L1 completed without a structured security decision.")
    sanitised_view = SanitisedDocumentView(
        document_sha256=context.forensic_report["file"]["sha256"],
        included_spans=tuple(
            span for span in context.model_safe_spans if span.span_id not in context.excluded_span_ids
        ),
        excluded_span_ids=tuple(sorted(context.excluded_span_ids)),
        security_decision=context.security_decision,
    )
    marked_text, context.l2_data_marker = build_l2_input(sanitised_view)
    if not marked_text:
        context.events.append("L2_NO_APPROVED_SPANS")
        return _pipeline_result(context, status="manual_review", stopped_at="L2_RESUME_AGENT")
    # The ranking contract accepts only blind C-#### identifiers.  Persisted systems
    # must allocate this collision-free; this local pipeline has no persistence.
    context.blind_candidate_id = f"C-{uuid4().int % 10_000:04d}"
    retry_feedback: list[str] | None = None
    for attempt in range(2):
        _progress(f"[L2] Calling OpenRouter model ({model}) on cited, sanitised spans...", show_progress)
        try:
            context.raw_agent_output = call_model(
                build_messages(marked_text, context.l2_data_marker, retry_feedback), model
            )
        except (ConnectionError, RuntimeError, ValueError) as error:
            context.events.extend(("L2_AGENT_UNAVAILABLE_OR_INVALID_JSON", str(error)))
            _progress(f"[L2] Needs manual review: {error}", show_progress)
            return _pipeline_result(context, status="manual_review", stopped_at="L2_RESUME_AGENT")

        post_l2_hooks = POST_L2_HOOKS[:1] if stop_after_l2 else POST_L2_HOOKS
        hook_label = "L2 schema/content hook" if stop_after_l2 else "L2 schema/content and L3 evidence hooks"
        _progress(f"[hooks] Running {hook_label}...", show_progress)
        outcome = run_hooks(context, post_l2_hooks)
        if outcome.action != "continue":
            _progress(f"[{outcome.stage}] Stopped by {outcome.hook_name}: {outcome.action}.", show_progress)
            return _pipeline_result(context, status=outcome.action, stopped_at=outcome.stage)
        if stop_after_l2:
            if context.agent_output is None:
                raise RuntimeError("L2 schema hook completed without an agent output.")
            context.events.append("L2_EXTRACTION_COMPLETE_DEVELOPMENT_ONLY")
            return _pipeline_result(
                context,
                status="l2_extracted",
                stopped_at="L2_RESUME_AGENT",
                l2_output=context.agent_output,
            )
        if context.verified_profile is None:
            raise RuntimeError("L3 hook completed without a verified profile.")
        repairable = [
            f"{event.claim_id}: {event.code}"
            for event in context.verified_profile.verification_events
            if event.code not in NON_REJECTION_CODES
        ]
        if repairable and attempt == 0:
            retry_feedback = repairable
            context.events.append("L2_L3_TARGETED_RETRY")
            continue
        break

    if context.verified_profile is None or not context.verified_profile.facts:
        context.events.append("L3_NO_VERIFIED_FACTS")
        return _pipeline_result(context, status="manual_review", stopped_at="L3_EVIDENCE_VERIFIER")
    _progress("[L3] Passed: evidence-backed profile verified successfully.", show_progress)
    return _pipeline_result(
        context,
        status="normalized_with_flag" if context.security_flagged else "normalized",
        stopped_at="L3_EVIDENCE_VERIFIER",
        profile=context.verified_profile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize PDF, DOCX, or TXT resumes using ordered intake hooks")
    parser.add_argument("resume", help="Path to a PDF, DOCX, or TXT resume")
    parser.add_argument("--model", default=None, help="OpenRouter model ID (default: $ATS_LLM_MODEL)")
    parser.add_argument("--l1b-model", default=None, help="OpenRouter model ID for L1b (default: same as --model)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    parser.add_argument(
        "--stop-after-l2",
        action="store_true",
        help="Development-only: return schema-validated but unverified L2 output; never use it for scoring or ranking.",
    )
    args = parser.parse_args()
    print(
        normalize_resume(
            args.resume,
            args.model,
            l1b_model=args.l1b_model,
            show_progress=not args.quiet,
            stop_after_l2=args.stop_after_l2,
        ).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
