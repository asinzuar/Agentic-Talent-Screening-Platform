"""Orchestrate resume intake through fixed pre-LLM and post-L2 hooks."""

from __future__ import annotations

import argparse
import sys
from uuid import uuid4
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .hooks import IntakeContext, POST_L2_HOOKS, PRE_LLM_HOOKS, run_hooks
from .l1_injection import DEFAULT_SEMANTIC_MODEL, SecurityAssessment
from .l2_resume_agent import DEFAULT_MODEL, build_l2_input, build_messages, call_ollama
from .l3_evidence_verifier import VerifiedCandidateProfile


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineResult(StrictModel):
    """A safe terminal result for every expected pipeline outcome."""

    schema_version: str = "1.0"
    status: Literal["normalized", "normalized_with_flag", "quarantined", "manual_review"]
    stopped_at: Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"]
    resume_sha256: str
    events: list[str] = Field(default_factory=list)
    security_assessment: SecurityAssessment | None = None
    requires_human_review: bool = False
    profile: VerifiedCandidateProfile | None = None


def _progress(message: str, enabled: bool) -> None:
    """Show human-readable progress without corrupting JSON stdout output."""
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _pipeline_result(
    context: IntakeContext,
    *,
    status: Literal["normalized", "normalized_with_flag", "quarantined", "manual_review"],
    stopped_at: Literal["L0_DOC_FORENSICS", "L1_SECURITY_AGENT", "L2_RESUME_AGENT", "L3_EVIDENCE_VERIFIER"],
    profile: VerifiedCandidateProfile | None = None,
) -> PipelineResult:
    if context.forensic_report is None:
        raise RuntimeError("Intake cannot return a result before L0 produces a forensic report.")
    return PipelineResult(
        status=status,
        stopped_at=stopped_at,
        resume_sha256=context.forensic_report["file"]["sha256"],
        events=context.events,
        security_assessment=context.security_assessment,
        requires_human_review=context.requires_human_review,
        profile=profile,
    )


def normalize_resume(
    resume_path: str | Path,
    model: str = DEFAULT_MODEL,
    *,
    l1b_model: str = DEFAULT_SEMANTIC_MODEL,
    show_progress: bool = True,
) -> PipelineResult:
    """Run fixed intake hooks, then quarantined L2 extraction and validation."""
    context = IntakeContext(resume_path=Path(resume_path), model=model, l1b_model=l1b_model)

    _progress("[hooks] Running ordered L0/L1 pre-LLM hooks...", show_progress)
    outcome = run_hooks(context, PRE_LLM_HOOKS)
    if outcome.action != "continue":
        _progress(f"[{outcome.stage}] Stopped by {outcome.hook_name}: {outcome.action}.", show_progress)
        return _pipeline_result(context, status=outcome.action, stopped_at=outcome.stage)

    if context.forensic_report is None:
        raise RuntimeError("Pre-LLM hooks completed without an L0 forensic report.")
    marked_text, context.l2_data_marker = build_l2_input(context.forensic_report, context.excluded_span_ids)
    if not marked_text:
        context.events.append("L2_NO_APPROVED_SPANS")
        return _pipeline_result(context, status="manual_review", stopped_at="L2_RESUME_AGENT")
    context.blind_candidate_id = uuid4().hex
    retry_feedback: list[str] | None = None
    for attempt in range(2):
        _progress(f"[L2] Calling local Ollama model ({model}) on cited, sanitised spans...", show_progress)
        try:
            context.raw_agent_output = call_ollama(
                build_messages(marked_text, context.l2_data_marker, retry_feedback), model
            )
        except (ConnectionError, RuntimeError, ValueError) as error:
            context.events.extend(("L2_AGENT_UNAVAILABLE_OR_INVALID_JSON", str(error)))
            _progress(f"[L2] Needs manual review: {error}", show_progress)
            return _pipeline_result(context, status="manual_review", stopped_at="L2_RESUME_AGENT")

        _progress("[hooks] Running L2 schema/content and L3 evidence hooks...", show_progress)
        outcome = run_hooks(context, POST_L2_HOOKS)
        if outcome.action != "continue":
            _progress(f"[{outcome.stage}] Stopped by {outcome.hook_name}: {outcome.action}.", show_progress)
            return _pipeline_result(context, status=outcome.action, stopped_at=outcome.stage)
        if context.verified_profile is None:
            raise RuntimeError("L3 hook completed without a verified profile.")
        repairable = [
            f"{event.claim_id}: {event.code}"
            for event in context.verified_profile.verification_events
            if event.code != "CLAIM_VERIFIED"
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name")
    parser.add_argument("--l1b-model", default=DEFAULT_SEMANTIC_MODEL, help="Local Ollama model for L1b")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()
    print(
        normalize_resume(
            args.resume,
            args.model,
            l1b_model=args.l1b_model,
            show_progress=not args.quiet,
        ).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
