"""Deterministic security risk aggregation and safe-remainder routing."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import DetectionObservation, GroundedFinding, IntakeSecurityDecision


def decide_security(
    observations: Sequence[DetectionObservation],
    findings: Sequence[GroundedFinding],
    l0_excluded: set[str],
    deterministic_score: int = 0,
) -> IntakeSecurityDecision:
    by_id = {observation.observation_id: observation for observation in observations}
    grounded = [
        (by_id[finding.observation_id], finding)
        for finding in findings
        if finding.grounding_status in {"grounded_exact", "grounded_adjacent_window"}
        and finding.observation_id in by_id
    ]
    ungrounded_meaningful = any(
        by_id.get(finding.observation_id, DetectionObservation(
            observation_id="missing", detector="l1_rule", category="missing", severity="low"
        )).severity in {"medium", "high"}
        and finding.grounding_status == "ungrounded"
        for finding in findings
    )
    severities = {observation.severity for observation, _ in grounded}
    risk = (
        "high" if "high" in severities or deterministic_score >= 60
        else "medium" if "medium" in severities or deterministic_score > 0
        else "low"
    )
    sanitised = set(l0_excluded)
    for observation, finding in grounded:
        if observation.severity in {"medium", "high"}:
            sanitised.update(finding.span_ids)
    # Unresolved observations may require a person, but they must not erase broad
    # content or block a clean remainder when a grounded attack is already present.
    continue_to_l2 = bool(grounded) or not ungrounded_meaningful
    notes: list[str] = []
    if ungrounded_meaningful:
        notes.append("One or more meaningful observations were retained for review but could not sanitise text.")
    if risk == "high":
        notes.append("High-risk grounded content was excluded; human review is mandatory.")
    if deterministic_score >= 60:
        notes.append("Risk includes the calibrated aggregate deterministic detector score.")
    return IntakeSecurityDecision(
        risk=risk, sanitised_span_ids=tuple(sorted(sanitised)),
        requires_human_review=risk == "high" or ungrounded_meaningful,
        continue_to_l2=continue_to_l2,
        decision_notes=tuple(notes),
    )
