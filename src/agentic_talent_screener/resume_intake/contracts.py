"""Shared, typed contracts for the resume-intake trust boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpan(_Strict):
    span_id: str = Field(pattern=r"^p\d+:s\d{4}$")
    page: int | None = None
    text: str
    normalised_text: str
    visible_text: str
    section: str = "other"
    visibility: Literal["visible", "hidden", "off_page", "occluded", "unrendered", "unknown"]
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    adjacent_before: str | None = None
    adjacent_after: str | None = None

    @classmethod
    def from_l0(cls, raw: dict[str, Any], before: str | None, after: str | None) -> "SourceSpan":
        location = raw.get("location", {})
        bbox = location.get("bbox")
        return cls(
            span_id=raw["span_id"], page=raw.get("page"), text=raw.get("text", ""),
            normalised_text=raw.get("normalisation", {}).get("text", raw.get("text", "")),
            visible_text=raw.get("visible_text", ""), section=raw.get("section", "other"),
            visibility=raw.get("visibility", "unknown"), char_start=location.get("char_start"),
            char_end=location.get("char_end"), bbox=tuple(bbox) if bbox else None,
            adjacent_before=before, adjacent_after=after,
        )


class DetectionObservation(_Strict):
    observation_id: str
    detector: Literal["l1_rule", "l1_semantic", "classifier"]
    category: str
    severity: Literal["low", "medium", "high"]
    excerpt: str = ""
    proposed_span_ids: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None
    rationale: str | None = None


class GroundedFinding(_Strict):
    observation_id: str
    grounding_status: Literal["grounded_exact", "grounded_adjacent_window", "ungrounded", "invalid"]
    span_ids: tuple[str, ...] = ()
    matched_text: str | None = None
    method: Literal["explicit_ids", "single_span", "adjacent_window", "offset", "none"]
    reason: str | None = None


class IntakeSecurityDecision(_Strict):
    risk: Literal["low", "medium", "high"]
    sanitised_span_ids: tuple[str, ...] = ()
    requires_human_review: bool = False
    continue_to_l2: bool = True
    decision_notes: tuple[str, ...] = ()


class SanitisedDocumentView(_Strict):
    """The only L0/L1 document representation L2 may receive."""

    document_sha256: str
    included_spans: tuple[SourceSpan, ...]
    excluded_span_ids: tuple[str, ...] = ()
    security_decision: IntakeSecurityDecision
