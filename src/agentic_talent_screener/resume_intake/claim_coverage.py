"""Deterministic completeness checks for high-impact resume claims.

This module intentionally does not infer new candidate facts.  It identifies
source lines that *must be accounted for* by L2: role-shaped Experience lines
and degree-shaped Education lines.  A cited, L3-verified fact covers an
anchor; otherwise the intake is retried once and then routed to human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .contracts import SourceSpan
from .l3_evidence_verifier import VerifiedCandidateProfile


CoverageKind = Literal["experience", "education"]

# These patterns are deliberately high precision.  They are coverage anchors,
# not a resume parser and never create claims by themselves.
_DEGREE = re.compile(
    r"\b(?:b\.?\s?(?:tech|e|s|sc|a)|m\.?\s?(?:tech|e|s|sc|a)|ph\.?\s?d|mba|"
    r"bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?|diploma)\b",
    re.I,
)
_ROLE = re.compile(
    r"\b(?:intern|apprentice|engineer|developer|architect|analyst|scientist|"
    r"manager|lead|director|consultant|designer|administrator|specialist|"
    r"researcher|professor|officer|associate|coordinator|sdet|qa)\b",
    re.I,
)
_EXPERIENCE_SECTIONS = frozenset({"experience", "work experience", "professional experience", "employment"})


@dataclass(frozen=True)
class CoverageAnchor:
    """A trusted source span that needs one verified fact of ``kind``."""

    anchor_id: str
    kind: CoverageKind
    span_id: str
    section: str

    def prompt_payload(self) -> dict[str, str]:
        return {"anchor_id": self.anchor_id, "kind": self.kind, "span_id": self.span_id}


@dataclass(frozen=True)
class CoverageReport:
    anchors: tuple[CoverageAnchor, ...]
    covered_anchor_ids: tuple[str, ...]
    missing_anchor_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_anchor_ids


def build_coverage_anchors(spans: Iterable[SourceSpan]) -> tuple[CoverageAnchor, ...]:
    """Identify visible, safe Experience/Education records requiring accounting.

    We inspect section labels assigned by L0 and do not inspect excluded text.
    A role cue is sufficient inside a recognised Experience section: role and
    date frequently occupy separate PDF spans, so requiring both would miss
    precisely the job-record omission this control is meant to catch.
    """
    anchors: list[CoverageAnchor] = []
    experience_index = 0
    education_index = 0
    for span in spans:
        if span.visibility != "visible" or not span.normalised_text.strip():
            continue
        text = span.normalised_text
        section = span.section.casefold()
        if section == "education" and _DEGREE.search(text):
            education_index += 1
            anchors.append(CoverageAnchor(
                anchor_id=f"education-{education_index:03d}", kind="education",
                span_id=span.span_id, section=span.section,
            ))
        elif section in _EXPERIENCE_SECTIONS and _ROLE.search(text):
            experience_index += 1
            anchors.append(CoverageAnchor(
                anchor_id=f"experience-{experience_index:03d}", kind="experience",
                span_id=span.span_id, section=span.section,
            ))
    return tuple(anchors)


def assess_coverage(
    anchors: Iterable[CoverageAnchor], profile: VerifiedCandidateProfile,
) -> CoverageReport:
    """Compare anchors with facts actually eligible to cross to matching."""
    anchor_list = tuple(anchors)
    cited_by_kind: dict[str, set[str]] = {}
    for fact in profile.facts:
        cited_by_kind.setdefault(fact.kind, set()).update(span.span_id for span in fact.source_spans)
    covered = tuple(anchor.anchor_id for anchor in anchor_list if anchor.span_id in cited_by_kind.get(anchor.kind, set()))
    covered_set = set(covered)
    return CoverageReport(
        anchors=anchor_list,
        covered_anchor_ids=covered,
        missing_anchor_ids=tuple(anchor.anchor_id for anchor in anchor_list if anchor.anchor_id not in covered_set),
    )


def missing_prompt_payload(report: CoverageReport) -> list[dict[str, str]]:
    """Return PII-free retry instructions: IDs/kinds only, never resume text."""
    missing = set(report.missing_anchor_ids)
    return [anchor.prompt_payload() for anchor in report.anchors if anchor.anchor_id in missing]
