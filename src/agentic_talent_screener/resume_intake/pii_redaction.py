"""Create the contact-PII-minimised span view permitted in model prompts.

The canonical L0 spans remain unchanged for deterministic processing and an
authorised audit.  This module creates a separate representation for every
LLM boundary; it never removes or alters the canonical evidence record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from .contracts import SourceSpan


RedactionCategory = Literal["email", "phone", "location", "link"]


@dataclass(frozen=True)
class PIIRedaction:
    """Audit metadata for a replacement, without retaining its original value."""

    span_id: str | None
    category: RedactionCategory
    char_start: int
    char_end: int
    replacement: str


_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
_PHONE_CANDIDATE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{5,}\d)(?!\w)")
_YEAR_RANGE = re.compile(r"^\d{4}\s*[-–]\s*\d{4}$")
_LOCATION_LABEL = re.compile(
    r"\b(?:location|address|based\s+in|city|residence)\s*[:\-]\s*[^\n|•]+",
    re.IGNORECASE,
)
_STREET_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9 .'-]{2,50}\s"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?)\b",
    re.IGNORECASE,
)
_CITY_REGION = re.compile(
    r"\b[A-Z][A-Za-z .'-]{1,40},\s*(?:[A-Z]{2}|"
    r"India|USA|United States|United Kingdom|UK|Canada|Australia|Singapore|Germany|France)\b"
)
_SUBSTANTIVE_SECTIONS = frozenset({"experience", "project", "education", "certification", "skills", "summary"})

_PLACEHOLDER: dict[RedactionCategory, str] = {
    "email": "[EMAIL_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "location": "[LOCATION_REDACTED]",
    "link": "[LINK_REDACTED]",
}


def _phone_matches(text: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for match in _PHONE_CANDIDATE.finditer(text):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if not 7 <= len(digits) <= 15 or _YEAR_RANGE.fullmatch(candidate.strip()):
            continue
        matches.append((match.start(), match.end()))
    return matches


def _replacement_ranges(text: str, header_context: bool) -> list[tuple[int, int, RedactionCategory]]:
    ranges: list[tuple[int, int, RedactionCategory]] = []
    for category, pattern in (("email", _EMAIL), ("link", _URL)):
        ranges.extend((match.start(), match.end(), category) for match in pattern.finditer(text))
    ranges.extend((start, end, "phone") for start, end in _phone_matches(text))
    if header_context:
        for pattern in (_LOCATION_LABEL, _STREET_ADDRESS, _CITY_REGION):
            ranges.extend((match.start(), match.end(), "location") for match in pattern.finditer(text))

    # Prefer the widest redaction when patterns overlap; then apply replacements
    # right-to-left so stored offsets remain offsets into the original text.
    ordered = sorted(ranges, key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, RedactionCategory]] = []
    for start, end, category in ordered:
        if any(start < kept_end and end > kept_start for kept_start, kept_end, _ in selected):
            continue
        selected.append((start, end, category))
    return selected


def redact_text(text: str, *, header_context: bool = False, span_id: str | None = None) -> tuple[str, tuple[PIIRedaction, ...]]:
    """Replace contact PII with typed tokens and return non-sensitive metadata."""
    redactions = tuple(
        PIIRedaction(
            span_id=span_id,
            category=category,
            char_start=start,
            char_end=end,
            replacement=_PLACEHOLDER[category],
        )
        for start, end, category in _replacement_ranges(text, header_context)
    )
    redacted = text
    for item in reversed(redactions):
        redacted = redacted[:item.char_start] + item.replacement + redacted[item.char_end:]
    return redacted, redactions


def redact_source_spans(spans: Sequence[SourceSpan]) -> tuple[tuple[SourceSpan, ...], tuple[PIIRedaction, ...]]:
    """Return a model-safe copy of canonical spans, retaining their stable IDs."""
    safe_spans: list[SourceSpan] = []
    all_redactions: list[PIIRedaction] = []
    header_context = True
    for span in spans:
        section = span.section.casefold()
        if section in _SUBSTANTIVE_SECTIONS:
            header_context = False
        normalised_text, redactions = redact_text(
            span.normalised_text, header_context=header_context, span_id=span.span_id
        )
        text, _ = redact_text(span.text, header_context=header_context, span_id=span.span_id)
        visible_text, _ = redact_text(span.visible_text, header_context=header_context, span_id=span.span_id)
        adjacent_before, _ = redact_text(span.adjacent_before or "", header_context=False, span_id=span.span_id)
        adjacent_after, _ = redact_text(span.adjacent_after or "", header_context=False, span_id=span.span_id)
        safe_spans.append(span.model_copy(update={
            "text": text,
            "normalised_text": normalised_text,
            "visible_text": visible_text,
            "adjacent_before": adjacent_before or None,
            "adjacent_after": adjacent_after or None,
        }))
        all_redactions.extend(redactions)
    return tuple(safe_spans), tuple(all_redactions)
