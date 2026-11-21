"""Deterministic grounding of detector observations to canonical L0 spans."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from .contracts import DetectionObservation, GroundedFinding, SourceSpan


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _matches(excerpt: str, text: str) -> bool:
    needle = _normalise(excerpt)
    return bool(needle) and needle in _normalise(text)


def ground_observation(
    observation: DetectionObservation, spans: Sequence[SourceSpan], max_window: int = 4
) -> GroundedFinding:
    """Resolve only exact evidence; never use semantic similarity to remove text."""
    if not observation.excerpt and not observation.proposed_span_ids:
        return GroundedFinding(
            observation_id=observation.observation_id, grounding_status="invalid", method="none",
            reason="observation has neither excerpt nor source span IDs",
        )
    by_id = {span.span_id: span for span in spans}
    if observation.proposed_span_ids:
        proposed = [by_id.get(span_id) for span_id in observation.proposed_span_ids]
        # Security findings may be grounded in hidden/off-page L0 spans.  Those
        # spans remain ineligible for candidate evidence and are excluded from L2.
        if all(proposed):
            joined = " ".join(span.normalised_text for span in proposed if span)
            if not observation.excerpt or _matches(observation.excerpt, joined):
                return GroundedFinding(
                    observation_id=observation.observation_id, grounding_status="grounded_exact",
                    span_ids=tuple(span.span_id for span in proposed if span), matched_text=observation.excerpt or None,
                    method="explicit_ids",
                )
    for span in spans:
        if _matches(observation.excerpt, span.normalised_text):
            return GroundedFinding(
                observation_id=observation.observation_id, grounding_status="grounded_exact",
                span_ids=(span.span_id,), matched_text=observation.excerpt, method="single_span",
            )
    for start, first in enumerate(spans):
        window: list[SourceSpan] = []
        for candidate in spans[start : start + max_window]:
            if candidate.page != first.page:
                break
            window.append(candidate)
            if len(window) > 1 and _matches(observation.excerpt, " ".join(item.normalised_text for item in window)):
                return GroundedFinding(
                    observation_id=observation.observation_id, grounding_status="grounded_adjacent_window",
                    span_ids=tuple(item.span_id for item in window), matched_text=observation.excerpt,
                    method="adjacent_window",
                )
    if observation.char_start is not None and observation.char_end is not None:
        overlapping = [
            span for span in spans
            if span.char_start is not None and span.char_end is not None
            and span.char_start < observation.char_end and span.char_end > observation.char_start
        ]
        if overlapping:
            return GroundedFinding(
                observation_id=observation.observation_id, grounding_status="grounded_exact",
                span_ids=tuple(span.span_id for span in overlapping), matched_text=observation.excerpt or None,
                method="offset",
            )
    return GroundedFinding(
        observation_id=observation.observation_id, grounding_status="ungrounded", method="none",
        reason="no exact visible L0 span or bounded adjacent-span window contains the evidence",
    )


def ground_observations(
    observations: Sequence[DetectionObservation], spans: Sequence[SourceSpan]
) -> tuple[GroundedFinding, ...]:
    return tuple(ground_observation(observation, spans) for observation in observations)
