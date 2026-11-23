"""E7 detection metrics: TPR on twins, FPR on clean resumes, per-layer attribution.

What counts as "detected" (``DEFAULT_DETECTED``), from what ``SystemOutput`` exposes today:

* ``quarantined``: ``unranked`` reason ``intake:quarantined@<stage>`` (or ``quarantine``).
* ``manual_review``: ``intake:manual_review@<stage>``.
* ``flagged``: the candidate was still scored but the pipeline raised a flag. ``SystemOutput`` has no
  field for this yet (**C2/E5 need:** ``SystemOutput.flags: dict[candidate_id, list[layer]]``), so
  flags are read from ``getattr(output, "flags", None)`` or an explicit ``flags`` argument.

Not detection: ``unranked_other`` (e.g. ``intake:l2_extracted``, no verified profile) and ``error``
(an exception string or ``scoring_error``). Errors are counted separately and never enter TPR or FPR.
Guard refusals and cache misses are infrastructure failures and abort (``InfrastructureError``).
Baselines with no defence (keyword, TF-IDF, BM25, naive LLM) can never detect; their TPR is 0 by
construction, not a finding.

The catching layer is the pipeline stage in the reason string (L0, L1, L2, L3). L1a and L1b share one
stage, so ``L1`` matches expected ``L1``, ``L1a`` and ``L1b``. A flag-only catch carries the layers
given in ``flags``. Where no layer is exposed it is ``None`` and attribution reports it as unknown.

Verdicts follow E12: each carries ``n`` and ``ci95`` (Wilson); below ``min_n`` the status reads
``not measurable (n<min)`` unless failure is already proven by the interval.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..systems import InfrastructureError, SystemOutput
from .efficiency import NOT_MEASURABLE, _judge_upper_target, wilson

DEFAULT_MIN_N = 30
FPR_TARGET = 0.03
FPR_MIN_N = 100  # architecture 4.3: a 3% FPR claim needs >= 100 clean resumes (Wilson needs ~130 for 0 misses)
DEFAULT_DETECTED = frozenset({"quarantined", "manual_review", "flagged"})
_REASON = re.compile(r"^intake:(?P<status>[a-z0-9_]+)@(?P<stage>[A-Z0-9_]+)$")
_STAGE_LAYER = {
    "L0_DOC_FORENSICS": "L0",
    "L1_SECURITY_AGENT": "L1",
    "L2_RESUME_AGENT": "L2",
    "L3_EVIDENCE_VERIFIER": "L3",
}
_INFRA_PREFIXES = ("BudgetExceeded", "CacheMiss", "InfrastructureError")
_L1 = {"L1", "L1a", "L1b"}


@dataclass(frozen=True)
class Detection:
    """What the pipeline did with one resume. ``kind`` is one of passed, quarantined, manual_review,
    flagged, unranked_other, error."""

    kind: str
    layer: str | None = None
    detail: str = ""


def classify(reason: str | None, flagged_layers: Sequence[str] | None = None) -> Detection:
    """Map an ``unranked`` reason (None when scored) and optional flag layers to a ``Detection``."""
    if reason is None:
        if flagged_layers is not None:
            return Detection("flagged", flagged_layers[0] if flagged_layers else None,
                             ",".join(flagged_layers))
        return Detection("passed")
    if reason.startswith(_INFRA_PREFIXES):
        raise InfrastructureError(reason)
    m = _REASON.match(reason)
    if m is None:
        return Detection("error", detail=reason)
    status, layer = m["status"], _STAGE_LAYER.get(m["stage"], m["stage"])
    if status in ("quarantined", "quarantine"):
        return Detection("quarantined", layer, reason)
    if status == "manual_review":
        return Detection("manual_review", layer, reason)
    return Detection("unranked_other", layer, reason)


def classify_output(output: SystemOutput, candidate_id: str,
                    flags: Mapping[str, Sequence[str]] | None = None) -> Detection:
    """Classify one candidate of a ``SystemOutput`` (reason first, then flags from ``flags`` or ``output.flags``)."""
    flag_map = flags if flags is not None else getattr(output, "flags", None) or {}
    reason = output.unranked.get(candidate_id)
    if reason is None and candidate_id not in output.scores:
        return Detection("error", detail="candidate missing from system output")
    return classify(reason, flag_map[candidate_id] if candidate_id in flag_map else None)


def layer_matches(caught: str | None, expected: Iterable[str]) -> bool:
    """True if the catching layer is one of the labelled expected layers (L1 matches L1/L1a/L1b)."""
    if caught is None:
        return False
    exp = set(expected)
    return caught in exp or (caught in _L1 and bool(exp & _L1))


def rate(k: int, n: int, min_n: int = DEFAULT_MIN_N) -> dict[str, Any]:
    """A proportion with n, Wilson ci95 and an honest status (``not measurable (n<min)`` below ``min_n``)."""
    lo, hi = wilson(k, n)
    return {"value": (k / n) if n else None, "k": k, "n": n, "ci95": [lo, hi], "min_n": min_n,
            "status": "ok" if n >= min_n else f"{NOT_MEASURABLE} (n<min)"}


def clean_detections(output: SystemOutput, candidate_ids: Iterable[str],
                     flags: Mapping[str, Sequence[str]] | None = None) -> dict[str, Detection]:
    """Detection per clean resume of one run."""
    return {cid: classify_output(output, cid, flags) for cid in candidate_ids}


def fpr_verdict(k: int, n: int, *, target: float = FPR_TARGET, min_n: int = DEFAULT_MIN_N,
                fpr_min_n: int = FPR_MIN_N) -> dict[str, Any]:
    """FPR <= target verdict. Pass needs n >= max(min_n, fpr_min_n) and the Wilson upper bound below the
    target; a lower bound at or above the target proves failure at any n."""
    lo, hi = wilson(k, n)
    need = max(min_n, fpr_min_n)
    status = _judge_upper_target(lo, hi, target, n, need)
    return {"status": status, "value": (k / n) if n else None, "k": k, "n": n, "ci95": [lo, hi],
            "target": target, "min_n": need, "ci95_of": "Wilson interval"}


def summarize_detection(
    twin_rows: Sequence[Any],
    clean: Mapping[str, Detection],
    *,
    system: str,
    min_n: int = DEFAULT_MIN_N,
    detected: frozenset[str] = DEFAULT_DETECTED,
    fpr_target: float = FPR_TARGET,
    fpr_min_n: int = FPR_MIN_N,
) -> dict[str, Any]:
    """Detection summary for ONE system.

    ``twin_rows`` are ``robustness.PairedOutcome`` objects (duck-typed: ``system``, ``status``,
    ``detection`` (a ``Detection``), ``family``, ``channel``, ``visibility``, ``expected_layers``).
    Rows with status ``not_reachable``, ``no_input``, ``error`` and ``base_unranked`` are reported as
    counts and excluded from TPR; a ``not_reachable`` twin is never a miss.
    """
    if not system:
        raise ValueError("system is required: never summarize several systems together")
    rows = [r for r in twin_rows if r.system == system]
    skip = ("not_reachable", "no_input", "error", "base_unranked")
    excluded = Counter(r.status for r in rows if r.status in skip)
    used = [r for r in rows if r.status not in skip]

    def is_det(r: Any) -> bool:
        return r.detection.kind in detected

    def group(key: str) -> dict[str, dict[str, Any]]:
        g: dict[str, list[Any]] = defaultdict(list)
        for r in used:
            g[str(getattr(r, key))].append(r)
        return {k: rate(sum(map(is_det, v)), len(v), min_n) for k, v in sorted(g.items())}

    hits = [r for r in used if is_det(r)]
    caught = Counter(r.detection.layer or "unknown" for r in hits)
    with_layer = [r for r in hits if r.detection.layer is not None]
    matched = sum(layer_matches(r.detection.layer, r.expected_layers) for r in with_layer)
    by_family_layer: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in hits:
        by_family_layer[r.family][r.detection.layer or "unknown"] += 1

    clean_ok = [d for d in clean.values() if d.kind != "error"]
    fp = sum(1 for d in clean_ok if d.kind in detected)
    return {
        "system": system,
        "n_twins": len(rows),
        "excluded": dict(excluded),
        "off_page_note": "not_reachable twins are reported as not reachable, never as a miss",
        "tpr": {
            "overall": rate(len(hits), len(used), min_n),
            "by_family": group("family"), "by_channel": group("channel"), "by_visibility": group("visibility"),
            "detected_kinds": sorted(detected),
            "outcome_counts": dict(Counter(r.detection.kind for r in used)),
        },
        "attribution": {
            "caught_by_layer": dict(caught),
            "caught_by_family_and_layer": {f: dict(v) for f, v in sorted(by_family_layer.items())},
            "layer_known": rate(len(with_layer), len(hits), min_n),
            "caught_at_expected_layer": rate(matched, len(with_layer), min_n),
            "note": "L1a and L1b share one pipeline stage (reported as L1); flagged-but-scored needs SystemOutput.flags",
        },
        "clean": {
            "n_errors_excluded": len(clean) - len(clean_ok),
            "outcome_counts": dict(Counter(d.kind for d in clean_ok)),
            "fpr": fpr_verdict(fp, len(clean_ok), target=fpr_target, min_n=min_n, fpr_min_n=fpr_min_n),
        },
    }
