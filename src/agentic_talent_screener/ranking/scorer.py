"""Pure, deterministic score-based ranking engine (SPEC §6.3, §13, L3/L4).

Matching is not done here. The Semantic Matching Agent links requirement options to verified
facts (``CandidateMatches``); this module checks those links against the VerifiedProfile,
takes evidence levels and quotes from the linked facts themselves, and turns them into a fit
score over the approved requirement profile only.

No I/O, no LLM calls, no global state: identical inputs always give identical outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from .config import DEFAULT_CONFIG, ScoringConfig
from .duration import explicit_duration
from .matching import MatchingAgent
from .schemas import (
    CandidateMatches,
    Criterion,
    CriterionResult,
    DurationEvidence,
    EvidenceLevel,
    Fact,
    OptionMatch,
    PoolRanking,
    Priority,
    RankingResult,
    Requirement,
    RequirementExplanation,
    RequirementProfile,
    SecurityFlag,
    Span,
    Tier,
    VerifiedProfile,
)


class ProfileNotApprovedError(ValueError):
    """Raised when asked to score against a requirement profile no human has approved."""


class MatchesMismatchError(ValueError):
    """Raised when matcher output belongs to a different candidate or profile version."""


# --------------------------------------------------------------------------------------
# Evidence grounding (L3) and flag exclusion (L4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Effective:
    level: EvidenceLevel | None  # None for education facts (no ladder)
    credit: float
    declared: EvidenceLevel | None = None
    span: Span | None = None
    adjustment: str | None = None
    grounded: bool = False


def _usable_spans(
    spans: Sequence[Span], flagged: frozenset[str], exclude: bool
) -> tuple[list[Span], list[Span]]:
    """Return (spans left after flag exclusion, of which grounded = visible with a quote)."""
    kept = [s for s in spans if not (exclude and s.location in flagged)]
    grounded = [s for s in kept if s.visible and s.quote.strip()]
    return kept, grounded


def _effective_level(
    declared: EvidenceLevel, spans: Sequence[Span], flagged: frozenset[str], exclude: bool
) -> tuple[EvidenceLevel, Span | None, str | None]:
    """Apply L3 grounding caps and L4 flag exclusion to a declared evidence level."""
    if declared == EvidenceLevel.E0:
        return EvidenceLevel.E0, None, None
    kept, grounded = _usable_spans(spans, flagged, exclude)
    if spans and not grounded:
        if len(kept) < len(spans):
            why = "L4: all cited spans are flagged and excluded"
        else:
            why = "L3: cited spans are hidden or empty"
        return EvidenceLevel.E0, None, f"{why} ({declared.name} -> E0)"
    if not spans and declared >= EvidenceLevel.E2:
        return EvidenceLevel.E1, None, f"L3: {declared.name} capped to E1 (no cited visible span)"
    return declared, grounded[0] if grounded else None, None


def _evaluate_fact(
    fact: Fact, flagged: frozenset[str], exclude: bool, cfg: ScoringConfig
) -> _Effective:
    kept, grounded = _usable_spans(fact.spans, flagged, exclude)
    if fact.kind == "education":
        # Education has no ladder: a grounded fact earns full credit and L3's level is ignored.
        if grounded:
            return _Effective(None, cfg.education_credit, span=grounded[0], grounded=True)
        if fact.spans and len(kept) < len(fact.spans):
            why = "L4: all cited spans are flagged and excluded"
        elif fact.spans:
            why = "L3: cited spans are hidden or empty"
        else:
            why = "L3: no cited visible span"
        return _Effective(None, 0.0, adjustment=why)
    level, span, adjustment = _effective_level(fact.evidence_level, fact.spans, flagged, exclude)
    return _Effective(
        level, cfg.credits.credit(level), fact.evidence_level, span, adjustment, bool(grounded)
    )


def _validate_matches(
    candidate: VerifiedProfile, profile: RequirementProfile, matches: CandidateMatches
) -> tuple[dict[str, list[OptionMatch]], int]:
    """Keep only links to options in this profile and facts in this candidate's profile."""
    if matches.candidate_id != candidate.candidate_id:
        raise MatchesMismatchError(
            f"matcher output is for {matches.candidate_id}, not {candidate.candidate_id}"
        )
    if (matches.profile_id, matches.profile_version) != (profile.profile_id, profile.version):
        raise MatchesMismatchError(
            f"matcher output is for profile {matches.profile_id} v{matches.profile_version}, "
            f"not {profile.profile_id} v{profile.version}"
        )
    options = {c.option_id for r in profile.requirements for c in r.criteria}
    facts = {f.fact_id for f in candidate.facts}
    links: dict[str, list[OptionMatch]] = {}
    dropped = 0
    for m in matches.matches:
        if m.option_id not in options or m.fact_id not in facts:
            dropped += 1
            continue
        links.setdefault(m.option_id, []).append(m)
    return links, dropped


# --------------------------------------------------------------------------------------
# Per-requirement scoring
# --------------------------------------------------------------------------------------


def _level_ok(eff: _Effective, min_evidence: EvidenceLevel | None) -> bool:
    if min_evidence is None or eff.level is None:
        return True
    return eff.level >= min_evidence


def _criterion(
    req: Requirement,
    crit: Criterion,
    is_substitute: bool,
    links: Mapping[str, list[OptionMatch]],
    facts: Mapping[str, Fact],
    flagged: frozenset[str],
    exclude: bool,
    cfg: ScoringConfig,
) -> CriterionResult:
    best: tuple[tuple[float, int], OptionMatch, _Effective] | None = None
    duration: DurationEvidence | None = None
    excluded_locations = flagged if exclude else frozenset()
    for m in links.get(crit.option_id, []):
        fact = facts[m.fact_id]
        eff = _evaluate_fact(fact, flagged, exclude, cfg)
        factor = 1.0 if m.relation == "equivalent" else cfg.adjacent_factor * m.similarity
        key = (eff.credit * factor, -1 if eff.level is None else int(eff.level))
        if best is None or key > best[0]:
            best = (key, m, eff)
        if req.min_years is not None and m.relation == "equivalent" and eff.credit > 0.0:
            # One directly linked claim must support the minimum; never sum claims.
            reported = explicit_duration(fact, req.min_years, excluded_locations)
            if reported is not None and (
                duration is None
                or (reported.lower_bound_months or 0.0) > (duration.lower_bound_months or 0.0)
            ):
                duration = reported

    base = {"option_id": crit.option_id, "text": crit.text, "is_substitute": is_substitute}
    if best is None:
        return CriterionResult(**base, credit=0.0, met=False)

    (credit, _), m, eff = best
    met = credit > 0.0 and _level_ok(eff, req.min_evidence)
    return CriterionResult(
        **base,
        credit=round(min(1.0, credit), 6),
        met=met,
        evidence_fact=m.fact_id,
        relation=m.relation,
        similarity=m.similarity,
        declared_level=eff.declared,
        effective_level=eff.level,
        quote=eff.span.quote if eff.span else None,
        location=eff.span.location if eff.span else None,
        adjustment=eff.adjustment,
        duration=duration,
    )


def _source_missing(req: Requirement, candidate: VerifiedProfile) -> str | None:
    """Name the fact kind a requirement depends on if intake did not extract it."""
    if req.kind == "experience" and req.min_years is not None and "experience" not in candidate.extracted_kinds:
        return "experience"
    if req.kind == "education" and "education" not in candidate.extracted_kinds:
        return "education"
    return None


def _evidence_gap(
    req: Requirement, credit: float, met: bool, d: CriterionResult, cfg: ScoringConfig
) -> str | None:
    if met and credit >= cfg.credits.credit(EvidenceLevel.E3):
        return None
    if d.evidence_fact is None:
        return f"no matching evidence for {d.text!r}" if len(req.criteria) > 1 else "no matching evidence"
    if d.effective_level == EvidenceLevel.E0 or (d.effective_level is None and d.adjustment):
        return "no usable evidence"
    if (
        req.min_evidence is not None
        and d.effective_level is not None
        and d.effective_level < req.min_evidence
    ):
        return f"evidence at {d.effective_level.name}, below the required {req.min_evidence.name}"
    if d.relation == "adjacent":
        return f"only adjacent evidence ({d.evidence_fact})"
    if d.effective_level == EvidenceLevel.E1:
        return "listed only (E1)"
    if d.effective_level is not None:
        return f"evidence only at {d.effective_level.name}"
    return None


@dataclass
class _Row:
    req: Requirement
    credit: float
    met: bool
    evidence_met: bool
    missing: str | None
    via_substitute: bool
    deciding: CriterionResult
    criteria: tuple[CriterionResult, ...]


def _gap(row: _Row, cfg: ScoringConfig) -> str | None:
    if row.missing:
        return f"{row.missing} facts not extracted: excluded, weight renormalised"
    gap = _evidence_gap(row.req, row.credit, row.evidence_met, row.deciding, cfg)
    if row.req.min_years is None:
        return gap
    minimum = f"{row.req.min_years:g}-year minimum"
    if row.met:
        return f"{minimum} supported as reported"
    note = f"{minimum} not established"
    return f"{gap}; {note}" if gap else note


def _requirement(
    req: Requirement,
    candidate: VerifiedProfile,
    links: Mapping[str, list[OptionMatch]],
    facts: Mapping[str, Fact],
    flagged: frozenset[str],
    exclude: bool,
    cfg: ScoringConfig,
) -> _Row:
    def run(crits: Iterable[Criterion], sub: bool) -> list[CriterionResult]:
        return [_criterion(req, c, sub, links, facts, flagged, exclude, cfg) for c in crits]

    opts, subs = run(req.options, False), run(req.substitutes, True)
    rank = lambda c: (c.credit, c.met)  # noqa: E731
    if req.combine == "any_of" or len(opts) == 1:
        deciding = max(opts, key=rank)
        credit, met = deciding.credit, deciding.met
    else:
        deciding = min(opts, key=rank)
        credit, met = sum(c.credit for c in opts) / len(opts), all(c.met for c in opts)

    via_sub = False
    if subs:
        best_sub = max(subs, key=rank)
        if (best_sub.credit, best_sub.met) > (credit, met):
            credit, met, deciding, via_sub = best_sub.credit, best_sub.met, best_sub, True

    evidence_met = met
    if req.min_years is not None:
        def qualifies(result: CriterionResult) -> bool:
            return result.met and result.duration is not None and result.duration.meets_minimum

        combine = any if req.combine == "any_of" else all
        met = combine(qualifies(c) for c in opts) or any(qualifies(c) for c in subs)

    missing = _source_missing(req, candidate) if credit == 0.0 else None
    return _Row(req, credit, met, evidence_met, missing, via_sub, deciding, tuple(opts + subs))


# --------------------------------------------------------------------------------------
# Core scoring (one pass, with or without flagged-span exclusion)
# --------------------------------------------------------------------------------------


@dataclass
class _Core:
    explanations: list[RequirementExplanation]
    point: float
    gate: float
    coverage: float | None
    notes: list[str]
    unassessed: list[str]


def _flagged_locations(flags: Iterable[SecurityFlag], cfg: ScoringConfig) -> frozenset[str]:
    return frozenset(
        location for f in flags if f.risk in cfg.exclude_flag_risks for location in f.locations
    )


def _score_pass(
    candidate: VerifiedProfile,
    profile: RequirementProfile,
    links: Mapping[str, list[OptionMatch]],
    cfg: ScoringConfig,
    exclude: bool,
) -> _Core:
    flagged = _flagged_locations(candidate.security_flags, cfg)
    facts = {f.fact_id: f for f in candidate.facts}
    rows = [
        _requirement(r, candidate, links, facts, flagged, exclude, cfg)
        for r in profile.active_requirements
    ]

    def mult(r: Requirement) -> float:
        return cfg.mandatory_multiplier if r.priority == Priority.MANDATORY else 1.0

    present = [row for row in rows if row.missing is None]
    den = sum(row.req.weight * mult(row.req) for row in present)
    num = sum(row.req.weight * mult(row.req) * row.credit for row in present)
    score = num / den if den > 0 else 0.0

    notes: list[str] = []
    for row in rows:
        if row.missing:
            notes.append(
                f"{row.req.requirement_id}: {row.missing} facts not extracted; weight renormalised"
            )

    m_weight = sum(row.req.weight for row in present if row.req.priority == Priority.MANDATORY)
    m_met = sum(
        row.req.weight for row in present if row.req.priority == Priority.MANDATORY and row.met
    )
    coverage = m_met / m_weight if m_weight > 0 else None
    gate = 1.0
    if coverage is not None and coverage < cfg.gate_threshold:
        gate = cfg.gate_floor + (1.0 - cfg.gate_floor) * (coverage / cfg.gate_threshold)
        notes.append(
            f"soft must-have gate: {coverage:.0%} of mandatory weight met "
            f"(< {cfg.gate_threshold:.0%}); score x{gate:.3f}"
        )

    explanations = [
        RequirementExplanation(
            requirement_id=row.req.requirement_id,
            statement=row.req.statement,
            kind=row.req.kind,
            priority=row.req.priority,
            weight=row.req.weight,
            effective_weight=round(row.req.weight * mult(row.req) / den, 9)
            if row.missing is None and den > 0
            else 0.0,
            missing=row.missing is not None,
            met=row.met,
            credit=round(min(1.0, row.credit), 6),
            via_substitute=row.via_substitute,
            deciding=row.deciding,
            criteria=row.criteria,
            gap=_gap(row, cfg),
            min_years=row.req.min_years,
            evidence_met=row.evidence_met if row.req.min_years is not None else None,
        )
        for row in rows
    ]

    return _Core(
        explanations=explanations,
        point=min(1.0, max(0.0, score * gate)),
        gate=gate,
        coverage=coverage,
        notes=notes,
        unassessed=[row.req.requirement_id for row in rows if row.missing],
    )


def _tier(fit: float, cfg: ScoringConfig) -> Tier:
    if fit >= cfg.tiers.a:
        return Tier.A
    return Tier.B if fit >= cfg.tiers.b else Tier.C


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def _require_approved(profile: RequirementProfile) -> None:
    if not profile.approved:
        raise ProfileNotApprovedError(
            f"requirement profile {profile.profile_id} v{profile.version} is not human-approved"
        )
    if not profile.active_requirements:
        raise ProfileNotApprovedError("requirement profile has no active requirements")


def _error_result(
    candidate_id: str,
    profile: RequirementProfile,
    cfg: ScoringConfig,
    matcher_version: str,
    msg: str,
) -> RankingResult:
    return RankingResult(
        candidate_id=candidate_id or "unknown",
        status="error",
        error=msg,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        config_version=cfg.config_version,
        matcher_version=matcher_version,
        notes=("scoring failed; candidate needs manual review",),
    )


def score_candidate(
    candidate: VerifiedProfile,
    profile: RequirementProfile,
    matches: CandidateMatches,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> RankingResult:
    """Score one candidate from the Matching Agent's links. Never raises for candidate problems."""
    _require_approved(profile)
    version = getattr(matches, "matcher_version", "unknown")
    try:
        matches = CandidateMatches.model_validate(matches)
        links, dropped = _validate_matches(candidate, profile, matches)
        final = _score_pass(candidate, profile, links, config, exclude=True)
        raw = _score_pass(candidate, profile, links, config, exclude=False)
        fit = round(final.point, 6)
        notes = list(final.notes)
        if dropped:
            notes.append(f"ignored {dropped} matcher link(s) to unknown options or facts")
        if candidate.security_flags:
            notes.append(
                f"{len(candidate.security_flags)} security flag(s); flagged spans excluded; "
                f"counterfactual delta {raw.point - final.point:+.4f}"
            )
        return RankingResult(
            candidate_id=candidate.candidate_id,
            profile_id=profile.profile_id,
            profile_version=profile.version,
            config_version=config.config_version,
            matcher_version=matches.matcher_version,
            fit_score=fit,
            raw_score=round(raw.point, 6),
            counterfactual_delta=round(raw.point - final.point, 6),
            tier=_tier(fit, config),
            mandatory_coverage=final.coverage,
            mandatory_gate_multiplier=round(final.gate, 6),
            unassessed=tuple(final.unassessed),
            explanations=tuple(final.explanations),
            security_flags=candidate.security_flags,
            intake_status=candidate.intake_status,
            overall_risk=candidate.overall_risk,
            notes=tuple(notes),
        )
    except Exception as exc:  # noqa: BLE001 - a malformed candidate must not crash the pool
        return _error_result(
            candidate.candidate_id, profile, config, str(version), f"{type(exc).__name__}: {exc}"
        )


def score_pool(
    candidates: Iterable[VerifiedProfile | Mapping[str, object]],
    profile: RequirementProfile,
    matcher: MatchingAgent,
    config: ScoringConfig = DEFAULT_CONFIG,
    shortlist_size: int | None = None,
) -> PoolRanking:
    """Match, score and rank a whole pool against one approved profile version.

    With ``shortlist_size`` N, the top N are marked ``shortlisted`` and the next
    ``config.near_cutoff_count`` are marked ``near_cutoff`` so HR can see who just missed out.
    """
    _require_approved(profile)
    if shortlist_size is not None and shortlist_size < 1:
        raise ValueError("shortlist_size must be at least 1")
    version = matcher.version

    results: list[RankingResult] = []
    for item in candidates:
        cid = str(item.get("candidate_id", "")) if isinstance(item, Mapping) else ""
        try:
            candidate = item if isinstance(item, VerifiedProfile) else VerifiedProfile.model_validate(item)
        except ValidationError as exc:
            msg = f"invalid candidate: {exc.error_count()} error(s): {exc.errors()[0]['msg']}"
            results.append(_error_result(cid, profile, config, version, msg))
            continue
        except Exception as exc:  # noqa: BLE001
            results.append(_error_result(cid, profile, config, version, f"{type(exc).__name__}: {exc}"))
            continue
        try:
            matches = matcher.match(candidate.matcher_view(), profile)
        except Exception as exc:  # noqa: BLE001 - no silent degradation: route to manual review
            results.append(
                _error_result(
                    candidate.candidate_id, profile, config, version,
                    f"matching failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(score_candidate(candidate, profile, matches, config))

    # Equal scores: more of the must-have weight met first, then blind ID for determinism.
    ok = sorted(
        (r for r in results if r.status == "ok"),
        key=lambda r: (-r.fit_score, -(r.mandatory_coverage or 0.0), r.candidate_id),
    )
    errs = [r for r in results if r.status == "error"]
    if shortlist_size is not None:  # errors need manual review; they are never shortlisted
        errs = [r.model_copy(update={"shortlisted": False}) for r in errs]

    ranked: list[RankingResult] = []
    for i, r in enumerate(ok):
        update: dict[str, object] = {"rank": i + 1}
        if shortlist_size is not None:
            update["shortlisted"] = i < shortlist_size
            update["near_cutoff"] = shortlist_size <= i < shortlist_size + config.near_cutoff_count
        ranked.append(r.model_copy(update=update))

    final = ranked + errs
    return PoolRanking(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        config_version=config.config_version,
        matcher_version=version,
        shortlist_size=shortlist_size,
        results=tuple(final),
        n_scored=len(ok),
        n_errors=len(errs),
        n_flagged=sum(1 for r in final if r.security_flags),
    )
