"""Pure, deterministic score-based ranking engine (SPEC §6.4, L3/L4).

No I/O, no LLM calls, no global state. Randomness is a seeded NumPy generator keyed on the
config seed and the blind candidate ID, so identical inputs always give identical outputs.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from .config import DEFAULT_CONFIG, ScoringConfig
from .schemas import (
    ComponentScore,
    EvidenceLevel,
    PoolRanking,
    RankingResult,
    RequirementExplanation,
    RequirementProfile,
    RequirementTag,
    SecurityFlag,
    SkillCatalogue,
    SkillClaim,
    Span,
    Tier,
    VerifiedProfile,
)

COMPONENT_ORDER = ("skill_match", "experience", "projects", "education", "tenure", "location")


class ProfileNotApprovedError(ValueError):
    """Raised when asked to score against a requirement profile no human has approved."""


# --------------------------------------------------------------------------------------
# Evidence grounding (L3) and flag exclusion (L4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Effective:
    level: EvidenceLevel
    span: Span | None = None
    adjustment: str | None = None


def _usable_spans(
    spans: Sequence[Span], flagged: frozenset[str], exclude: bool
) -> tuple[list[Span], list[Span]]:
    """Return (spans left after flag exclusion, of which grounded = visible with a quote)."""
    kept = [s for s in spans if not (exclude and s.location in flagged)]
    grounded = [s for s in kept if s.visible and s.quote.strip()]
    return kept, grounded


def _effective_level(
    declared: EvidenceLevel, spans: Sequence[Span], flagged: frozenset[str], exclude: bool
) -> _Effective:
    """Apply L3 grounding caps and L4 flag exclusion to a declared evidence level."""
    if declared == EvidenceLevel.E0:
        return _Effective(EvidenceLevel.E0)
    kept, grounded = _usable_spans(spans, flagged, exclude)
    if spans and not grounded:
        if len(kept) < len(spans):
            why = "L4: all cited spans are flagged and excluded"
        else:
            why = "L3: cited spans are hidden or empty"
        return _Effective(EvidenceLevel.E0, None, f"{why} ({declared.name} -> E0)")
    if not spans and declared >= EvidenceLevel.E2:
        return _Effective(
            EvidenceLevel.E1, None, f"L3: {declared.name} capped to E1 (no cited visible span)"
        )
    return _Effective(declared, grounded[0] if grounded else None)


def _best_claims(
    claims: Iterable[SkillClaim], flagged: frozenset[str], exclude: bool
) -> dict[str, tuple[_Effective, EvidenceLevel]]:
    """Per skill: best effective level (and the declared level of that claim)."""
    out: dict[str, tuple[_Effective, EvidenceLevel]] = {}
    for c in claims:
        eff = _effective_level(c.evidence_level, c.spans, flagged, exclude)
        cur = out.get(c.skill_id)
        if cur is None or eff.level > cur[0].level:
            out[c.skill_id] = (eff, c.evidence_level)
    return out


# --------------------------------------------------------------------------------------
# Core scoring (one pass, with or without flagged-span exclusion)
# --------------------------------------------------------------------------------------


@dataclass
class _Core:
    components: list[ComponentScore]
    explanations: list[RequirementExplanation]
    point: float
    gate: float
    coverage: float | None
    notes: list[str]
    score_by_name: dict[str, float | None]


def _flagged_locations(flags: Iterable[SecurityFlag], cfg: ScoringConfig) -> frozenset[str]:
    return frozenset(f.location for f in flags if f.risk in cfg.exclude_flag_risks)


def _skill_match(
    candidate: VerifiedProfile,
    profile: RequirementProfile,
    catalogue: SkillCatalogue,
    cfg: ScoringConfig,
    flagged: frozenset[str],
    exclude: bool,
) -> tuple[float, float | None, list[RequirementExplanation]]:
    best = _best_claims(candidate.claims, flagged, exclude)
    edges: dict[str, list[tuple[str, float]]] = {}
    for e in catalogue.adjacency:
        edges.setdefault(e.counts_towards, []).append((e.has_skill, e.similarity))

    num = den = 0.0
    m_weight = m_covered = 0.0
    explanations: list[RequirementExplanation] = []

    for req in profile.active_requirements:
        mult = cfg.mandatory_multiplier if req.tag == RequirementTag.MANDATORY else 1.0
        eff, declared = best.get(req.skill_id, (_Effective(EvidenceLevel.E0), EvidenceLevel.E0))
        direct = cfg.credits.credit(eff.level)

        adj_credit, adj_skill, adj_level = 0.0, None, EvidenceLevel.E0
        for other, sim in edges.get(req.skill_id, []):
            o = best.get(other)
            if o is None:
                continue
            c = cfg.adjacent_factor * sim * cfg.credits.credit(o[0].level)
            if c > adj_credit:
                adj_credit, adj_skill, adj_level = c, other, o[0].level

        credit = max(direct, adj_credit)
        via = adj_skill if adj_credit > direct else None
        num += req.weight * mult * credit
        den += req.weight * mult

        if req.tag == RequirementTag.MANDATORY:
            m_weight += req.weight
            level_for_gate = adj_level if via else eff.level
            if level_for_gate >= cfg.gate_min_level:
                m_covered += req.weight

        if credit >= cfg.credits.credit(EvidenceLevel.E3):
            gap = None
        elif via:
            gap = f"only adjacent evidence via {via}"
        elif eff.level == EvidenceLevel.E0:
            gap = "no usable evidence"
        elif eff.level == EvidenceLevel.E1:
            gap = "listed only (E1)"
        else:
            gap = f"evidence only at {eff.level.name}"

        explanations.append(
            RequirementExplanation(
                skill_id=req.skill_id,
                label=req.label,
                tag=req.tag,
                declared_level=declared,
                effective_level=eff.level,
                credit=round(min(1.0, credit), 6),
                via_adjacent_skill=via,
                quote=eff.span.quote if eff.span else None,
                location=eff.span.location if eff.span else None,
                adjustment=eff.adjustment,
                gap=gap,
            )
        )

    score = num / den if den > 0 else 0.0
    coverage = (m_covered / m_weight) if m_weight > 0 else None
    return min(1.0, max(0.0, score)), coverage, explanations


def _relevance(skill_ids: Iterable[str], req_ids: frozenset[str]) -> float:
    """Share of (up to) a third of the requirements that this item touches, capped at 1."""
    overlap = len(req_ids & set(skill_ids))
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / max(1, math.ceil(len(req_ids) / 3)))


def _experience(
    candidate: VerifiedProfile,
    req_ids: frozenset[str],
    cfg: ScoringConfig,
    flagged: frozenset[str],
    exclude: bool,
) -> float | None:
    if candidate.roles is None:
        return None
    months = 0.0
    for r in candidate.roles:
        _, grounded = _usable_spans(r.spans, flagged, exclude)
        if not grounded:
            continue  # L3: a role without a cited visible span does not count
        factor = cfg.internship_month_factor if r.kind == "internship" else 1.0
        months += r.months * factor * _relevance(r.skill_ids, req_ids)
    return min(1.0, (months / 12.0) / cfg.experience_target_years)


def _projects(
    candidate: VerifiedProfile,
    req_ids: frozenset[str],
    cfg: ScoringConfig,
    flagged: frozenset[str],
    exclude: bool,
) -> float | None:
    if candidate.projects is None:
        return None
    total = 0.0
    for p in candidate.projects:
        eff = _effective_level(p.evidence_level, p.spans, flagged, exclude)
        total += cfg.credits.credit(eff.level) * _relevance(p.skill_ids, req_ids)
    return min(1.0, total / cfg.projects_target)


def _education(candidate: VerifiedProfile, cfg: ScoringConfig) -> float | None:
    edu = candidate.education
    if edu is None or edu.highest_level is None:
        return None
    level = cfg.education_level_scores.get(edu.highest_level)
    if level is None:
        return None
    if edu.field_relevance is None:
        return level
    b = cfg.education_field_blend
    return (1.0 - b) * level + b * edu.field_relevance


def _tenure(candidate: VerifiedProfile, cfg: ScoringConfig) -> tuple[float | None, str]:
    """Fairness-bounded: internships and gaps are ignored, <3 job roles => not scored."""
    if candidate.roles is None:
        return None, "roles not extracted"
    jobs = [r for r in candidate.roles if r.kind == "job"]
    if len(jobs) < cfg.tenure_min_job_roles:
        return None, f"fewer than {cfg.tenure_min_job_roles} job roles: tenure not scored"
    avg = sum(r.months for r in jobs) / len(jobs)
    frac = min(1.0, avg / cfg.tenure_full_months)
    return cfg.tenure_floor + (1.0 - cfg.tenure_floor) * frac, ""


def _score_pass(
    candidate: VerifiedProfile,
    profile: RequirementProfile,
    catalogue: SkillCatalogue,
    cfg: ScoringConfig,
    exclude: bool,
) -> _Core:
    flagged = _flagged_locations(candidate.security_flags, cfg)
    req_ids = frozenset(r.skill_id for r in profile.active_requirements)

    skill, coverage, explanations = _skill_match(
        candidate, profile, catalogue, cfg, flagged, exclude
    )
    tenure, tenure_note = _tenure(candidate, cfg)
    raw_scores: dict[str, float | None] = {
        "skill_match": skill,
        "experience": _experience(candidate, req_ids, cfg, flagged, exclude),
        "projects": _projects(candidate, req_ids, cfg, flagged, exclude),
        "education": _education(candidate, cfg),
        "tenure": tenure,
        "location": candidate.location_match,
    }

    nominal = cfg.weights.as_dict()
    present_total = sum(nominal[n] for n in COMPONENT_ORDER if raw_scores[n] is not None)
    notes: list[str] = []
    components: list[ComponentScore] = []
    point = 0.0
    for name in COMPONENT_ORDER:
        s = raw_scores[name]
        present = s is not None
        eff_w = nominal[name] / present_total if present else 0.0
        if present:
            point += eff_w * s  # type: ignore[operator]
        note = ""
        if not present:
            note = tenure_note if name == "tenure" else "missing: weight renormalised"
            notes.append(f"{name}: {note}")
        components.append(
            ComponentScore(
                name=name,  # type: ignore[arg-type]
                present=present,
                score=None if s is None else round(min(1.0, max(0.0, s)), 6),
                nominal_weight=nominal[name],
                effective_weight=round(eff_w, 9),
                note=note,
            )
        )

    gate = 1.0
    if coverage is not None and coverage < cfg.gate_threshold:
        gate = cfg.gate_floor + (1.0 - cfg.gate_floor) * (coverage / cfg.gate_threshold)
        notes.append(
            f"soft must-have gate: {coverage:.0%} of mandatory weight covered at "
            f"{cfg.gate_min_level.name}+ (< {cfg.gate_threshold:.0%}); score x{gate:.3f}"
        )

    return _Core(
        components=components,
        explanations=explanations,
        point=min(1.0, max(0.0, point * gate)),
        gate=gate,
        coverage=coverage,
        notes=notes,
        score_by_name=raw_scores,
    )


# --------------------------------------------------------------------------------------
# Uncertainty: seeded Monte Carlo CI
# --------------------------------------------------------------------------------------


def _confidence_interval(
    core: _Core, candidate_id: str, cfg: ScoringConfig
) -> tuple[float, float]:
    rng = np.random.default_rng([cfg.mc_seed, zlib.crc32(candidate_id.encode())])
    nominal = np.array([cfg.weights.as_dict()[n] for n in COMPONENT_ORDER])
    scores = np.array(
        [np.nan if core.score_by_name[n] is None else core.score_by_name[n] for n in COMPONENT_ORDER],
        dtype=float,
    )
    present = ~np.isnan(scores)
    base = float(np.dot(nominal[present], scores[present]) / nominal[present].sum())

    draws = np.empty((cfg.mc_runs, len(COMPONENT_ORDER)))
    draws[:, present] = scores[present] + rng.normal(
        0.0, cfg.mc_component_sd, (cfg.mc_runs, int(present.sum()))
    )
    # A missing component widens the CI: sample it around the present-component average.
    draws[:, ~present] = base + rng.normal(
        0.0, cfg.mc_missing_sd, (cfg.mc_runs, int((~present).sum()))
    )
    draws = np.clip(draws, 0.0, 1.0)
    runs = np.clip(core.gate * draws @ nominal, 0.0, 1.0)
    lo, hi = np.percentile(runs, [2.5, 97.5])
    return min(float(lo), core.point), max(float(hi), core.point)


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
    candidate_id: str, profile: RequirementProfile, cfg: ScoringConfig, cat: SkillCatalogue, msg: str
) -> RankingResult:
    return RankingResult(
        candidate_id=candidate_id or "unknown",
        status="error",
        error=msg,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        config_version=cfg.config_version,
        catalogue_version=cat.version,
        notes=("scoring failed; candidate needs manual review",),
    )


def score_candidate(
    candidate: VerifiedProfile,
    profile: RequirementProfile,
    catalogue: SkillCatalogue | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> RankingResult:
    """Score one candidate. Never raises for candidate-specific problems."""
    cat = catalogue or SkillCatalogue(version="none")
    _require_approved(profile)
    try:
        final = _score_pass(candidate, profile, cat, config, exclude=True)
        raw = _score_pass(candidate, profile, cat, config, exclude=False)
        lo, hi = _confidence_interval(final, candidate.candidate_id, config)
        fit = round(final.point, 6)
        notes = list(final.notes)
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
            catalogue_version=cat.version,
            fit_score=fit,
            ci_low=round(lo, 6),
            ci_high=round(hi, 6),
            raw_score=round(raw.point, 6),
            counterfactual_delta=round(raw.point - final.point, 6),
            tier=_tier(fit, config),
            mandatory_coverage=final.coverage,
            mandatory_gate_multiplier=round(final.gate, 6),
            components=tuple(final.components),
            explanations=tuple(final.explanations),
            security_flags=candidate.security_flags,
            notes=tuple(notes),
        )
    except Exception as exc:  # noqa: BLE001 - a malformed candidate must not crash the pool
        return _error_result(
            candidate.candidate_id, profile, config, cat, f"{type(exc).__name__}: {exc}"
        )


def _tie_bands(ok: list[RankingResult]) -> dict[str, tuple[int | None, tuple[str, ...]]]:
    """Union candidates whose 95% CIs overlap into tie bands (connected components)."""
    parent = {r.candidate_id: r.candidate_id for r in ok}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    overlaps: dict[str, list[str]] = {r.candidate_id: [] for r in ok}
    for i, a in enumerate(ok):
        for b in ok[i + 1 :]:
            if a.ci_low <= b.ci_high and b.ci_low <= a.ci_high:
                overlaps[a.candidate_id].append(b.candidate_id)
                overlaps[b.candidate_id].append(a.candidate_id)
                parent[find(a.candidate_id)] = find(b.candidate_id)

    band_ids: dict[str, int] = {}
    out: dict[str, tuple[int | None, tuple[str, ...]]] = {}
    for r in ok:  # ok is sorted, so band numbers follow rank order
        others = tuple(sorted(overlaps[r.candidate_id]))
        if not others:
            out[r.candidate_id] = (None, ())
            continue
        root = find(r.candidate_id)
        band_ids.setdefault(root, len(band_ids) + 1)
        out[r.candidate_id] = (band_ids[root], others)
    return out


def score_pool(
    candidates: Iterable[VerifiedProfile | Mapping[str, object]],
    profile: RequirementProfile,
    catalogue: SkillCatalogue | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> PoolRanking:
    """Score and rank a whole pool against one approved profile version."""
    cat = catalogue or SkillCatalogue(version="none")
    _require_approved(profile)

    results: list[RankingResult] = []
    for item in candidates:
        if isinstance(item, VerifiedProfile):
            results.append(score_candidate(item, profile, cat, config))
            continue
        cid = str(item.get("candidate_id", "")) if isinstance(item, Mapping) else ""
        try:
            results.append(score_candidate(VerifiedProfile.model_validate(item), profile, cat, config))
        except ValidationError as exc:
            results.append(_error_result(cid, profile, config, cat, f"invalid candidate: {exc.error_count()} error(s): {exc.errors()[0]['msg']}"))
        except Exception as exc:  # noqa: BLE001
            results.append(_error_result(cid, profile, config, cat, f"{type(exc).__name__}: {exc}"))

    ok = sorted((r for r in results if r.status == "ok"), key=lambda r: (-r.fit_score, r.candidate_id))
    errs = [r for r in results if r.status == "error"]
    bands = _tie_bands(ok)
    n = len(ok)

    ranked: list[RankingResult] = []
    for i, r in enumerate(ok):
        below = sum(1 for o in ok if o.fit_score < r.fit_score)
        pct = 100.0 * below / (n - 1) if n > 1 else 100.0
        band, with_ = bands[r.candidate_id]
        ranked.append(
            r.model_copy(
                update={"rank": i + 1, "percentile": round(pct, 2), "tie_band": band, "tie_with": with_}
            )
        )

    final = ranked + errs
    return PoolRanking(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        config_version=config.config_version,
        catalogue_version=cat.version,
        results=tuple(final),
        n_scored=len(ok),
        n_errors=len(errs),
        n_flagged=sum(1 for r in final if r.security_flags),
    )
