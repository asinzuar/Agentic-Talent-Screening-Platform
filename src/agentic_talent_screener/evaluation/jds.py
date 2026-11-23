"""Synthetic job descriptions and ranking ground truth for the core suite (E14).

Ground truth is **synthetic labels by construction**, not human grades: a candidate's grade for a JD
is computed from the candidate's gold skill facts (known because the resume text was generated from
them, see ``datasets/synthetic.py``) against the JD's skill requirements. Every report that uses these
labels must say so. They follow the generator's evidence ladder (E1-E4), so a system that reads the
ladder the same way is advantaged by construction; this is a pipeline-consistency check, not evidence
about real hiring outcomes (open question Q30).

Everything is deterministic for ``(seed, index)`` and offline.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from agentic_talent_screener.ranking.schemas import (
    Criterion,
    EvidenceLevel,
    Priority,
    Requirement,
    RequirementProfile,
)

from .datasets.schemas import GoldFact
from .datasets.synthetic import _DOMAINS

LABEL_NOTE = (
    "Ranking labels are SYNTHETIC: grades 0-5 computed from gold facts x JD requirements by construction "
    "(not human-graded). They follow the generator's own evidence ladder."
)
#: share of a requirement's weight earned by a gold skill at each evidence level
LEVEL_CREDIT = {"E1": 0.25, "E2": 0.5, "E3": 0.75, "E4": 1.0}
MANDATORY, PREFERRED = 4, 2  # requirements per JD
MAX_JDS = 60


def synthetic_jds(n: int = 10, seed: int = 0) -> list[RequirementProfile]:
    """``n`` approved profiles, cycling through the generator's domains; each has 4 mandatory and
    2 preferred skill requirements drawn from the domain's skill list."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_JDS:
        raise ValueError(f"n_jds must be an integer in [1, {MAX_JDS}]")
    domains = sorted(_DOMAINS)
    out: list[RequirementProfile] = []
    for j in range(n):
        domain = domains[j % len(domains)]
        titles, skills = _DOMAINS[domain]
        rng = random.Random(f"jd:{seed}:{j}")
        chosen = rng.sample(skills, MANDATORY + PREFERRED)
        reqs: list[Requirement] = []
        for i, skill in enumerate(chosen):
            mandatory = i < MANDATORY
            rid = f"{'M' if mandatory else 'P'}{i + 1}"
            reqs.append(Requirement(
                requirement_id=rid, statement=skill, kind="skill",
                options=(Criterion(option_id=f"{rid}.a", text=skill),),
                min_evidence=EvidenceLevel.E2,
                priority=Priority.MANDATORY if mandatory else Priority.PREFERRED,
                weight=1.0 if mandatory else 0.5,
                provenance="jd_qualification", quote=f"Experience with {skill}",
            ))
        out.append(RequirementProfile(
            profile_id=f"synthetic-jd-{j:02d}", version=1, role_title=f"{rng.choice(titles)} ({domain})",
            requirements=tuple(reqs), approved=True, approved_by="synthetic-eval",
            approved_at=datetime(2026, 1, 1, tzinfo=UTC),
        ))
    return out


def relevance_grade(jd: RequirementProfile, facts: Sequence[GoldFact]) -> int:
    """Grade 0-5: weight-averaged credit of the candidate's gold skills for the JD's skill requirements."""
    level = {f.value.lower(): f.evidence_level for f in facts if f.kind == "skill" and f.evidence_level}
    total = earned = 0.0
    for r in jd.active_requirements:
        total += r.weight
        credits = [LEVEL_CREDIT[lv] for o in r.options if (lv := level.get(o.text.lower()))]
        earned += r.weight * max(credits, default=0.0)
    return int(round(5 * earned / total)) if total else 0


def relevance_labels(
    jds: Sequence[RequirementProfile], gold: Mapping[str, Sequence[GoldFact]]
) -> dict[str, dict[str, int]]:
    """``{jd_id: {candidate_id: grade}}`` for every candidate in ``gold``."""
    return {jd.profile_id: {cid: relevance_grade(jd, facts) for cid, facts in gold.items()} for jd in jds}
