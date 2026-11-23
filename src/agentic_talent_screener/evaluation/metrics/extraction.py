"""E11 extraction metrics: verified facts against synthetic gold facts, by fact type (slide 9, R43).

SYNTHETIC: the gold facts come from the generator that wrote the resume, so they are right by construction.
The target (about 85% right, slide 9) is a hypothesis to test, not a result; this module never calls it one.

Mapping from pipeline facts (kinds skill, certification, experience, education, project) to gold types:

* ``skill``       a ``skill`` fact that mentions a gold skill (precision); a gold skill is recalled when any
                  ``skill`` fact mentions it. Level agreement is reported separately (see below).
* ``role``        an ``experience`` fact containing a gold job title (the longest title contained wins). An
                  experience fact that contains no title but sits inside a gold source line is a grounded
                  bullet: neutral, counted apart, never a precision error.
* ``date_range``  the dates (years, as L3 extracts them) of a role-matched fact. Recalled when both the start
                  year and the end year (or ``present``) are among them; a fact with dates outside the role's
                  two years is a precision error.
* ``education``   an ``education`` fact containing a gold degree and field of study.
* ``institution`` NOT MEASURABLE on this pipeline by design: L3 drops any education claim that names an
                  institution (PII-blind boundary), so recall is 0 by policy, not by extraction error. The
                  count of verified facts that leak a gold college name is reported instead (target 0).

Precision = TP / (TP + FP) over predicted facts of that type; recall = TP_gold / gold of that type. Both carry
n and a Wilson interval. The overall row is micro-averaged over skill, role, date_range and education.

Skill evidence level (E3's labels) follows the L3 heuristic by construction, so level agreement is a
consistency check, not accuracy (evidence/T14/review.md); reported as agreement / over / under, with n.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..datasets.schemas import GoldFact
from .citation import LEVELS, mentions, norm, rate, verdict_lower
from .efficiency import NOT_MEASURABLE

EXTRACTION_TARGET = 0.85  # slide 9
MEASURED_TYPES = ("skill", "role", "date_range", "education")
ALL_TYPES = (*MEASURED_TYPES, "institution")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
INSTITUTION_REASON = "L3 refuses institution names by design (PII-blind boundary), so recall is 0 by policy"
LABEL = "SYNTHETIC gold facts (known by construction); targets are hypotheses, not results"


@dataclass
class Counts:
    """Per fact type: ``tp``/``fp`` over predicted facts (precision), ``found``/``gold`` over gold facts (recall)."""

    tp: int = 0
    fp: int = 0
    neutral: int = 0  # grounded in a gold line but not a gold type (not an error)
    found: int = 0
    gold: int = 0


@dataclass
class ResumeExtraction:
    resume_id: str
    counts: dict[str, Counts] = field(default_factory=lambda: {t: Counts() for t in ALL_TYPES})
    level_pairs: list[tuple[str, str]] = field(default_factory=list)  # (gold level, predicted level)
    institution_leaks: int = 0


def _title(g: GoldFact) -> str:
    return g.value.split(" @ ")[0]


def _years(g: GoldFact) -> tuple[str, str | None]:
    start, _, end = g.value.partition("..")
    return start[:4], (None if end == "present" else end[:4])


def _has(statement: str, phrase: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(norm(phrase))}(?![A-Za-z0-9])", norm(statement)) is not None


def _role_index(statement: str, roles: Sequence[GoldFact]) -> int | None:
    """Index into ``roles`` of the longest gold title contained in ``statement`` (None if no title)."""
    hits = [(len(_title(g)), i) for i, g in enumerate(roles) if _has(statement, _title(g))]
    return max(hits)[1] if hits else None


def _edu_matches(statement: str, g: GoldFact) -> bool:
    body = g.value.rsplit(" (", 1)[0]  # "BSc Computer Science (2020)" -> degree + field
    degree, _, field_ = body.partition(" ")
    return _has(statement, degree) and _has(statement, field_)


def match_resume(resume_id: str, gold: Sequence[GoldFact], facts: Sequence[Any]) -> ResumeExtraction:
    """Score one resume's verified facts against its gold facts. ``facts`` are ``VerifiedFact``-like objects
    (``kind``, ``statement``, ``dates``, ``evidence_level``)."""
    out = ResumeExtraction(resume_id)
    c = out.counts
    g_skill = [g for g in gold if g.kind == "skill"]
    g_role = [g for g in gold if g.kind == "role"]
    g_date = [g for g in gold if g.kind == "date_range"]
    g_edu = [g for g in gold if g.kind == "education"]
    g_inst = [g for g in gold if g.kind == "institution"]
    lines = [norm(g.source_text) for g in gold]
    skill_facts = [f for f in facts if f.kind == "skill"]
    exp_facts = [f for f in facts if f.kind == "experience"]
    edu_facts = [f for f in facts if f.kind == "education"]
    for typ, items in (("skill", g_skill), ("role", g_role), ("date_range", g_date), ("education", g_edu),
                       ("institution", g_inst)):
        c[typ].gold = len(items)

    for f in skill_facts:  # skills: precision by fact, recall by gold skill
        if any(mentions(f.statement, g.value) for g in g_skill):
            c["skill"].tp += 1
        else:
            c["skill"].fp += 1
    for g in g_skill:
        if any(mentions(f.statement, g.value) for f in skill_facts):
            c["skill"].found += 1
            top = max(LEVELS[f.evidence_level] for f in facts if mentions(f.statement, g.value))
            out.level_pairs.append((g.evidence_level or "", f"E{top}"))

    roles_found: set[int] = set()
    dates_found: set[int] = set()
    for f in exp_facts:  # roles and their dates
        idx = _role_index(f.statement, g_role)
        if idx is None:
            if any(norm(f.statement) in line for line in lines):
                c["role"].neutral += 1
            else:
                c["role"].fp += 1
            continue
        c["role"].tp += 1
        roles_found.add(idx)
        role_no = g_role[idx].role_index
        gold_date = next((g for g in g_date if g.role_index == role_no), None)
        if gold_date is not None and f.dates:
            start, end = _years(gold_date)
            years = {m for d in f.dates for m in _YEAR.findall(d)}
            if years <= {start, end or start}:
                c["date_range"].tp += 1
                if start in years and (end is None or end in years):
                    dates_found.add(role_no if role_no is not None else -1)
            else:
                c["date_range"].fp += 1
    c["role"].found = len(roles_found)
    c["date_range"].found = len(dates_found)

    edu_found: set[int] = set()
    for f in edu_facts:
        hit = [i for i, g in enumerate(g_edu) if _edu_matches(f.statement, g)]
        if hit:
            c["education"].tp += 1
            edu_found.update(hit)
        else:
            c["education"].fp += 1
    c["education"].found = len(edu_found)

    out.institution_leaks = sum(_has(f.statement, g.value) for g in g_inst for f in facts)
    return out


def summarize_extraction(rows: Sequence[ResumeExtraction], *, min_n: int = 30,
                         target: float = EXTRACTION_TARGET) -> dict[str, Any]:
    """Precision and recall by fact type plus a micro overall over the measured types."""
    tot = {t: Counts() for t in ALL_TYPES}
    for r in rows:
        for typ, k in r.counts.items():
            for name in ("tp", "fp", "neutral", "found", "gold"):
                setattr(tot[typ], name, getattr(tot[typ], name) + getattr(k, name))

    def block(typ: str) -> dict[str, Any]:
        k = tot[typ]
        return {"precision": verdict_lower(k.tp, k.tp + k.fp, target, min_n),
                "recall": verdict_lower(k.found, k.gold, target, min_n),
                "n_predicted": k.tp + k.fp, "n_gold": k.gold, "neutral_grounded": k.neutral}

    by_type: dict[str, Any] = {typ: block(typ) for typ in MEASURED_TYPES}
    leaks = sum(r.institution_leaks for r in rows)
    na = {"status": f"{NOT_MEASURABLE} ({INSTITUTION_REASON})", "value": None, "ci95": None}
    by_type["institution"] = {
        "precision": {**na, "n": 0}, "recall": {**na, "n": tot["institution"].gold},
        "n_predicted": 0, "n_gold": tot["institution"].gold, "leaked_into_verified_facts": leaks,
        "note": "excluded from the overall row; leaked_into_verified_facts must be 0"}
    p_tp = sum(tot[t].tp for t in MEASURED_TYPES)
    p_n = p_tp + sum(tot[t].fp for t in MEASURED_TYPES)
    r_tp = sum(tot[t].found for t in MEASURED_TYPES)
    r_n = sum(tot[t].gold for t in MEASURED_TYPES)
    overall = {"precision": verdict_lower(p_tp, p_n, target, min_n), "recall": verdict_lower(r_tp, r_n, target, min_n),
               "types": list(MEASURED_TYPES), "averaging": "micro (pooled over facts)"}
    pairs = [p for r in rows for p in r.level_pairs]
    levels = {
        "n": len(pairs), "agreement": rate(sum(g == p for g, p in pairs), len(pairs), min_n),
        "predicted_above_gold": rate(sum(LEVELS[p] > LEVELS.get(g, 0) for g, p in pairs), len(pairs), min_n),
        "predicted_below_gold": rate(sum(LEVELS[p] < LEVELS.get(g, 0) for g, p in pairs), len(pairs), min_n),
        "caveat": "gold levels follow the L3 heuristic by construction: a consistency check, not accuracy"}
    return {"label": LABEL, "target": target, "target_source": "slide 9 (~85% of facts right), R43",
            "n_resumes": len(rows), "min_n": min_n, "overall": overall, "by_type": by_type, "skill_level": levels}


__all__ = ["EXTRACTION_TARGET", "LABEL", "MEASURED_TYPES", "ResumeExtraction", "match_resume", "summarize_extraction"]
