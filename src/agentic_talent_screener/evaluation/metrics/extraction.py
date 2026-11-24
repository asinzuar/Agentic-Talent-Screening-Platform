"""E11 extraction metrics: verified facts against synthetic gold facts, by fact type (slide 9, R43).

SYNTHETIC: the gold facts come from the generator that wrote the resume, so they are right by construction.
The target (about 85% right, slide 9) is a hypothesis to test, not a result; this module never calls it one.

Mapping from pipeline facts (kinds skill, certification, experience, education, project) to gold types:

* ``skill``       a ``skill`` fact that mentions a gold skill (precision); a gold skill is recalled when any
                  ``skill`` fact mentions it. Level agreement is reported separately (see below).
* ``role``        an ``experience`` fact containing a gold job title, matched ONE-TO-ONE (``assign_roles``: title,
                  then employer and dates; a repeated title maps its facts to different gold roles). An
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
from .citation import LEVELS, mentions, norm, rate, verdict_lower, verdict_upper
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
    institution_leaks: int = 0  # verified facts that name a gold college
    n_facts: int = 0


def _title(g: GoldFact) -> str:
    return g.value.split(" @ ")[0]


def _years(g: GoldFact) -> tuple[str, str | None]:
    start, _, end = g.value.partition("..")
    return start[:4], (None if end == "present" else end[:4])


def _has(statement: str, phrase: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(norm(phrase))}(?![A-Za-z0-9])", norm(statement)) is not None


def _fact_years(f: Any) -> set[str]:
    return {m for d in f.dates for m in _YEAR.findall(d)}


def _allowed_years(g: GoldFact) -> set[str]:
    start, end = _years(g)
    return {start, end or start}


def assign_roles(exp_facts: Sequence[Any], roles: Sequence[GoldFact], dates: Sequence[GoldFact]) -> dict[int, int]:
    """One-to-one assignment of experience facts to gold roles: ``{fact index: role index}``.

    A fact is a candidate for every gold role whose title it contains (only the longest titles, so "Senior Software
    Engineer" is not also "Software Engineer"). Candidates are scored on employer (+3), then dates: the fact's years
    all inside the role's two years (+2, +1 more when they cover both), outside them (-1), no dates (0). Pairs are
    taken best score first and each fact and each gold role is used once, so two facts for a repeated title land on
    two different roles, by employer or dates. A fact left over once every gold role of its title is used is a
    duplicate: neutral, never an error and never a second match of the same gold fact."""
    date_of = {g.role_index: g for g in dates}
    pairs: list[tuple[int, int, int]] = []
    for fi, f in enumerate(exp_facts):
        hits = [(len(_title(g)), ri) for ri, g in enumerate(roles) if _has(f.statement, _title(g))]
        if not hits:
            continue
        top = max(h[0] for h in hits)
        years = _fact_years(f)
        for size, ri in hits:
            if size != top:
                continue
            g = roles[ri]
            score = 0
            company = g.value.split(" @ ", 1)[1] if " @ " in g.value else ""
            if company and _has(f.statement, company):
                score += 3
            gd = date_of.get(g.role_index)
            if years and gd is not None:
                start, end = _years(gd)
                if years <= _allowed_years(gd):
                    score += 2 + (1 if start in years and (end is None or end in years) else 0)
                else:
                    score -= 1
            pairs.append((-score, fi, ri))
    out: dict[int, int] = {}
    used: set[int] = set()
    for _, fi, ri in sorted(pairs):
        if fi not in out and ri not in used:
            out[fi] = ri
            used.add(ri)
    return out


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

    seen_skills: set[str] = set()  # one-to-one: a gold skill is credited to the first fact that names it
    for f in skill_facts:  # skills: precision by fact, recall by gold skill
        named = [g.value for g in g_skill if mentions(f.statement, g.value)]
        if not named:
            c["skill"].fp += 1
        elif set(named) <= seen_skills:
            c["skill"].neutral += 1  # a repeat of skills already credited
        else:
            c["skill"].tp += 1
        seen_skills.update(named)
    for g in g_skill:
        if any(mentions(f.statement, g.value) for f in skill_facts):
            c["skill"].found += 1
            top = max(LEVELS[f.evidence_level] for f in facts if mentions(f.statement, g.value))
            out.level_pairs.append((g.evidence_level or "", f"E{top}"))

    assigned = assign_roles(exp_facts, g_role, g_date)
    roles_found: set[int] = set()
    dates_found: set[int] = set()
    for fi, f in enumerate(exp_facts):  # roles and their dates
        ri = assigned.get(fi)
        if ri is None:
            if any(_has(f.statement, _title(g)) for g in g_role) or any(norm(f.statement) in line for line in lines):
                c["role"].neutral += 1  # a duplicate of a role already matched, or a grounded bullet
            else:
                c["role"].fp += 1
            continue
        c["role"].tp += 1
        roles_found.add(ri)
        role_no = g_role[ri].role_index
        gold_date = next((g for g in g_date if g.role_index == role_no), None)
        if gold_date is not None and f.dates:
            start, end = _years(gold_date)
            years = _fact_years(f)
            if years <= _allowed_years(gold_date):
                c["date_range"].tp += 1
                if start in years and (end is None or end in years):
                    dates_found.add(role_no if role_no is not None else -1)
            else:
                c["date_range"].fp += 1
    c["role"].found = len(roles_found)
    c["date_range"].found = len(dates_found)

    edu_found: set[int] = set()  # one-to-one: a gold education is credited once; a repeat is neutral
    for f in edu_facts:
        hit = [i for i, g in enumerate(g_edu) if _edu_matches(f.statement, g)]
        if not hit:
            c["education"].fp += 1
        elif set(hit) <= edu_found:
            c["education"].neutral += 1
        else:
            c["education"].tp += 1
        edu_found.update(hit)
    c["education"].found = len(edu_found)

    out.n_facts = len(facts)
    out.institution_leaks = sum(any(_has(f.statement, g.value) for g in g_inst) for f in facts)
    return out


def summarize_extraction(rows: Sequence[ResumeExtraction], *, min_n: int = 30,
                         target: float = EXTRACTION_TARGET) -> dict[str, Any]:
    """Precision and recall by fact type plus a micro overall over the measured types."""
    rows_in = list(rows)
    tot = {t: Counts() for t in ALL_TYPES}
    for r in rows:
        for typ, k in r.counts.items():
            for name in ("tp", "fp", "neutral", "found", "gold"):
                setattr(tot[typ], name, getattr(tot[typ], name) + getattr(k, name))

    def prec(types: Sequence[str]) -> dict[str, Any]:
        rows = [(r.resume_id, sum(r.counts[t].tp for t in types), sum(r.counts[t].tp + r.counts[t].fp for t in types))
                for r in rows_in]
        return verdict_lower(sum(x[1] for x in rows), sum(x[2] for x in rows), target, min_n, clusters=rows)

    def rec(types: Sequence[str]) -> dict[str, Any]:
        rows = [(r.resume_id, sum(r.counts[t].found for t in types), sum(r.counts[t].gold for t in types))
                for r in rows_in]
        return verdict_lower(sum(x[1] for x in rows), sum(x[2] for x in rows), target, min_n, clusters=rows)

    def block(typ: str) -> dict[str, Any]:
        k = tot[typ]
        return {"precision": prec([typ]), "recall": rec([typ]),
                "n_predicted": k.tp + k.fp, "n_gold": k.gold, "neutral_grounded": k.neutral}

    by_type: dict[str, Any] = {typ: block(typ) for typ in MEASURED_TYPES}
    leaks = verdict_upper(sum(r.institution_leaks for r in rows), sum(r.n_facts for r in rows), 0.0, min_n,
                          clusters=[(r.resume_id, r.institution_leaks, r.n_facts) for r in rows])
    leaks["unit"] = "n = verified facts; a fact leaks when it names the resume's college"
    na = {"status": f"{NOT_MEASURABLE} ({INSTITUTION_REASON})", "value": None, "ci95": None}
    by_type["institution"] = {
        "precision": {**na, "n": 0}, "recall": {**na, "n": tot["institution"].gold},
        "n_predicted": 0, "n_gold": tot["institution"].gold,
        "leaked_into_verified_facts": leaks["k"],
        "note": "recall excluded from the overall row; leaks are enforced as `institution_leaks` (target 0)"}
    overall = {"precision": prec(MEASURED_TYPES), "recall": rec(MEASURED_TYPES),
               "types": list(MEASURED_TYPES), "averaging": "micro (pooled over facts)"}
    pairs = [p for r in rows for p in r.level_pairs]
    levels = {
        "n": len(pairs), "agreement": rate(sum(g == p for g, p in pairs), len(pairs), min_n),
        "predicted_above_gold": rate(sum(LEVELS[p] > LEVELS.get(g, 0) for g, p in pairs), len(pairs), min_n),
        "predicted_below_gold": rate(sum(LEVELS[p] < LEVELS.get(g, 0) for g, p in pairs), len(pairs), min_n),
        "caveat": "gold levels follow the L3 heuristic by construction: a consistency check, not accuracy; "
                  "disagreement below gold means the fact was found only at a lower level (e.g. from the skills list)"}
    return {"label": LABEL, "target": target, "target_source": "slide 9 (~85% of facts right), R43",
            "n_resumes": len(rows), "min_n": min_n, "overall": overall, "by_type": by_type, "institution_leaks": leaks,
            "unit": "n = predicted facts (precision) or gold facts (recall); n_clusters = resumes",
            "skill_level": levels}


__all__ = ["EXTRACTION_TARGET", "LABEL", "MEASURED_TYPES", "ResumeExtraction", "match_resume", "summarize_extraction"]
