"""E11 claims suite: the claim-evidence test (``run --suite claims``).

Does every extracted claim and every explanation line cite a visible span that supports it? Four measurements
and one human step (deck slides 7, 8 and 9; requirements R12, R43, R45):

1. citation precision (verified profile, explanation quotes, plus an audit of L2's raw claims),
2. claim recall and precision against synthetic gold facts, by fact type,
3. unsupported-claim (A7) acceptance on twins, and
4. evidence-level inflation,

plus the human audit kit (``audit.py``). Everything here is deterministic and offline-testable; the only LLM
calls are L1b and L2 (the pipeline under test) and the optional judge, all through ``CachedChat`` with the
free-only guard and the live-call cap. Guard refusals, cache misses and the call limit are infrastructure
failures: they stop that phase, keep what was measured, and are reported apart from every metric.

Ground truth is synthetic (gold facts known by construction); the report says so on every page.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_talent_screener.ranking import matching
from agentic_talent_screener.ranking.config import DEFAULT_CONFIG
from agentic_talent_screener.ranking.schemas import (
    CandidateMatches,
    Criterion,
    EvidenceLevel,
    MatcherView,
    OptionMatch,
    Priority,
    Requirement,
    RequirementProfile,
)
from agentic_talent_screener.ranking.scorer import score_pool
from agentic_talent_screener.resume_intake import pipeline
from agentic_talent_screener.resume_intake.l3_evidence_verifier import NON_REJECTION_CODES
from agentic_talent_screener.resume_intake.ranking_adapter import to_ranking_profile

from . import audit
from .attacks.twins import generate_twins, write_twins
from .budget import BudgetExceeded, BudgetGuard
from .cache import CacheMiss, CachedChat, LLMCache
from .store import EvalStore
from .datasets.schemas import GoldFact
from .datasets.synthetic import make_resume, safe_output_dir, write_pool
from .jds import synthetic_jds
from .manifest import build_manifest, write_manifest
from .metrics import citation as cit
from .metrics import extraction as ext
from .report import NO_TARGET, _table, category, combine, fmt_value, norm_verdict
from .runner import DEFAULT_MAX_LLM_CALLS, RAW_FILE, LiveCallLimit, _reason, clean_json
from .systems import (
    InfrastructureError,
    _routed_through,
    _Surfacing,
    build_hooks,
    seeded_matcher_messages,
)

KIND = "claims"
LABEL = ("SYNTHETIC: gold facts are known by construction (the resumes were generated from them); A7 twins are "
         "synthetic; no human has graded anything unless an audit was imported. Targets are hypotheses to test.")
FORMAT_ORDER = ("pdf", "docx", "txt")
RESULTS_FILE, REPORT_FILE, PER_ITEM_FILE = "results.json", "report.md", "per_item.jsonl"
MAX_EXAMPLES = 25
_INFRA = (InfrastructureError, BudgetExceeded, CacheMiss)


@dataclass(frozen=True)
class ClaimsSettings:
    seed: int = 0
    pool_size: int = 150
    n_bases: int = 20
    n_jds: int = 10
    grid: str = "core"
    min_n: int = 30
    offline: bool = False
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    cache_dir: str = "data/eval/cache"
    judge: bool = False
    judge_n: int = 30
    audit_n: int = audit.DEFAULT_AUDIT_N
    l3_verifier: str = "deterministic"
    l3_model: str = "ats-l3-qwen2.5-3b"

    def validate(self) -> None:
        if self.l3_verifier not in {"deterministic", "ollama"}:
            raise ValueError("Unknown L3 verifier.")
        if self.offline and self.l3_verifier == "ollama":
            raise ValueError("Offline claims runs cannot call Ollama; use the explicit NLI replay command locally.")
        if not 1 <= self.pool_size <= 500 or not 1 <= self.n_bases <= 500 or not 1 <= self.n_jds <= 60:
            raise ValueError("need 1 <= pool_size <= 500, 1 <= n_bases <= 500 and 1 <= n_jds <= 60")
        if self.min_n < 1 or self.max_llm_calls < 0 or not 0 <= self.judge_n <= 500 or not 1 <= self.audit_n <= 100:
            raise ValueError("min_verdict_n >= 1, max_llm_calls >= 0, 0 <= judge_n <= 500 and 1 <= audit_n <= 100")
        if self.grid not in ("core", "full"):
            raise ValueError("grid must be 'core' or 'full'")


# ----------------------------------------------------------------------------- matcher for the probe


class SubstringMatcher:
    """Deterministic lexical linker: a fact supports an option when it mentions the option's text. No LLM, so
    what the suite measures is the evidence ladder and the citations, not matcher noise."""

    version = "substring-matcher-eval-v1"

    def match(self, candidate: MatcherView, profile: RequirementProfile) -> CandidateMatches:
        links = tuple(
            OptionMatch(option_id=o.option_id, fact_id=f.fact_id, relation="equivalent")
            for f in candidate.facts for r in profile.active_requirements for o in r.options
            if cit.mentions(f.statement, o.text))
        return CandidateMatches(candidate_id=candidate.candidate_id, profile_id=profile.profile_id,
                                profile_version=profile.version, matcher_version=self.version, matches=links)


def probe_profile(index: int, skills: Sequence[str]) -> RequirementProfile:
    """An approved profile with one mandatory requirement per skill (the A7 scoring probe)."""
    reqs = tuple(
        Requirement(requirement_id=f"M{i}", statement=s, kind="skill",
                    options=(Criterion(option_id=f"M{i}.a", text=s),), priority=Priority.MANDATORY, weight=1.0,
                    provenance="jd_qualification", quote=f"Experience with {s}")
        for i, s in enumerate(skills, start=1))
    return RequirementProfile(profile_id=f"a7-probe-{index:04d}", version=1, role_title="A7 probe", requirements=reqs,
                              approved=True, approved_by="synthetic-eval", approved_at=datetime(2026, 1, 1, tzinfo=UTC))


# ----------------------------------------------------------------------------- intake with probes


@dataclass
class IntakeRecord:
    candidate_id: str
    status: str  # normalized, normalized_with_flag, quarantined, manual_review, l2_extracted, error
    stopped_at: str = ""
    report: dict[str, Any] | None = None
    excluded: frozenset[str] = frozenset()
    l2_output: Any = None
    profile: Any = None  # VerifiedCandidateProfile
    handoff: Any = None  # RankingHandoff
    detail: str = ""
    initial_l2_output: Any = None

    @property
    def has_profile(self) -> bool:
        return self.profile is not None and self.handoff is not None


def _probe_post(post: tuple[Any, ...], sink: dict[str, Any], support_verifier: Any = None) -> tuple[Any, ...]:
    """Wrap the first post-L2 hook so the validated (pre-L3) L2 claims and the exclusions are recorded."""
    inner = post[0]

    def hook(ctx: Any) -> Any:
        out = inner(ctx)
        sink["l2_output"] = ctx.agent_output
        sink.setdefault("initial_l2_output", ctx.agent_output)
        ctx.support_verifier = support_verifier
        sink["excluded"] = frozenset(ctx.excluded_span_ids)
        return out
    return (hook, *post[1:])


def run_intake(chat: CachedChat, candidates: Sequence[tuple[str, Path]], *, seed: int,
               model: str | None, support_verifier: Any = None) -> tuple[list[IntakeRecord], str | None]:
    """Run intake on each (id, path). Stops at the first infrastructure failure and returns what it has, plus the
    reason (None when every candidate was processed)."""
    records: list[IntakeRecord] = []
    sink: dict[str, Any] = {}
    surfacing = _Surfacing(chat)
    pre, post, build = build_hooks(None, sink, seed)
    post = _probe_post(post, sink, support_verifier)
    try:
        with _routed_through(surfacing, pre, post, build, seeded_matcher_messages(seed)):
            for cid, path in candidates:
                sink.clear()
                try:
                    result = pipeline.normalize_resume(path, model, show_progress=False)
                    surfacing.check()
                    rec = IntakeRecord(cid, result.status, result.stopped_at, sink.get("report"),
                                       sink.get("excluded", frozenset()), sink.get("l2_output"), result.profile)
                    if "L3_NLI_PROCESSING_FAILED" in result.events:
                        rec.detail = "local L3 NLI processing failure"
                    if result.status in ("normalized", "normalized_with_flag") and result.profile is not None \
                            and result.security_assessment is not None:
                        rec.handoff = to_ranking_profile(result.profile, result.security_assessment,
                                                         forensic_report=sink.get("report"))
                except InfrastructureError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad resume must not sink the pool
                    surfacing.check()
                    rec = IntakeRecord(cid, "error", detail=type(exc).__name__, report=sink.get("report"))
                rec.initial_l2_output = sink.get("initial_l2_output")
                records.append(rec)
                if rec.detail == "local L3 NLI processing failure":
                    return records, rec.detail
    except _INFRA as exc:
        return records, _reason(exc)
    return records, None


# ----------------------------------------------------------------------------- per-resume evaluation


@dataclass
class RecordEval:
    checks: list[cit.CitationCheck] = field(default_factory=list)  # profile + l2_raw
    levels: list[cit.LevelRow] = field(default_factory=list)
    extraction: ext.ResumeExtraction | None = None
    l3_events: Counter = field(default_factory=Counter)
    n_l2_claims: int = 0
    n_facts: int = 0
    pairs: list[tuple[str, str, str]] = field(default_factory=list)  # (claim, span_text, key) for the judge


def evaluate_record(candidate_id: str, report: Mapping[str, Any] | None, excluded: Sequence[str], profile: Any,
                    l2_output: Any, gold: Sequence[GoldFact] | None = None) -> RecordEval:
    """Audit one resume: the verified profile's citations, the raw L2 claims' citations, evidence levels, and
    extraction against ``gold`` (when given). Pure function of its inputs; used directly by the tests."""
    spans = cit.span_index(report, excluded)
    out = RecordEval()
    if l2_output is not None:
        out.n_l2_claims = len(l2_output.proposed_claims)
        for c in l2_output.proposed_claims:
            union = " ".join(spans[s].text for s in c.source_span_ids if s in spans and spans[s].visible)
            for sid in c.source_span_ids:
                out.checks.append(cit.check_citation(
                    c.value, sid, c.value, spans, source="l2_raw", unit_id=c.claim_id, claim_id=c.claim_id,
                    candidate_id=candidate_id, kind=c.kind, support_text=union))
    if profile is not None:
        out.n_facts = len(profile.facts)
        out.l3_events.update(e.code for e in profile.verification_events if e.code not in NON_REJECTION_CODES)
        for f in profile.facts:
            union = " ".join(spans[s.span_id].text for s in f.source_spans
                             if s.span_id in spans and spans[s.span_id].visible)
            for sp in f.source_spans:
                chk = cit.check_citation(
                    f.statement, sp.span_id, sp.quote, spans, source="profile", unit_id=f.fact_id,
                    claim_id=f.fact_id, candidate_id=candidate_id, kind=f.kind, support_text=union)
                out.checks.append(chk)
                if chk.ok:
                    out.pairs.append((f.statement, union, f"{candidate_id}|{f.fact_id}|{sp.span_id}"))
        out.levels = cit.level_rows(candidate_id, profile.facts, spans)
        if gold is not None:
            out.extraction = ext.match_resume(candidate_id, gold, profile.facts)
    return out


def explanation_checks(candidate_id: str, ranking_result: Any, handoff_profile: Any,
                       spans: Mapping[str, cit.SpanInfo], jd_id: str) -> tuple[list[cit.CitationCheck], list[dict[str, Any]], int]:
    """Citation checks for every explanation line that relies on evidence, plus the audit candidates.

    A line relies on evidence when its criterion earned credit or carries a quote or location. Credit with no
    citation is a ``QUOTE_EMPTY`` failure. Returns (checks, audit items, n criteria that earned nothing and
    cite nothing)."""
    facts = {f.fact_id: f for f in handoff_profile.facts}
    checks: list[cit.CitationCheck] = []
    items: list[dict[str, Any]] = []
    idle = 0
    for expl in ranking_result.explanations:
        if expl.missing:
            continue
        for crit in expl.criteria:
            if not (crit.credit > 0 or crit.quote or crit.location):
                idle += 1
                continue
            fact = facts.get(crit.evidence_fact or "")
            claim = fact.statement if fact is not None else crit.text
            unit = f"{jd_id}:{expl.requirement_id}:{crit.option_id}"
            if not crit.location or not crit.quote:
                chk = cit.CitationCheck("explanation", unit, crit.evidence_fact or "", crit.location or "",
                                        "QUOTE_EMPTY", candidate_id, "criterion")
            else:
                support = spans[crit.location].text if crit.location in spans else ""
                chk = cit.check_citation(claim, crit.location, crit.quote, spans, source="explanation",
                                         unit_id=unit, claim_id=crit.evidence_fact or "", candidate_id=candidate_id,
                                         kind="criterion", support_text=support)
            checks.append(chk)
            if crit.location in spans and crit.quote:
                level = crit.effective_level.name if crit.effective_level is not None else "n/a"
                items.append({
                    "key": f"{candidate_id}|{unit}", "candidate_id": candidate_id, "requirement": expl.statement,
                    "explanation": f"{crit.text}: {'met' if crit.met else 'partly'} at {level} "
                                   f"(credit {crit.credit:.2f}) via the fact \"{claim}\"",
                    "quote": crit.quote, "span_id": crit.location, "span_text": spans[crit.location].text[:400],
                    "auto_ok": chk.ok, "auto_failure": chk.failure})
    return checks, items, idle


# ----------------------------------------------------------------------------- the run


def _pool(settings: ClaimsSettings, root: Path) -> tuple[list[Any], Path]:
    resumes = [make_resume(i, settings.seed) for i in range(settings.pool_size)]
    folder = write_pool(resumes, f"data/eval/claims/s{settings.seed}-p{settings.pool_size}/pool", ("pdf",), root)
    return resumes, folder


def _a7_phase(settings: ClaimsSettings, chat: CachedChat, root: Path, resumes: Sequence[Any], matcher: Any,
              model: str | None, support_verifier: Any = None) -> dict[str, Any]:
    bases = list(resumes[:settings.n_bases])
    twins = generate_twins(bases, settings.grid, settings.seed, ["A7"])
    out_dir = write_twins(twins, f"data/eval/claims/s{settings.seed}-b{settings.n_bases}-{settings.grid}/twins",
                          FORMAT_ORDER, root)
    inputs, unreadable = [], 0
    for t in twins:
        path = next((out_dir / f"{t.label.twin_id}.{f}" for f in FORMAT_ORDER
                     if f in t.label.formats and (out_dir / f"{t.label.twin_id}.{f}").exists()), None)
        if path is None:
            unreadable += 1
            continue
        inputs.append((t, path))
    records, stop = run_intake(chat, [(t.label.twin_id, p) for t, p in inputs], seed=settings.seed, model=model,
                               support_verifier=support_verifier)
    by_id = {r.candidate_id: r for r in records}
    cases: list[cit.A7Case] = []
    outcomes: Counter = Counter()
    for k, (t, _) in enumerate(inputs):
        rec = by_id.get(t.label.twin_id)
        if rec is None:
            outcomes["not_run"] += 1
            continue
        outcomes["profile" if rec.has_profile else rec.status] += 1
        if not rec.has_profile:
            continue
        skills = cit.a7_skills(t.label.payload)
        if not skills:
            outcomes["no_skills_in_payload"] += 1
            continue
        profile = rec.handoff.profile.model_copy(update={"candidate_id": "C-0001"})
        result = score_pool([profile], probe_profile(k, skills), matcher, DEFAULT_CONFIG).results[0]
        by_req = {e.statement: e for e in result.explanations}
        for s in skills:
            e = by_req.get(s)
            deciding = e.deciding if e is not None else None
            level = int(deciding.effective_level) if deciding is not None and deciding.effective_level is not None else None
            cases.append(cit.a7_case(t.label.twin_id, s, rec.profile.facts, level, bool(e and e.credit > 0)))
    summary = cit.summarize_a7(cases, min_n=settings.min_n, n_unreadable=unreadable)
    summary["twin_outcomes"] = dict(sorted(outcomes.items()))
    summary["n_twins_planned"] = len(twins)
    return {"summary": summary, "stop": stop, "n_processed": len(records), "n_planned": len(inputs)}


def run_claims(settings: ClaimsSettings, *, base: str | Path, run_dir: str | Path,
               call: Callable[..., dict[str, Any]] | None = None, model: str | None = None,
               matcher: matching.MatchingAgent | None = None) -> dict[str, Any]:
    """Run the claims suite and return the raw bundle (``raw.json`` content). ``call`` is the live provider
    function (default ``llm.chat_json``); tests pass a fake. Writes the audit kit under ``run_dir/audit``.

    LLM responses are cached in the shared eval store (``LLMCache(store, synthetic=True)``, E15b) at the repo root.
    The suite is free-only: ``BudgetGuard(0.0)`` with no router, no meter and no paid path, so a non-free model is
    refused (``free_only``) before any cache or network access."""
    settings.validate()
    from agentic_talent_screener.resume_intake.nli import OllamaVerifier
    from agentic_talent_screener.llm import ollama_model_digest
    support_verifier = OllamaVerifier(settings.l3_model) if settings.l3_verifier == "ollama" else None
    digest = ollama_model_digest(settings.l3_model) if support_verifier else None
    root = Path(base).resolve()
    safe_output_dir(settings.cache_dir, root)  # validated for compatibility; the cache is the root eval store
    limiter = LiveCallLimit(settings.max_llm_calls, call)
    store = EvalStore.for_root(root)
    try:
        chat = CachedChat(LLMCache(store, synthetic=True), BudgetGuard(0.0), offline=settings.offline, call=limiter,
                          default_model=(lambda: model) if model else None)
        raw = _run_claims(settings, root, run_dir, chat, limiter, model, matcher or SubstringMatcher(), support_verifier)
        raw["settings"].update(l3_verifier=settings.l3_verifier, l3_model=settings.l3_model if support_verifier else None,
                               l3_model_digest=digest)
        if support_verifier:
            raw["manifest_extra"]["models"].append(settings.l3_model)
            raw["citation"]["support_metric_note"] = (
                "Word-overlap diagnostic, not semantic ground truth. Use independently labelled NLI pairs for accuracy.")
        return raw
    finally:
        store.close()  # idempotent; compacts the file


def _run_claims(settings: ClaimsSettings, root: Path, run_dir: str | Path, chat: CachedChat, limiter: LiveCallLimit,
                model: str | None, matcher: matching.MatchingAgent, support_verifier: Any = None) -> dict[str, Any]:
    resumes, pool_dir = _pool(settings, root)
    gold = {r.resume_id: r.gold_facts() for r in resumes}
    errors: dict[str, str] = {}

    records, stop = run_intake(chat, [(r.resume_id, pool_dir / f"{r.resume_id}.pdf") for r in resumes],
                               seed=settings.seed, model=model, support_verifier=support_verifier)
    # Only synthetic suite inputs are captured. Replay freezes the first valid L2
    # response so verifier-dependent targeted retries cannot change the comparison.
    replay_path = Path(run_dir) / "l3_inputs.jsonl"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    with replay_path.open("w", encoding="utf-8") as stream:
        for rec in records:
            if rec.initial_l2_output is not None and rec.report is not None:
                stream.write(json.dumps({"synthetic": True, "candidate_id": rec.candidate_id,
                    "agent_output": rec.initial_l2_output.model_dump(), "forensic_report": rec.report,
                    "excluded_span_ids": sorted(rec.excluded), "blind_candidate_id": "C-0001"}) + "\n")
    if stop:
        errors["extraction"] = stop
    evals = {rec.candidate_id: evaluate_record(
        rec.candidate_id, rec.report, rec.excluded, rec.profile, rec.l2_output, gold.get(rec.candidate_id))
        for rec in records}
    outcomes = Counter("profile" if r.has_profile else r.status for r in records)
    with_profile = [r for r in records if r.has_profile]

    # explanations: score the pool against the synthetic JDs and audit every cited line
    jds = synthetic_jds(settings.n_jds, settings.seed)
    exp_checks: list[cit.CitationCheck] = []
    audit_items: list[dict[str, Any]] = []
    idle = 0
    if with_profile:
        blind = {f"C-{i + 1:04d}": r for i, r in enumerate(with_profile)}
        profiles = [r.handoff.profile.model_copy(update={"candidate_id": b}) for b, r in blind.items()]
        spans_of = {b: cit.span_index(r.report, r.excluded) for b, r in blind.items()}
        for jd in jds:
            ranking = score_pool(profiles, jd, matcher, DEFAULT_CONFIG)
            for res in ranking.results:
                if res.status != "ok" or res.candidate_id not in blind:
                    continue
                r = blind[res.candidate_id]
                c, items, n_idle = explanation_checks(r.candidate_id, res, r.handoff.profile,
                                                      spans_of[res.candidate_id], jd.profile_id)
                exp_checks += c
                audit_items += items
                idle += n_idle
    else:
        errors.setdefault("explanations", "no verified profile to explain")

    checks = [c for e in evals.values() for c in e.checks] + exp_checks
    judge = _judge_phase(settings, chat, evals, model) if settings.judge else None
    if judge and judge["unavailable"]:
        errors["judge"] = "; ".join(sorted(judge["unavailable_reasons"]))

    a7 = _a7_phase(settings, chat, root, resumes, matcher, model, support_verifier)
    if a7["stop"]:
        errors["a7"] = a7["stop"]

    rows = [e.extraction for e in evals.values() if e.extraction is not None]
    levels = [lv for e in evals.values() for lv in e.levels]
    l3 = Counter()
    for e in evals.values():
        l3.update(e.l3_events)
    n_l2 = sum(e.n_l2_claims for e in evals.values())
    n_facts = sum(e.n_facts for e in evals.values())
    manifest = audit.export_kit(audit_items, run_dir, n=settings.audit_n, seed=settings.seed) if audit_items else None
    status = "partial" if errors else "complete"
    n_cit = len([c for c in checks if c.source != "l2_raw"])
    items = [{"type": "resume", "candidate_id": r.candidate_id, "status": r.status, "stopped_at": r.stopped_at,
              "has_profile": r.has_profile, "n_l2_claims": evals[r.candidate_id].n_l2_claims,
              "n_facts": evals[r.candidate_id].n_facts,
              "n_citation_failures": sum(not c.ok for c in evals[r.candidate_id].checks if c.source == "profile"),
              "n_l2_claim_failures": sum(not c.ok for c in evals[r.candidate_id].checks if c.source == "l2_raw")}
             for r in records]
    failures = [{"source": c.source, "candidate_id": c.candidate_id, "unit": c.unit_id, "span_id": c.span_id,
                 "failure": c.failure, "kind": c.kind} for c in checks if not c.ok][:MAX_EXAMPLES]
    return {
        "schema": 1, "kind": KIND, "label": LABEL,
        "settings": {"suite": "claims", "seed": settings.seed, "pool_size": settings.pool_size,
                     "n_bases": settings.n_bases, "n_jds": settings.n_jds, "grid": settings.grid,
                     "min_verdict_n": settings.min_n, "offline": settings.offline,
                     "max_llm_calls": settings.max_llm_calls, "budget_usd": 0.0, "judge": settings.judge,
                     "audit_n": settings.audit_n, "matcher": getattr(matcher, "version", "custom")},
        "live_calls_used": limiter.count, "cache": {"hits": chat.cache.hits, "misses": chat.cache.misses},
        "status": status, "errors": errors,
        "intake": {"n_planned": len(resumes), "n_processed": len(records), "outcomes": dict(sorted(outcomes.items())),
                   "n_with_profile": len(with_profile), "n_l2_claims": n_l2, "n_verified_facts": n_facts,
                   "l3_rejections_by_code": dict(sorted(l3.items())),
                   "l3_rejection_note": "L2 claims that L3 dropped; each is a hallucination, a hidden or excluded "
                                        "span, or an unsupported value the pipeline caught before ranking"},
        "citation": cit.summarize_citations(checks, min_n=settings.min_n),
        "judge": judge,
        "extraction": ext.summarize_extraction(rows, min_n=settings.min_n) if rows else None,
        "inflation": cit.summarize_inflation(levels, min_n=settings.min_n),
        "a7": a7["summary"], "a7_progress": {"processed": a7["n_processed"], "planned": a7["n_planned"]},
        "explanations": {"n_citations": len([c for c in exp_checks]), "n_idle_criteria": idle,
                         "n_jds": len(jds), "n_audit_candidates": len(audit_items),
                         "audit_rows_exported": manifest["n_rows"] if manifest else 0},
        "failure_examples": failures, "n_citations_checked": n_cit,
        "items": items,
        "manifest_extra": {"dataset_hashes": {}, "models": [model] if model else []},
    }


def _judge_phase(settings: ClaimsSettings, chat: CachedChat, evals: Mapping[str, RecordEval],
                 model: str | None) -> dict[str, Any]:
    """Optional judge on a seeded sample of citations that passed the lexical check."""
    import hashlib

    pairs = sorted((p for e in evals.values() for p in e.pairs),
                   key=lambda p: hashlib.sha256(f"{settings.seed}|{p[2]}".encode()).hexdigest())[:settings.judge_n]
    verdicts: list[bool] = []
    reasons: Counter = Counter()
    for claim, text, _ in pairs:
        v, why = cit.judge_support(chat, claim, text, model)
        if v is None:
            reasons[why or "unavailable"] += 1
        else:
            verdicts.append(v)
    n_ok = sum(verdicts)
    return {"requested": len(pairs), "n_judged": len(verdicts), "unavailable": sum(reasons.values()),
            "unavailable_reasons": dict(reasons), "judge_supported": cit.rate(n_ok, len(verdicts), settings.min_n),
            "agreement_with_lexical": cit.rate(n_ok, len(verdicts), settings.min_n),
            "note": "judged citations all passed the lexical check; the share the judge also supports is the "
                    "agreement. Unavailable verdicts (cache miss offline, guard, call limit) are not failures."}


def execute_claims(settings: ClaimsSettings, *, base: str | Path, runs_dir: str | Path = "runs",
                   run_id: str | None = None, argv_note: dict[str, Any] | None = None, **kw: Any) -> Path:
    """Run, then write ``manifest.json``, ``raw.json`` and the report files. Returns the run directory."""
    run_id = run_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run id must be 1-64 characters of letters, digits, '_' or '-'")
    root = Path(base).resolve()
    run_dir = safe_output_dir(Path(runs_dir) / run_id, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_claims(settings, base=root, run_dir=run_dir, **kw)
    extra = {"run_id": run_id, "suite": "claims", "offline": settings.offline, "budget_usd": 0.0,
             "max_llm_calls": settings.max_llm_calls, "live_calls_used": raw["live_calls_used"], "cache": raw["cache"],
             "pool_size": settings.pool_size, "n_bases": settings.n_bases, "n_jds": settings.n_jds,
             "min_verdict_n": settings.min_n, "status": raw["status"], "suites_executed": ["claims"],
             "labels": "synthetic, not human-graded", **(argv_note or {})}
    write_manifest(run_dir, build_manifest(seed=settings.seed, models=raw["manifest_extra"]["models"], extra=extra))
    (run_dir / RAW_FILE).write_text(json.dumps(clean_json(raw), indent=1, sort_keys=True), encoding="utf-8")
    results = build_claims_report(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with EvalStore.for_root(root) as store:  # the run summary goes into the committed store, like the core suite
        store.put_run(run_id, manifest, results, manifest.get("git_sha", "unknown"), synthetic=True)
    return run_dir


# ----------------------------------------------------------------------------- report


def _m(items: dict[str, dict[str, Any]], key: str, stat: Mapping[str, Any] | None, *, target: Any = None,
       direction: str = "info", verdict: str | None = None, note: str | None = None) -> None:
    if not stat:
        return
    v = verdict if verdict is not None else norm_verdict(stat.get("status"), stat.get("note"))
    if stat.get("value") is None and category(v) not in ("not measurable", "fail"):
        v = "not measurable (no valid observations)"
    items[key] = {"value": stat.get("value"), "ci95": stat.get("ci95"), "n": stat.get("n"),
                  "n_clusters": stat.get("n_clusters"), "cluster_unit": stat.get("cluster_unit"), "target": target,
                  "verdict": v, "direction": direction, "unit": "rate", "system": "full_engine", "label": None,
                  "note": note}


def _no_target(stat: Mapping[str, Any]) -> str:
    return f"not measurable ({stat['status'].split('(', 1)[1][:-1]})" if str(stat.get("status", "")).startswith(
        "not measurable") else NO_TARGET


def derive_claims_results(raw: Mapping[str, Any], manifest: Mapping[str, Any],
                          audit_results: Mapping[str, Any] | None) -> dict[str, Any]:
    m: dict[str, dict[str, Any]] = {}
    cs = raw["citation"]
    _m(m, "citation.precision", cs["overall"], target=cit.CITATION_TARGET, direction="higher_better")
    for src, blk in cs["by_source"].items():
        raw_l2 = src == "l2_raw"
        _m(m, f"citation.precision.{src}", blk, target=None if raw_l2 else cit.CITATION_TARGET,
           direction="higher_better", verdict=_no_target(blk) if raw_l2 else norm_verdict(blk["status"]),
           note="context: L2's raw claims before L3" if raw_l2 else None)
    _m(m, "citation.claim_level", cs["claim_level"], direction="higher_better", verdict=_no_target(cs["claim_level"]))
    ex = raw.get("extraction")
    if ex:
        for part in ("precision", "recall"):
            _m(m, f"extraction.{part}.overall", ex["overall"][part], target=ext.EXTRACTION_TARGET,
               direction="higher_better")
            for typ, blk in ex["by_type"].items():
                _m(m, f"extraction.{part}.{typ}", blk[part], target=ext.EXTRACTION_TARGET, direction="higher_better",
                   verdict=norm_verdict(blk[part]["status"]))
        _m(m, "extraction.institution_leaks", ex["institution_leaks"], target=0.0, direction="lower_better")
        _m(m, "extraction.skill_level_agreement", ex["skill_level"]["agreement"], direction="higher_better",
           verdict=_no_target(ex["skill_level"]["agreement"]), note="consistency check, not accuracy")
    a7 = raw["a7"]
    _m(m, "a7.above_e1", a7, target=0.0, direction="lower_better")
    _m(m, "a7.accepted_above_e1", a7["accepted_above_e1"], direction="lower_better",
       verdict=_no_target(a7["accepted_above_e1"]))
    _m(m, "a7.scored_above_e1", a7["scored_above_e1"], direction="lower_better",
       verdict=_no_target(a7["scored_above_e1"]))
    _m(m, "a7.by_twin", a7["by_twin"], direction="lower_better", verdict=_no_target(a7["by_twin"]))
    infl = raw["inflation"]
    _m(m, "inflation.rate", infl, target=0.0, direction="lower_better")
    _m(m, "inflation.skills_list_only_above_e1", infl["skills_list_only_above_e1"], target=0.0,
       direction="lower_better")
    if audit_results and audit_results.get("n_ratings"):
        _m(m, "audit.citation_right", audit_results["citation_right"], target=cit.CITATION_TARGET,
           direction="higher_better")
    slide9 = _slide9(raw, m, audit_results)
    cats = Counter(category(r["verdict"]) for r in slide9)
    return clean_json({
        "schema": 1, "kind": KIND, "run_id": manifest.get("run_id"), "status": raw["status"], "label": raw["label"],
        "settings": raw["settings"], "live_calls_used": raw["live_calls_used"], "cache": raw["cache"],
        "errors": raw["errors"], "intake": raw["intake"],
        "summary": {"slide9_rows": len(slide9), **{c: cats.get(c, 0) for c in ("pass", "fail", "inconclusive", "not measurable")}},
        "slide9": slide9, "metrics": m, "citation": cs, "judge": raw.get("judge"), "extraction": ex, "a7": a7,
        "inflation": infl, "explanations": raw["explanations"], "audit": audit_results,
        "failure_examples": raw["failure_examples"], "caveats": _caveats(raw, audit_results), "manifest": manifest})


def _slide9(raw: Mapping[str, Any], m: Mapping[str, Any], audit_results: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ex = raw.get("extraction")
    erow: dict[str, Any] = {"layer": "Extraction", "question": "Do we capture the right facts?",
                            "target": "about 85% of facts right (precision and recall reported)", "system": "full_engine",
                            "unit": "rate", "components": []}
    if ex is None:
        erow.update(verdict=f"not measurable ({raw['errors'].get('extraction', 'no resume produced a verified profile')})",
                    measured="n/a", value=None, ci95=None, n=None)
    else:
        p, r = m["extraction.precision.overall"], m["extraction.recall.overall"]
        lk = m["extraction.institution_leaks"]
        erow.update(verdict=combine([p["verdict"], r["verdict"], lk["verdict"]]), value=r["value"], ci95=r["ci95"],
                    n=r["n"],
                    measured=(f"precision {fmt_value(p)} (n={p['n']}, {p['n_clusters']} resumes); recall {fmt_value(r)} "
                              f"(n={r['n']}, {r['n_clusters']} resumes); institution leaks {lk['value']:.0%} "
                              f"({lk['verdict']}); synthetic gold"),
                    components=[{"name": "precision", **p}, {"name": "recall", **r},
                                {"name": "institution leaks (target 0)", **lk}])
    rows.append(erow)
    cp = m["citation.precision.explanation"] if "citation.precision.explanation" in m else {
        "value": None, "ci95": None, "n": 0, "verdict": "not measurable (no explanation lines were produced)"}
    pf = m.get("citation.precision.profile")
    comps = [{"name": "explanation lines: automatic re-check (consistency, not accuracy)", **cp}]
    if pf:
        comps.append({"name": "profile facts' citations (extraction evidence, not this layer)", **pf})
    xrow: dict[str, Any] = {"layer": "Explanations", "question": "Are the reasons true and useful?",
                            "target": ">= 95% of cited quotes check out (citation precision)", "system": "full_engine",
                            "unit": "rate", "value": cp["value"], "ci95": cp["ci95"], "n": cp["n"]}
    clusters = f", {cp['n_clusters']} resumes" if cp.get("n_clusters") is not None else ""
    measured = (f"explanation lines, automatic re-check (consistency, not accuracy; deterministic substring matcher, "
                f"not the live LLM matcher) {fmt_value(cp)} (n={cp['n']}{clusters})")
    if audit_results and audit_results.get("n_ratings"):
        ah = m.get("audit.citation_right")
        comps.append({"name": "human audit", **(ah or {})})
        if ah:
            measured += (f"; human audit {fmt_value(ah)} (n={ah['n']}), mean usefulness "
                         f"{audit_results['usefulness']['mean']:.1f}/5")
        xrow["verdict"] = combine([cp["verdict"], ah["verdict"]]) if ah else cp["verdict"]
    else:
        measured += "; human audit not imported"
        xrow["verdict"] = (cp["verdict"] if category(cp["verdict"]) == "fail"
                           else "not measurable (human audit pending: import ratings with `audit import`)")
    xrow.update(measured=measured, components=comps)
    rows.append(xrow)
    return rows


def _caveats(raw: Mapping[str, Any], audit_results: Mapping[str, Any] | None) -> list[dict[str, str]]:
    c = [{"kind": "synthetic_labels", "text": raw["label"]},
         {"kind": "scope", "text": cit.CIRCULARITY_NOTE},
         {"kind": "scope", "text": "Evidence-level agreement with E3's gold levels is a consistency check: both follow "
                                   "the same L3 ladder by construction (evidence/T14/review.md)."},
         {"kind": "scope", "text": "Institution facts are excluded by design: L3 refuses institution names "
                                   "(PII-blind boundary), so institution recall is not measurable."},
         {"kind": "scope", "text": "Explanation lines are scored with a deterministic lexical matcher "
                                   f"({raw['settings']['matcher']}), so they test the ladder and the citations, not "
                                   "the live LLM matcher."}]
    for part, why in sorted(raw["errors"].items()):
        c.append({"kind": "system_partial", "where": part, "text": f"infrastructure, not a result: {why}"})
    if not (audit_results and audit_results.get("n_ratings")):
        c.append({"kind": "not_measurable", "where": "explanations.human_audit",
                  "text": "human audit not imported (agents cannot grade)"})
    return c


def _md(res: Mapping[str, Any]) -> str:
    s, mt = res["settings"], res["metrics"]
    L = [f"# Claims report (E11): run {res['run_id']}", "", f"> **SYNTHETIC DATA.** {res['label']}", ""]
    sm, it = res["summary"], res["intake"]
    L += [f"Status: **{res['status']}**. Seed {s['seed']}, pool {s['pool_size']}, {s['n_bases']} A7 bases, "
          f"{s['n_jds']} JDs, offline={s['offline']}, live LLM calls {res['live_calls_used']} of {s['max_llm_calls']}.",
          f"Intake: {it['n_processed']} of {it['n_planned']} resumes processed, outcomes {it['outcomes']}; "
          f"{it['n_l2_claims']} L2 claims, {it['n_verified_facts']} verified facts, L3 rejections "
          f"{it['l3_rejections_by_code'] or 'none'}.", ""]
    if res["errors"]:
        L += ["**Infrastructure (not results):** " + "; ".join(f"{k}: {v}" for k, v in sorted(res["errors"].items())), ""]
    L += [f"Targets: {sm['pass']} pass, {sm['fail']} fail, {sm['inconclusive']} inconclusive, "
          f"{sm['not measurable']} not measurable (of {sm['slide9_rows']} slide-9 layers).", "", "## Slide-9 rows", ""]
    L += _table(("Layer", "Target", "Measured (95% CI)", "n", "Verdict"),
                [(r["layer"], r["target"], r["measured"], "n/a" if r["n"] is None else r["n"], f"**{r['verdict']}**")
                 for r in res["slide9"]])
    cs = res["citation"]
    if s.get("l3_verifier") == "ollama":
        L += ["**Experimental local L3:** citation support below uses word overlap and is a lexical diagnostic, "
              "not semantic ground truth. See the independently labelled NLI report for classifier accuracy.", ""]
    L += ["## 1. Citation precision", "",
          f"Unit: {cs['unit']}. Target {100 * cs['target']:.0f}% (slide 9); n >= {cs['min_n']} for a verdict.", ""]
    L += _table(("Source", "Precision (95% CI)", "n (clusters)", "Verdict", "Failures"),
                [(src, fmt_value({**b, "unit": "rate"}), f"{b['n']} ({b['n_clusters']} resumes)",
                  NO_TARGET + " (context: L2's raw claims before L3)" if src == "l2_raw" else norm_verdict(b["status"]),
                  b["failures"] or "none") for src, b in cs["by_source"].items()])
    L += [f"Claim level (every citation of a claim passes): {fmt_value({**cs['claim_level'], 'unit': 'rate'})} "
          f"(n={cs['claim_level']['n']}).", ""]
    if res.get("judge"):
        j = res["judge"]
        L += [f"Judge (opt-in, free-only, cached): {j['n_judged']} of {j['requested']} judged, "
              f"{j['unavailable']} unavailable ({j['unavailable_reasons'] or 'none'}); supported "
              f"{fmt_value({**j['judge_supported'], 'unit': 'rate'})}.", ""]
    L += ["## 2. Extraction against gold facts (synthetic)", ""]
    ex = res.get("extraction")
    if ex:
        rows = []
        for typ, b in ex["by_type"].items():
            p, r = b["precision"], b["recall"]
            rows.append((typ, fmt_value({**p, "unit": "rate"}), p.get("n"), fmt_value({**r, "unit": "rate"}),
                         r.get("n"), norm_verdict(r["status"]) if typ == "institution"
                         else combine([norm_verdict(p["status"]), norm_verdict(r["status"])])))
        o = ex["overall"]
        rows.append(("**overall (micro)**", fmt_value({**o["precision"], "unit": "rate"}), o["precision"]["n"],
                     fmt_value({**o["recall"], "unit": "rate"}), o["recall"]["n"],
                     combine([norm_verdict(o["precision"]["status"]), norm_verdict(o["recall"]["status"])])))
        L += _table(("Fact type", "Precision (95% CI)", "n", "Recall (95% CI)", "n", "Verdict vs 85%"), rows)
        lk = ex["institution_leaks"]
        L += [f"Institution leaks (verified facts that name the resume's college; PII-blind boundary, target 0): "
              f"{lk['k']} of {lk['n']} facts over {lk.get('n_clusters', 0)} resumes, verdict "
              f"**{norm_verdict(lk['status'])}**.", f"n is {ex['unit']}.", ""]
        lv = ex["skill_level"]
        L += [f"Skill level against gold (n={lv['n']}): agreement {fmt_value({**lv['agreement'], 'unit': 'rate'})}, "
              f"predicted above gold {fmt_value({**lv['predicted_above_gold'], 'unit': 'rate'})}, predicted below gold "
              f"{fmt_value({**lv['predicted_below_gold'], 'unit': 'rate'})}; {lv['caveat']}.", ""]
    else:
        L += [f"not measurable ({res['errors'].get('extraction', 'no verified profiles')})", ""]
    a7, infl = res["a7"], res["inflation"]
    L += ["## 3. Unsupported claims (A7)", "",
          f"{a7['definition']}. Target 0 (slide 8: capped at E1). Rate {fmt_value({**a7, 'unit': 'rate'})} "
          f"(k={a7['k']}, n={a7['n']} skills over {a7['n_twins']} twins), verdict **{norm_verdict(a7['status'])}**. "
          f"Twin outcomes {a7['twin_outcomes']}. {a7['note']}", "",
          "## 4. Evidence-level inflation", "",
          f"Facts whose level exceeds what the cited span supports: {fmt_value({**infl, 'unit': 'rate'})} "
          f"(k={infl['k']}, n={infl['n']}), verdict **{norm_verdict(infl['status'])}**; by kind "
          f"{infl['inflated_by_kind'] or 'none'}. {infl['circularity']}", "",
          "## 5. Human audit", ""]
    au = res.get("audit")
    if au and au.get("n_ratings"):
        cr = au["citation_right"]
        L += [f"{au['n_ratings']} ratings over {au['n_rows_rated']} rows ({au['n_pending']} pending, {au['n_rejected']} "
              f"rejected). Citation right {fmt_value({**cr, 'unit': 'rate'})} ({norm_verdict(cr['status'])}); mean "
              f"usefulness {au['usefulness']['mean']:.2f}/5; agreement with the automatic check "
              f"{fmt_value({**au['agreement_with_automatic_check'], 'unit': 'rate'})}. {au.get('note', '')}", ""]
        if au.get("rows_with_edited_content"):
            L += ["**Warning:** the explanation, quote or span cells were edited by the rater on rows "
                  + ", ".join(au["rows_with_edited_content"]) + "; those ratings are kept but check them.", ""]
    else:
        L += [f"{res['explanations']['audit_rows_exported']} lines exported to `audit/audit_sheet.csv` and "
              "`audit_sheet.md`; no ratings imported yet. Humans rate usefulness 1-5 and mark each citation right "
              "or wrong, then run `audit import <run_dir> <csv>`.", ""]
    L += ["## Failure examples (ids only)", ""]
    L += [f"- {f['source']} {f['candidate_id']} {f['unit']} span {f['span_id'] or '-'}: {f['failure']}"
          for f in res["failure_examples"]] or ["none"]
    L += ["", "## Caveats", ""]
    L += [f"- {c['text']}" if "where" not in c else f"- `{c['where']}`: {c['text']}" for c in res["caveats"]]
    L += ["", "## Run manifest", "", "```json", json.dumps(res["manifest"], indent=2, sort_keys=True), "```", ""]
    return "\n".join(L)


def build_claims_report(run_dir: str | Path) -> dict[str, Any]:
    """Write ``results.json``, ``report.md`` and ``per_item.jsonl`` for a claims run, merging any imported audit.
    Makes no model calls, so re-running it (after ``audit import``) costs nothing."""
    d = Path(run_dir)
    try:
        raw = json.loads((d / RAW_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{d.name}: not a finished run (need {RAW_FILE} and manifest.json)") from exc
    res = derive_claims_results(raw, manifest, audit.load_results(d))
    for name, text in ((RESULTS_FILE, json.dumps(res, indent=1, sort_keys=True, allow_nan=False)), (REPORT_FILE, _md(res)),
                       (PER_ITEM_FILE, "".join(json.dumps(clean_json(i), sort_keys=True, allow_nan=False) + "\n"
                                               for i in raw.get("items", [])))):
        p = d / name
        if p.is_symlink():
            p.unlink()
        p.write_text(text, encoding="utf-8")
    return res


__all__ = ["ClaimsSettings", "IntakeRecord", "SubstringMatcher", "build_claims_report", "derive_claims_results",
           "evaluate_record", "execute_claims", "explanation_checks", "probe_profile", "run_claims", "run_intake"]
