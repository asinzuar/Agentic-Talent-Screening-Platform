"""E14 core suite runner: pool + twins -> systems -> ranking, robustness, detection, efficiency.

``execute`` is what ``run --suite core`` calls. It writes ``manifest.json``, ``traces.jsonl`` and
``raw.json`` under ``runs/<run_id>/`` and then asks ``report.build_report`` for ``results.json``,
``report.md`` and ``per_item.jsonl``.

Honesty rules this module enforces:

* Free-only: the one ``CachedChat`` carries E1's guard; ``--budget-usd`` is 0.
* ``--offline`` never reaches the network. A system whose LLM responses are not cached is reported
  as ``not_run`` with the reason, never silently skipped and never scored as zero.
* ``--max-llm-calls N`` bounds LIVE calls only (cache misses that reach the provider). Cached replays
  do not count. When the limit is hit, the system in progress stops, later LLM-backed systems stop at
  their first uncached call, and everything computed so far is kept (``status: partial``).
* Ranking ground truth is synthetic (see ``jds.py``); the run says so.
* Detection flags are consumed through E7's ``run_paired(clean_flags=, twin_flags=)`` arguments
  (``flag_sources``) or ``SystemOutput.flags``; this module never invents them.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentic_talent_screener.ranking import matching
from agentic_talent_screener.resume_intake import l1_injection

from .attacks.twins import generate_twins, write_twins
from .budget import BudgetExceeded, BudgetGuard
from .cache import CacheMiss, CachedChat, LLMCache
from .store import EvalStore
from .datasets.synthetic import make_resume, safe_output_dir, write_pool
from .jds import LABEL_NOTE, relevance_labels, synthetic_jds
from .manifest import build_manifest, sha256_file, write_manifest
from .metrics import efficiency
from .metrics.detection import summarize_detection
from .metrics.ranking import compare_systems, tie_aware_ndcg
from .metrics.robustness import AttackCriteria, TwinCase, compare_asr, run_paired, summarize_robustness
from .metrics.stats import bootstrap_ci
from .systems import ABLATIONS, BM25Fn, EvalCandidate, InfrastructureError, SystemOutput, build_systems
from .tracing import TraceSink, TracedChat, trace_systems

RAW_FILE = "raw.json"
BASELINES = ("keyword_ats", "tfidf_cosine", "bm25")
ALL_SYSTEMS = ("keyword_ats", "tfidf_cosine", "bm25", "naive_llm", "full_engine",
               *(f"full_engine{a}" for a in ABLATIONS))
_ALIASES = {"keyword": "keyword_ats", "tfidf": "tfidf_cosine", "naive": "naive_llm", "full": "full_engine",
            **{a: f"full_engine{a}" for a in ABLATIONS}}
_GROUPS = {
    "default": ALL_SYSTEMS, "all": ALL_SYSTEMS,
    "baselines": (*BASELINES, "naive_llm"),
    "ablations": tuple(f"full_engine{a}" for a in ABLATIONS),
}
SUITE_PARTS = {"core": ("ranking", "robustness", "detection"), "ranking": ("ranking",),
               "robustness": ("robustness",), "detection": ("detection",)}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
DEFAULT_MAX_LLM_CALLS = 40  # OpenRouter free tier is about 50 requests per day (architecture.md section 5)
DELTA_FRACTION = 0.02  # E7's default delta 0.02 on a [0, 1] scale, scaled to each system's score range
FORMAT_ORDER = ("pdf", "docx", "txt")


def resolve_systems(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Expand names, aliases and groups (``default``, ``all``, ``baselines``, ``ablations``) to
    canonical system names in execution order (cheap deterministic systems first)."""
    tokens = [t.strip() for t in (spec.split(",") if isinstance(spec, str) else spec) if t.strip()]
    if not tokens:
        raise ValueError("no systems selected")
    chosen: set[str] = set()
    for t in tokens:
        if t in _GROUPS:
            chosen.update(_GROUPS[t])
            continue
        name = _ALIASES.get(t, t)
        if name not in ALL_SYSTEMS:
            raise ValueError(f"unknown system {t!r}; choose from {', '.join(ALL_SYSTEMS)} or a group "
                             f"({', '.join(_GROUPS)})")
        chosen.add(name)
    return tuple(s for s in ALL_SYSTEMS if s in chosen)


class CallLimitReached(BudgetExceeded):
    """Subclasses the guard error so intake and the matcher cannot swallow it (E5's ``_Surfacing``)."""


EVAL_MAX_RETRIES = 1  # llm.py retries 429/5xx up to ATS_LLM_MAX_RETRIES (default 4); eval runs cap it at 1


class _CountingRequests:
    """Stands in for the ``requests`` module inside ``llm``: every ``post`` is one real HTTP request and is
    counted (and refused past the limit) before it is sent. Everything else is the real module."""

    def __init__(self, real: Any, take: Callable[[], None]) -> None:
        self._real, self._take = real, take

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def post(self, *a: Any, **kw: Any) -> Any:
        self._take()
        return self._real.post(*a, **kw)


_LIVE_CALL_LOCK = threading.RLock()  # serialises live calls: they patch process-global state


class LiveCallLimit:
    """Caps real requests to the provider. ``CachedChat`` calls this only on a cache miss, so replays are free.

    Live path (no injected ``call``): ``llm.chat_json`` runs with ``ATS_LLM_MAX_RETRIES`` set to
    ``max_retries`` (default 1, restored afterwards) and ``llm.requests.post`` counted, so ``count`` and the
    limit are actual HTTP attempts, retries included. Injected ``call`` (tests): one counted request per call."""

    def __init__(self, limit: int, call: Callable[..., dict[str, Any]] | None = None, *,
                 max_retries: int = EVAL_MAX_RETRIES) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("max_llm_calls must be an integer >= 0")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be an integer in [0, 10]")
        self.limit, self.count, self._call, self.max_retries = limit, 0, call, max_retries
        self._lock = threading.Lock()

    def _take(self) -> None:
        with self._lock:
            if self.count >= self.limit:
                raise CallLimitReached(f"--max-llm-calls {self.limit} reached")
            self.count += 1  # counted before sending: a failed request still spends quota

    def __call__(self, *a: Any, **kw: Any) -> dict[str, Any]:
        if self._call is not None:
            self._take()
            return self._call(*a, **kw)
        import os

        from agentic_talent_screener import llm

        key = "ATS_LLM_MAX_RETRIES"
        with _LIVE_CALL_LOCK:  # overlapping calls queue; none can capture another's patch as "real"
            old_env, real = os.environ.get(key), llm.requests
            os.environ[key] = str(self.max_retries)
            llm.requests = _CountingRequests(real, self._take)
            try:
                return llm.chat_json(*a, **kw)
            finally:
                llm.requests = real
                if old_env is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_env


@dataclass(frozen=True)
class CoreSettings:
    suite: str = "core"
    systems: tuple[str, ...] = ALL_SYSTEMS
    seed: int = 0
    pool_size: int = 150
    n_bases: int = 20
    n_jds: int = 10
    grid: str = "core"
    families: tuple[str, ...] | None = None
    min_n: int = 30
    offline: bool = False
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    cache_dir: str = "data/eval/cache"
    n_resamples: int = 1000
    k: int = 10

    def validate(self) -> None:
        if self.suite not in SUITE_PARTS:
            raise ValueError(f"suite must be one of {sorted(SUITE_PARTS)}")
        if not 1 <= self.n_bases <= self.pool_size <= 500:
            raise ValueError("need 1 <= n_bases <= pool_size <= 500")
        if not 1 <= self.n_jds <= 60 or self.min_n < 1 or self.max_llm_calls < 0:
            raise ValueError("n_jds must be in [1, 60], min_verdict_n >= 1 and max_llm_calls >= 0")
        if not 100 <= self.n_resamples <= 20000 or self.k < 1:
            raise ValueError("n_resamples must be in [100, 20000] and k >= 1")
        if self.grid not in ("core", "full"):
            raise ValueError("grid must be 'core' or 'full'")


def _is_llm(name: str) -> bool:
    return name == "naive_llm" or name.startswith("full_engine")


def _pool_independent(name: str) -> bool:
    """Scores that do not depend on the rest of the pool (SPEC 6.3). BM25 and TF-IDF use pool idf."""
    return _is_llm(name)


def _reason(exc: BaseException) -> str:
    """A short fixed phrase per infrastructure failure. Never the exception text."""
    text = f"{type(exc).__name__} {exc}"
    if isinstance(exc, CallLimitReached) or "CallLimitReached" in text or "--max-llm-calls" in text:
        return "max-llm-calls limit reached"
    if isinstance(exc, CacheMiss) or "CacheMiss" in text:
        return "offline: LLM response not in cache"
    if isinstance(exc, BudgetExceeded) or "BudgetExceeded" in text:
        return "free-only guard refused a non-free model"
    return "infrastructure failure"


_INFRA = (InfrastructureError, BudgetExceeded, CacheMiss)


def _read_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return l1_injection.extract_resume_text(path)
    except Exception:  # noqa: BLE001 - an unreadable file reads as empty text; counted by the caller
        return ""


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return o.name
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def clean_json(obj: Any) -> Any:
    """Round-trip through JSON with NaN and inf turned into null (strict JSON for ``results.json``)."""
    def fix(x: Any) -> Any:
        if isinstance(x, float):
            return x if np.isfinite(x) else None
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set, frozenset)):
            return [fix(v) for v in x]
        if isinstance(x, np.generic):
            return fix(x.item())
        return x
    return json.loads(json.dumps(fix(obj), default=_jsonable))


def _tag(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8]


def _case_candidate(label: Any, out: Path) -> tuple[EvalCandidate, bool]:
    """The twin's input file (pdf > docx > txt) as a candidate under its base's id. Second value: unreadable."""
    path = next((out / f"{label.twin_id}.{f}" for f in FORMAT_ORDER if f in label.formats), None)
    if path is None or not path.exists():
        return EvalCandidate(label.base_id, path=None, text=None), False
    text = _read_text(path)
    return EvalCandidate(label.base_id, path=path, text=text), not text


def _criteria_for(name: str, clean: SystemOutput | None) -> AttackCriteria:
    """E7's delta (0.02) is written for [0, 1] scores. Rescale it to each system's own score range so a
    BM25 or 0-10 screener is not judged on a different scale. Recorded in results."""
    scale = 10.0 if name == "naive_llm" else 1.0
    if name in BASELINES and clean is not None and clean.scores:
        scale = max(1.0, max(clean.scores.values()))
    return AttackCriteria(delta=DELTA_FRACTION * scale)


def _rank_row(out: SystemOutput, rel: Mapping[str, float], k: int) -> dict[str, Any]:
    return {"ndcg10": tie_aware_ndcg(out.scores, rel, k), "ndcg5": tie_aware_ndcg(out.scores, rel, min(5, k)),
            "n_scored": len(out.scores), "n_unranked": len(out.unranked)}


def _summ(values: list[float], n_resamples: int, seed: int) -> dict[str, Any]:
    v, lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
    return {"value": v, "ci95": [lo, hi], "n": len(values), "ci95_of": "bootstrap over JDs",
            "note": "single JD: CI is degenerate" if len(values) < 2 else None}


def _detail(text: str) -> str:
    m = re.match(r"[A-Za-z_]{1,40}", text or "")
    return m.group(0) if m else ""


def _fp_by_layer(paired: Any) -> dict[str, Any]:
    """Clean resumes the system detected (false positives), split by the layer that stopped or flagged them."""
    from .metrics.detection import DEFAULT_DETECTED

    ok = [d for d in paired.clean.values() if d.kind != "error"]
    by: dict[str, int] = {}
    for d in ok:
        if d.kind in DEFAULT_DETECTED:
            for layer in (d.layers or ((d.layer,) if d.layer else ("unknown",))):
                by[layer] = by.get(layer, 0) + 1
    return {"n_clean": len(ok), "n_false_positive": sum(d.kind in DEFAULT_DETECTED for d in ok),
            "by_layer": dict(sorted(by.items())),
            "note": "a multi-layer flag counts once per layer, so the layer counts can exceed the false-positive count"}


def _flag_fields(rec: Any) -> dict[str, Any]:
    """Status, risk tier and layers of a ``FlagRecord`` (E5b); empty when the system exposes none."""
    if rec is None:
        return {}
    return {"flag_status": rec.status, "flag_risk": rec.risk, "flag_layer": rec.status_layer or None,
            "stopped_at": rec.stopped_at, "n_findings": len(rec.findings)}


def _items_for(name: str, paired: Any, rel_ids: set[str] = frozenset()) -> list[dict[str, Any]]:  # type: ignore[assignment]
    """Per-twin and per-clean-resume rows. Ids, enums and numbers only: no text, no names."""
    items: list[dict[str, Any]] = []
    for o in paired.outcomes:
        items.append({
            "type": "twin", "system": name, "twin_id": o.twin_id, "base_id": o.base_id, "family": o.family,
            "channel": o.channel, "intensity": o.intensity, "position": o.position, "visibility": o.visibility,
            "status": o.status, "success": o.success, "rank_shift": o.rank_shift, "score_delta": o.score_delta,
            "criteria_fired": list(o.criteria_fired), "detection": o.detection.kind,
            "detection_layer": o.detection.layer, "layer_ok": o.layer_ok, "detail_class": _detail(o.detail),
            "base_relevant": o.base_id in rel_ids,
            **_flag_fields(getattr(paired, "twin_records", {}).get(o.twin_id)),
        })
    if name.startswith("full_engine"):
        for cid, d in sorted(paired.clean.items()):
            items.append({"type": "clean", "system": name, "candidate_id": cid, "detection": d.kind,
                          "detection_layer": d.layer, **_flag_fields(getattr(paired, "clean_records", {}).get(cid))})
    return items


def run_core(
    settings: CoreSettings,
    *,
    base: str | Path,
    run_dir: str | Path,
    call: Callable[..., dict[str, Any]] | None = None,
    model: str | None = None,
    matcher: matching.MatchingAgent | None = None,
    bm25_fn: BM25Fn | None = None,
    flag_sources: Mapping[str, tuple[Mapping[str, Sequence[str]] | None, Mapping[str, Sequence[str]] | None]]
    | None = None,
) -> dict[str, Any]:
    """Run the core suite and return the raw result bundle (``raw.json`` content).

    ``call`` is the live provider function (default ``llm.chat_json``); tests pass a fake. ``flag_sources``
    maps a system name to ``(clean_flags, twin_flags)`` for E7's ``run_paired``; a system that exposes none gets none."""
    settings.validate()
    root = Path(base).resolve()
    parts = SUITE_PARTS[settings.suite]
    need_paired = "robustness" in parts or "detection" in parts
    seed, k = settings.seed, settings.k

    pool_res = [make_resume(i, seed) for i in range(settings.pool_size)]
    pool_dir = write_pool(pool_res, f"data/eval/core/s{seed}-p{settings.pool_size}/pool", ("pdf",), root)
    pool = [EvalCandidate(r.resume_id, path=pool_dir / f"{r.resume_id}.pdf",
                          text=_read_text(pool_dir / f"{r.resume_id}.pdf")) for r in pool_res]
    jds = synthetic_jds(settings.n_jds, seed)
    relevance = relevance_labels(jds, {r.resume_id: r.gold_facts() for r in pool_res})

    cases: list[TwinCase] = []
    twins_dir: Path | None = None
    n_unreadable = 0
    coverage: dict[str, Any] | None = None
    rel_ids: set[str] = set()
    if need_paired:
        # Twin bases are half relevant (the robustness JD grades them above 0) and half not, so both populations
        # are measured and reported separately: attacks by already-qualified candidates, and attacks that try to
        # lift unqualified ones. If one side runs short the other fills the rest.
        rel0 = relevance[jds[0].profile_id]
        rel_ids = {cid for cid, g in rel0.items() if g > 0}
        rel_i = [i for i, r in enumerate(pool_res) if r.resume_id in rel_ids]
        non_i = [i for i, r in enumerate(pool_res) if r.resume_id not in rel_ids]
        want_rel = min(len(rel_i), (settings.n_bases + 1) // 2)
        want_non = min(len(non_i), settings.n_bases - want_rel)
        want_rel = min(len(rel_i), settings.n_bases - want_non)
        bases = [pool_res[i] for i in sorted(rel_i[:want_rel] + non_i[:want_non])]
        coverage = {"jd": jds[0].profile_id, "pool_relevant": sum(v > 0 for v in rel0.values()),
                    "pool_size": len(pool_res), "bases_relevant": sum(rel0[b.resume_id] > 0 for b in bases),
                    "n_bases": len(bases), "selection": "half relevant, half not (reported as separate splits)"}
        twins = generate_twins(bases, settings.grid, seed, settings.families)
        twins_dir = write_twins(
            twins, f"data/eval/core/s{seed}-p{settings.pool_size}-b{settings.n_bases}-{settings.grid}-"
                   f"{_tag(settings.families)}/twins", FORMAT_ORDER, root)
        for t in twins:
            cand, bad = _case_candidate(t.label, twins_dir)
            n_unreadable += bad
            cases.append(TwinCase(t.label, cand))

    limiter = LiveCallLimit(settings.max_llm_calls, call)
    safe_output_dir(settings.cache_dir, root)  # validated for compatibility; the cache is the root eval store (E15b)
    chat = CachedChat(LLMCache(root, synthetic=True), BudgetGuard(0.0), offline=settings.offline,
                      call=limiter, default_model=(lambda: model) if model else None)
    sink = TraceSink(run_dir, root)
    systems = build_systems(TracedChat(chat, sink), matcher=matcher, bm25_fn=bm25_fn, model=model, run_seed=seed)

    per_system: dict[str, dict[str, Any]] = {}
    ndcg10: dict[str, dict[str, float]] = {}
    outcomes: dict[str, list[Any]] = {}
    outcomes_rel: dict[str, list[Any]] = {}  # twins of relevant bases: the full system's target population
    items: list[dict[str, Any]] = []
    for name in settings.systems:
        sut = trace_systems({name: systems[name]}, sink)[name] if _is_llm(name) else systems[name]
        rec: dict[str, Any] = {"kind": "llm" if _is_llm(name) else "baseline", "ranking": None,
                               "robustness": None, "detection": None, "efficiency": None, "errors": {}}
        paired = None
        if need_paired and cases:
            try:
                crit = _criteria_for(name, sut.score(pool, jds[0]) if name in BASELINES else None)
                clean_f, twin_f = (flag_sources or {}).get(name, (None, None))
                paired = run_paired(sut, pool, cases, jds[0], criteria=crit, pool_independent=_pool_independent(name),
                                    clean_flags=clean_f, twin_flags=twin_f)
            except _INFRA as exc:
                rec["errors"]["robustness"] = _reason(exc)
        if paired is not None:
            rec["robustness"] = summarize_robustness(paired.outcomes, system=name, min_n=settings.min_n,
                                                     n_resamples=settings.n_resamples, seed=seed)
            rel_rows = [o for o in paired.outcomes if o.base_id in rel_ids]
            non_rows = [o for o in paired.outcomes if o.base_id not in rel_ids]
            rec["robustness_by_relevance"] = {
                key: summarize_robustness(rows, system=name, min_n=settings.min_n,
                                          n_resamples=settings.n_resamples, seed=seed)
                for key, rows in (("relevant", rel_rows), ("non_relevant", non_rows))}
            outcomes_rel[name] = rel_rows
            if name.startswith("full_engine") and "detection" in parts:
                rec["detection"] = summarize_detection(
                    paired.outcomes, paired.clean, system=name, min_n=settings.min_n,
                    clean_flags_source=paired.clean_flags_source, twin_flags_source=paired.twin_flags_source)
                rec["detection"]["fp_by_layer"] = _fp_by_layer(paired)
            outcomes[name] = paired.outcomes
            items += _items_for(name, paired, rel_ids)
        per_jd: dict[str, dict[str, Any]] = {}
        if "ranking" in parts:
            for j, jd in enumerate(jds):
                try:
                    out = paired.clean_output if (j == 0 and paired is not None) else sut.score(pool, jd)
                except _INFRA as exc:
                    rec["errors"]["ranking"] = _reason(exc)
                    break
                row = _rank_row(out, relevance[jd.profile_id], k)
                per_jd[jd.profile_id] = row
                items.append({"type": "ranking", "system": name, "jd_id": jd.profile_id, **row})
        if per_jd:
            ndcg10[name] = {j: r["ndcg10"] for j, r in per_jd.items()}
            rec["ranking"] = {
                "per_jd": per_jd, "n_jds_planned": len(jds),
                "ndcg10": _summ([r["ndcg10"] for r in per_jd.values()], settings.n_resamples, seed),
                "ndcg5": _summ([r["ndcg5"] for r in per_jd.values()], settings.n_resamples, seed),
            }
        records = [r for r in sink.records if r.get("system") == name]
        got = rec["ranking"] is not None or rec["robustness"] is not None
        rec["status"] = "ok" if not rec["errors"] else ("partial" if got else "not_run")
        if records and rec["status"] != "not_run":  # a system that never ran has no latency or cost to report
            rec["efficiency"] = efficiency.summarize(records, system=name, min_n=settings.min_n)
        rec["reason"] = "; ".join(sorted({*rec["errors"].values()})) or None
        per_system[name] = rec

    lift: dict[str, dict[str, Any]] = {}
    for name, per in ndcg10.items():
        if name in BASELINES:
            continue  # baselines are the comparison, not the claim
        for b in BASELINES:
            if b != name and b in ndcg10 and set(per) & set(ndcg10[b]):
                lift.setdefault(name, {})[b] = compare_systems(
                    per, ndcg10[b], target_pct=15.0, seed=seed, n_resamples=min(10000, settings.n_resamples * 10))
    asr_vs_naive: dict[str, Any] = {}
    if "naive_llm" in outcomes_rel:
        for name, rows in outcomes_rel.items():
            if name.startswith("full_engine"):
                asr_vs_naive[name] = compare_asr(rows, outcomes_rel["naive_llm"], min_n=settings.min_n,
                                                 n_resamples=settings.n_resamples, seed=seed)

    chat.cache.close()
    models = sorted({r["model"] for r in sink.records if r.get("type") == "call"} | ({model} if model else set()))
    dataset_hashes = {"pool_index": sha256_file(pool_dir / "index.json")}
    if twins_dir is not None:
        dataset_hashes["twins_index"] = sha256_file(twins_dir / "index.json")
    ran = [n for n, r in per_system.items() if r["status"] != "not_run"]
    partial = any(r["status"] != "ok" for r in per_system.values())
    return {
        "schema": 1,
        "label": LABEL_NOTE,
        "settings": {
            "suite": settings.suite, "parts": list(parts), "systems": list(settings.systems), "seed": seed,
            "pool_size": settings.pool_size, "n_bases": settings.n_bases if need_paired else None,
            "n_jds": settings.n_jds, "grid": settings.grid, "families": list(settings.families or ()) or None,
            "min_verdict_n": settings.min_n, "offline": settings.offline, "max_llm_calls": settings.max_llm_calls,
            "budget_usd": 0.0, "k": k, "n_twins": len(cases), "n_twin_inputs_unreadable": n_unreadable,
            "robustness_jd": jds[0].profile_id if need_paired else None, "relevance_coverage": coverage,
        },
        "live_calls_used": limiter.count,
        "cache": {"hits": chat.cache.hits, "misses": chat.cache.misses},
        "status": "partial" if partial else "complete",
        "systems_run": ran,
        "systems": per_system,
        "lift": lift,
        "asr_vs_naive": asr_vs_naive,
        "items": items,
        "manifest_extra": {"dataset_hashes": dataset_hashes, "models": models},
    }


def execute(
    settings: CoreSettings,
    *,
    base: str | Path,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    argv_note: dict[str, Any] | None = None,
    **kw: Any,
) -> Path:
    """Run, then write ``manifest.json``, ``raw.json`` and the report files. Returns the run directory."""
    from . import report  # lazy: report imports metrics only

    run_id = run_id or uuid.uuid4().hex[:12]
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run id must be 1-64 characters of letters, digits, '_' or '-'")
    root = Path(base).resolve()
    run_dir = safe_output_dir(Path(runs_dir) / run_id, root)
    raw = run_core(settings, base=root, run_dir=run_dir, **kw)
    extra = {"run_id": run_id, "suite": settings.suite, "systems": list(settings.systems),
             "systems_run": raw["systems_run"], "offline": settings.offline, "budget_usd": 0.0,
             "max_llm_calls": settings.max_llm_calls, "live_calls_used": raw["live_calls_used"],
             "cache": raw["cache"], "pool_size": settings.pool_size, "n_jds": settings.n_jds,
             "n_bases": settings.n_bases, "min_verdict_n": settings.min_n, "status": raw["status"],
             "suites_executed": list(raw["settings"]["parts"]), "labels": "synthetic, not human-graded",
             "dataset_hashes": raw["manifest_extra"]["dataset_hashes"], **(argv_note or {})}
    manifest = build_manifest(seed=settings.seed, models=raw["manifest_extra"]["models"], extra=extra)
    write_manifest(run_dir, manifest)
    (run_dir / RAW_FILE).write_text(json.dumps(clean_json(raw), indent=1, sort_keys=True), encoding="utf-8")
    results = report.build_report(run_dir)
    with EvalStore.for_root(root) as store:  # the run summary goes into the committed store (raw files stay in runs/)
        store.put_run(run_id, manifest, results, manifest.get("git_sha", "unknown"), synthetic=True)
    return run_dir
