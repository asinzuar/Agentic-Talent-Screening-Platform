"""E12 metrics from traces: latency, cost (modelled, not measured), reliability, slide-9 verdicts.

Everything here is computed from ``traces.jsonl`` records (metadata only). With free models the
measured cost is $0; the per-resume cost reported against the INR 1 target is a MODELLED
figure from ESTIMATED tokens times a configurable price table.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Defaults from architecture.md section 5 (O3: paid Gemma 4 31B, cheapest listed providers).
# Assumptions; verify before relying on them.
DEFAULT_USD_PER_M_INPUT = 0.10
DEFAULT_USD_PER_M_OUTPUT = 0.34
DEFAULT_INR_PER_USD = 85.0  # assumption
TARGETS = {"latency_s_max": 20.0, "inr_per_resume": 1.0, "schema_valid": 0.99}
DEFAULT_MIN_N = 30  # mirrors EvalConfig.min_verdict_n
INFRA = ("budget_guard", "cache_miss", "infrastructure")


@dataclass(frozen=True)
class PriceTable:
    usd_per_m_input: float = DEFAULT_USD_PER_M_INPUT
    usd_per_m_output: float = DEFAULT_USD_PER_M_OUTPUT
    inr_per_usd: float = DEFAULT_INR_PER_USD
    label: str = "assumed paid Gemma 4 31B (architecture.md section 5, O3)"

    def __post_init__(self) -> None:
        for name in ("usd_per_m_input", "usd_per_m_output", "inr_per_usd"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite number >= 0")

    def usd(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens * self.usd_per_m_input + completion_tokens * self.usd_per_m_output) / 1e6


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _prop(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"value": (k / n) if n else None, "n": n, "ci95": [lo, hi]}


NOT_MEASURABLE = "not measurable"
MIN_QUALIFIED_SHARE = 0.5  # latency verdict needs at least this share of resumes timed per-resume and live


def _latency(resumes: list[dict[str, Any]], min_n: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Latency stats plus verdict. Only per-resume-timed, non-replayed resumes qualify; pool means
    and cache replays never produce a max."""
    total = len(resumes)
    errored = [r for r in resumes if r.get("status") == "error"]
    ok = [r for r in resumes if r.get("status") != "error"]
    qualified = [r for r in ok if r["timing_basis"] == "exact" and r.get("cache_hits", 0) == 0]
    pooled = [r for r in ok if r["timing_basis"] != "exact"]
    info: dict[str, Any] = {
        "n_resumes": total, "n_qualified": len(qualified), "n_pool_mean": len(pooled),
        "n_with_cache_replay": sum(1 for r in ok if r.get("cache_hits", 0) > 0),
        "n_errored_excluded": len(errored),
    }
    if pooled:
        w = [r["wall_ms"] / 1000 for r in pooled]
        info["pool_mean_s_information_only"] = {"mean_of_means": sum(w) / len(w), "note": "a mean of pool means is not the slow case"}
    if qualified:
        w = [r["wall_ms"] / 1000 for r in qualified]
        info.update(p50=percentile(w, 50), p95=percentile(w, 95), max=max(w))
    timeouts = [r for r in errored if r.get("error_kind") == "timeout"]
    if timeouts:
        return info, {"status": "fail", "note": f"{len(timeouts)} resume(s) timed out", "target": TARGETS["latency_s_max"], "n": len(qualified), "ci95": None}
    if pooled:
        return info, {"status": NOT_MEASURABLE, "n": len(qualified), "ci95": None, "n_qualified": len(qualified),
                      "note": "pooled timing: only a pool mean exists; needs per-resume timing (C2 on_stage) or per-candidate runs"}
    if not total or len(qualified) < MIN_QUALIFIED_SHARE * total:
        return info, {"status": NOT_MEASURABLE, "n": len(qualified), "ci95": None, "n_qualified": len(qualified),
                      "note": "mostly or entirely cache replay / errors: latencies are not representative"}
    n = len(qualified)
    under = sum(1 for r in qualified if r["wall_ms"] / 1000 < TARGETS["latency_s_max"])
    common = {"value": info["max"], "target": TARGETS["latency_s_max"], "n": n, "n_qualified": n, "n_resumes": total,
              "ci95": list(wilson(under, n)), "ci95_of": "share of resumes under the target (Wilson)"}
    if info["max"] >= TARGETS["latency_s_max"]:  # an observed slow case is proof, whatever n
        return info, {"status": "fail", **common}
    if n < min_n:
        return info, {"status": f"{NOT_MEASURABLE} (n<min)", "min_n": min_n, **common}
    return info, {"status": "pass", "min_n": min_n, **common}


def _bootstrap_mean_ci(values: list[float], seed: int = 0, resamples: int = 1000) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def _judge_upper_target(lo: float, hi: float, target: float, n: int, min_n: int) -> str:
    """For 'must stay below target' metrics (cost): proven fail if the whole CI is at or above the target."""
    if lo >= target:
        return "fail"
    if n < min_n:
        return f"{NOT_MEASURABLE} (n<min)"
    return "pass" if hi < target else "inconclusive"


def _judge_lower_target(lo: float, hi: float, target: float, n: int, min_n: int) -> str:
    """For 'must reach target' metrics (schema-valid): proven fail if the CI upper bound is below the
    target; pass needs n >= min_n and the CI lower bound at or above the target."""
    if hi < target:
        return "fail"
    if n < min_n:
        return f"{NOT_MEASURABLE} (n<min)"
    return "pass" if lo >= target else "inconclusive"


def summarize(records: Iterable[dict[str, Any]], prices: PriceTable | None = None, *, system: str,
              min_n: int = DEFAULT_MIN_N) -> dict[str, Any]:
    """Summary and slide-9 verdicts for ONE system. Pooling systems would dilute cost, latency and
    reliability, so ``system`` is required; use ``summarize_all`` for a mixed run."""
    if not system:
        raise ValueError("system is required: never summarize several systems together")
    prices = prices or PriceTable()
    recs = [r for r in records if r.get("system") == system]
    calls = [r for r in recs if r["type"] == "call"]
    resumes = [r for r in recs if r["type"] == "resume"]
    out: dict[str, Any] = {"system": system, "n_resumes": len(resumes), "n_calls": len(calls)}

    out["latency_s_per_resume"], lat_verdict = _latency(resumes, min_n)
    stage: dict[str, list[float]] = defaultdict(list)
    for r in calls:
        stage[r["call_site"]].append(r["latency_ms"] / 1000)
    out["call_latency_s_by_call_site"] = {
        k: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v)}
        for k, v in sorted(stage.items())
    }

    model_calls = [r for r in calls if r["error_kind"] not in INFRA]
    out["infrastructure_failures"] = len(calls) - len(model_calls)
    # schema validity: calls where the model returned output; live (cache miss) only for the verdict
    with_output = [r for r in model_calls if r["schema_valid"] is not None]
    live = [r for r in with_output if r["cache"] == "miss"]
    replayed = [r for r in with_output if r["cache"] == "hit"]
    out["schema_valid"] = _prop(sum(1 for r in live if r["schema_valid"]), len(live))
    out["schema_valid"]["basis"] = "live calls only"
    out["schema_valid_replayed_biased"] = {
        **_prop(sum(1 for r in replayed if r["schema_valid"]), len(replayed)),
        "basis": "replayed outputs (biased: failures are not cached)",
    }
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in live:
        by_site[r["call_site"]].append(r)
    out["schema_valid_by_call_site"] = {
        k: _prop(sum(1 for r in v if r["schema_valid"]), len(v)) for k, v in sorted(by_site.items())
    }
    out["cache_hit_rate"] = _prop(sum(1 for r in calls if r["cache"] == "hit"), len(calls))
    out["retry_rate"] = {"value": None, "status": "not measurable until C1",
                         "note": "CachedChat calls the underlying function once per miss; llm.py retries are internal"}
    out["failure_kinds"] = dict(Counter(r["error_kind"] for r in calls if r["error_kind"]))
    out["terminal_failure_rate"] = _prop(sum(1 for r in model_calls if r["error_kind"]), len(model_calls))
    out["tool_call_correctness"] = "N/A: no agent has tools"
    out["calls_per_resume"] = (len(calls) / len(resumes)) if resumes else None

    sent = [r for r in calls if r["error_kind"] not in INFRA]  # infra-failed calls were never sent
    pt = sum(r["est_prompt_tokens"] for r in sent)
    ct = sum(r["est_completion_tokens"] for r in sent)
    modelled: dict[str, Any] = {
        "label": "modelled, not measured",
        "tokens_basis": "estimated from character counts (~4 chars/token), not provider usage",
        "price_table": {"label": prices.label, "usd_per_m_input": prices.usd_per_m_input,
                        "usd_per_m_output": prices.usd_per_m_output},
        "inr_per_usd_assumption": prices.inr_per_usd,
        "est_prompt_tokens_total": pt,
        "est_completion_tokens_total": ct,
        "usd_total": prices.usd(pt, ct),
    }
    if resumes:
        usd = prices.usd(pt, ct) / len(resumes)
        modelled.update(usd_per_resume=usd, inr_per_resume=usd * prices.inr_per_usd,
                        target_inr_per_resume=TARGETS["inr_per_resume"])
    out["cost"] = {
        "measured_usd": 0.0,
        "measured_note": "free-only guard: every model is ':free', so measured cost is $0",
        "modelled_not_measured": modelled,
    }

    v: dict[str, dict[str, Any]] = {"latency_max_lt_20s": lat_verdict}
    inr = modelled.get("inr_per_resume")
    if inr is None:
        v["modelled_cost_lt_1_inr"] = {"status": "no data", "n": 0, "ci95": None}
    else:
        per_call = [prices.usd(r["est_prompt_tokens"], r["est_completion_tokens"]) * prices.inr_per_usd for r in sent]
        cpr = len(sent) / len(resumes)
        if per_call:
            lo, hi = (x * cpr for x in _bootstrap_mean_ci(per_call))
            lo, hi = min(lo, inr), max(hi, inr)
        else:
            lo = hi = 0.0
        v["modelled_cost_lt_1_inr"] = {
            "status": _judge_upper_target(lo, hi, TARGETS["inr_per_resume"], len(resumes), min_n),
            "value": inr, "target": TARGETS["inr_per_resume"], "n": len(resumes), "min_n": min_n,
            "ci95": [lo, hi], "ci95_of": "bootstrap over calls x calls per resume (modelled)",
            "label": "modelled, not measured"}
    sv = out["schema_valid"]
    if sv["value"] is None:
        note = "replay only, not measured" if replayed else "no live model output"
        v["schema_valid_ge_99pct"] = {"status": NOT_MEASURABLE, "note": note, "n": 0, "ci95": None}
    else:
        lo, hi = sv["ci95"]
        v["schema_valid_ge_99pct"] = {
            "status": _judge_lower_target(lo, hi, TARGETS["schema_valid"], sv["n"], min_n),
            "value": sv["value"], "target": TARGETS["schema_valid"], "n": sv["n"], "min_n": min_n,
            "ci95": [lo, hi], "ci95_of": "Wilson interval"}
    v["tool_call_correctness"] = {"status": "n/a", "note": "no agent has tools"}
    out["verdicts"] = v
    return out


def summarize_all(records: Iterable[dict[str, Any]], prices: PriceTable | None = None, *, min_n: int = DEFAULT_MIN_N) -> dict[str, dict[str, Any]]:
    """One summary (and verdict set) per system seen in the traces."""
    recs = list(records)
    names = sorted({r["system"] for r in recs if r.get("system")})
    return {name: summarize(recs, prices, system=name, min_n=min_n) for name in names}
