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
INFRA = ("budget_guard", "cache_miss")


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


def summarize(records: Iterable[dict[str, Any]], prices: PriceTable | None = None,
              *, system: str | None = None) -> dict[str, Any]:
    prices = prices or PriceTable()
    recs = [r for r in records if system is None or r.get("system") == system]
    calls = [r for r in recs if r["type"] == "call"]
    resumes = [r for r in recs if r["type"] == "resume"]
    out: dict[str, Any] = {"system": system, "n_resumes": len(resumes), "n_calls": len(calls)}

    walls_s = [r["wall_ms"] / 1000 for r in resumes if r.get("status") != "error"]
    if walls_s:
        out["latency_s_per_resume"] = {
            "p50": percentile(walls_s, 50), "p95": percentile(walls_s, 95), "max": max(walls_s),
            "timing_basis": sorted({r["timing_basis"] for r in resumes}),
        }
    live = [r for r in calls if r["cache"] == "miss" and r["error_kind"] not in INFRA]
    out["latency_basis"] = "live calls present" if live else "cache replay only; latencies are not representative"
    stage: dict[str, list[float]] = defaultdict(list)
    for r in calls:
        stage[r["call_site"]].append(r["latency_ms"] / 1000)
    out["call_latency_s_by_call_site"] = {
        k: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v)}
        for k, v in sorted(stage.items())
    }

    model_calls = [r for r in calls if r["error_kind"] not in INFRA]
    out["infrastructure_failures"] = len(calls) - len(model_calls)
    n = len(model_calls)
    out["schema_valid"] = _prop(sum(1 for r in model_calls if r["schema_valid"]), n)
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in model_calls:
        by_site[r["call_site"]].append(r)
    out["schema_valid_by_call_site"] = {
        k: _prop(sum(1 for r in v if r["schema_valid"]), len(v)) for k, v in sorted(by_site.items())
    }
    out["cache_hit_rate"] = _prop(sum(1 for r in calls if r["cache"] == "hit"), len(calls))
    out["retry_rate_boundary"] = _prop(sum(1 for r in model_calls if r["attempts"] > 1), n)
    out["failure_kinds"] = dict(Counter(r["error_kind"] for r in calls if r["error_kind"]))
    out["terminal_failure_rate"] = _prop(sum(1 for r in model_calls if r["error_kind"]), n)
    out["tool_call_correctness"] = "N/A: no agent has tools"
    out["calls_per_resume"] = (len(calls) / len(resumes)) if resumes else None

    pt = sum(r["est_prompt_tokens"] for r in calls)
    ct = sum(r["est_completion_tokens"] for r in calls)
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
    out["verdicts"] = verdicts(out)
    return out


def verdicts(s: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pass / fail vs slide 9. 'no data' when a metric could not be computed or is not meaningful."""
    v: dict[str, dict[str, Any]] = {}
    lat = s.get("latency_s_per_resume")
    if lat is None or s["latency_basis"].startswith("cache replay"):
        v["latency_max_lt_20s"] = {"status": "no data", "note": "needs live calls"}
    else:
        v["latency_max_lt_20s"] = {"status": "pass" if lat["max"] < TARGETS["latency_s_max"] else "fail",
                                   "value": lat["max"], "target": TARGETS["latency_s_max"]}
    inr = s["cost"]["modelled_not_measured"].get("inr_per_resume")
    if inr is None:
        v["modelled_cost_lt_1_inr"] = {"status": "no data"}
    else:
        v["modelled_cost_lt_1_inr"] = {"status": "pass" if inr < TARGETS["inr_per_resume"] else "fail", "value": inr,
                                       "target": TARGETS["inr_per_resume"], "label": "modelled, not measured"}
    sv = s["schema_valid"]
    if sv["value"] is None:
        v["schema_valid_ge_99pct"] = {"status": "no data"}
    else:
        v["schema_valid_ge_99pct"] = {"status": "pass" if sv["value"] >= TARGETS["schema_valid"] else "fail",
                                      "value": sv["value"], "target": TARGETS["schema_valid"]}
    v["tool_call_correctness"] = {"status": "n/a", "note": "no agent has tools"}
    return v
