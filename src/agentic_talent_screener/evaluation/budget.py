"""Spend guard (E1 free-only, E15a paid opt-in).

Free ids (``vendor/model:free``) always pass. A paid id passes only when ALL of these hold: the run has
``allow_paid=True``, the per-run cap is above 0, and the id is EXACTLY in ``PAID_ALLOWLIST``.

Spend is never taken from an estimate or from model output (T6 review: estimates were bypassable, model-controlled
cost, NaN, billed-then-failed calls). It is measured from OpenRouter's own counter, ``GET /key`` -> ``data.usage``,
read before the run and before every paid call. Failed or billed calls are covered because the counter includes them.
A paid call is refused unless usage delta + worst case for the next call fits BOTH the per-run cap and the phase cap.
Anything unreadable (key missing, HTTP error, NaN, negative) fails closed. The API key is never logged or returned.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class BudgetExceeded(RuntimeError):
    pass


class PolicyExhausted(BudgetExceeded):
    """Every allowed model in the routing chain is rate limited; the run stops with partial results."""


PAID_NEEDS_APPROVAL = ("paid models need --allow-paid (user spend approval A22) and a budget above 0; "
                       "only ':free' models are allowed otherwise")

# Exactly vendor/model plus the single ':free' suffix; ASCII only, no variant tags, no whitespace or control chars.
_FREE_MODEL = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*:free", re.ASCII)

#: the only paid ids that can ever pass (exact match), with USD per token (input, output)
PAID_ALLOWLIST: frozenset[str] = frozenset({"qwen/qwen3.8-flash"})
PAID_PRICES_USD_PER_TOKEN: Mapping[str, tuple[float, float]] = {"qwen/qwen3.8-flash": (0.15 / 1e6, 0.47 / 1e6)}
PAID_MODEL = "qwen/qwen3.8-flash"
MAX_RUN_BUDGET_USD = 0.50  # default config ceiling for --budget-usd; raising it is an explicit code/config change
PHASE_CAP_USD = 1.50  # M1 total across runs
PHASE = "M1"
CHARS_PER_TOKEN_BOUND = 3  # prompt tokens <= chars / 3 (deliberately pessimistic)
DEFAULT_RETRIES = 1  # runner caps llm.py retries at 1
FREE_FIRST_MODELS = ("google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free")


def is_free_model(model: str) -> bool:
    return isinstance(model, str) and _FREE_MODEL.fullmatch(model) is not None


def is_allowed_paid_model(model: str) -> bool:
    return isinstance(model, str) and model in PAID_ALLOWLIST  # exact full match


def valid_budget(value: Any, ceiling: float = MAX_RUN_BUDGET_USD) -> float:
    """A finite real number in [0, ceiling]; NaN, inf, negatives, bools and non-numbers are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("budget_usd must be a number")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError("budget_usd must be a finite number >= 0")
    if v > ceiling:
        raise ValueError(f"budget_usd must be at most {ceiling} USD per run (higher needs an explicit config change)")
    return v


# ---------------------------------------------------------------------------------------------- ledger


class SpendLedger(Protocol):
    """The slice of E15b's store (``record_spend`` / ``phase_total``) that this module needs."""

    def record_spend(self, run_id: str, model: str, usd_measured: float, usd_estimated: float, requests: int,
                     phase: str) -> Any: ...

    def phase_total(self, phase: str) -> float: ...


def _amount(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError("spend amounts must be finite numbers >= 0")
    return float(v)


class JsonLedger:
    """Fallback ledger: ``<runs_dir>/spend_ledger.json`` until ``store.py`` (E15b) merges."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        rows = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError("spend ledger is malformed")
        return rows

    def record_spend(self, run_id: str, model: str, usd_measured: float, usd_estimated: float, requests: int,
                     phase: str) -> None:
        row = {"run_id": run_id, "model": model, "usd_measured": _amount(usd_measured),
               "usd_estimated": _amount(usd_estimated), "requests": int(requests), "phase": phase}
        with self._lock:
            rows = self._load()
            rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"entries": rows}, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)

    def phase_total(self, phase: str) -> float:
        return sum(_amount(r.get("usd_measured")) for r in self._load() if r.get("phase") == phase)


# ---------------------------------------------------------------------------------------------- /key usage


def fetch_key_usage(get: Callable[..., Any] | None = None, *, timeout: float = 15.0) -> float:
    """Provider-reported cumulative USD usage for the key (GET <base>/key, ``data.usage``). Fails closed.

    Error text never includes request details, so the key cannot leak through it."""
    try:
        from agentic_talent_screener.llm import LLMSettings

        settings = LLMSettings()
        if settings.openrouter_api_key is None:
            raise BudgetExceeded("OpenRouter key usage unreadable: no API key configured; paid call refused")
        if get is None:
            import requests

            get = requests.get
        resp = get(f"{settings.openrouter_base_url}/key",
                   headers={"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"},
                   timeout=timeout)
        resp.raise_for_status()
        usage = resp.json()["data"]["usage"]
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed; the message deliberately omits exc text
        raise BudgetExceeded(f"OpenRouter key usage unreadable ({type(exc).__name__}); paid call refused") from None
    if isinstance(usage, bool) or not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0:
        raise BudgetExceeded("OpenRouter key usage was not a finite number >= 0; paid call refused")
    return float(usage)


def worst_case_usd(model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any,
                   max_retries: int = DEFAULT_RETRIES) -> float:
    """Upper bound for one logical call: ((chars/3) x input price + max_tokens x output price) x (retries + 1)."""
    if model not in PAID_PRICES_USD_PER_TOKEN:
        raise BudgetExceeded("no price known for this model; paid call refused")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise BudgetExceeded("max_tokens is required (a positive integer) for a paid call")
    chars = sum(len(str(m.get("content", ""))) for m in messages) + len(json.dumps(schema, default=str))
    p_in, p_out = PAID_PRICES_USD_PER_TOKEN[model]
    return (math.ceil(chars / CHARS_PER_TOKEN_BOUND) * p_in + max_tokens * p_out) * (max_retries + 1)


class SpendMeter:
    """Authorises each paid call against the provider's own usage counter and the phase ledger."""

    def __init__(self, reader: Callable[[], float], ledger: SpendLedger, *, run_id: str, run_cap_usd: float,
                 phase_cap_usd: float = PHASE_CAP_USD, phase: str = PHASE, max_retries: int = DEFAULT_RETRIES) -> None:
        self.run_cap = valid_budget(run_cap_usd)
        self.phase_cap = valid_budget(phase_cap_usd, ceiling=PHASE_CAP_USD)
        self._read, self.ledger, self.run_id, self.phase, self.max_retries = reader, ledger, run_id, phase, max_retries
        self.usage_start: float | None = None
        self.last_delta = 0.0
        self.estimated = 0.0  # sum of worst cases authorised (an upper bound, kept next to the measured figure)
        self.paid_calls = 0
        self.lock = threading.RLock()  # callers hold it across authorize + the call so checks cannot interleave
        self.start_error: str | None = None

    def start(self) -> None:
        try:
            self.usage_start = self._read()
        except BudgetExceeded as exc:
            self.usage_start, self.start_error = None, str(exc)

    def spent(self) -> float:
        if self.usage_start is None:
            raise BudgetExceeded(self.start_error or "usage before the run was never read; paid call refused")
        delta = self._read() - self.usage_start
        if not math.isfinite(delta) or delta < -1e-12:
            raise BudgetExceeded("OpenRouter usage went backwards or is invalid; paid call refused")
        self.last_delta = max(delta, 0.0)
        return self.last_delta

    def authorize(self, model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any) -> float:
        worst = worst_case_usd(model, messages, schema, max_tokens, self.max_retries)
        spent = self.spent()
        if spent + worst > self.run_cap:
            raise BudgetExceeded(f"per-run cap {self.run_cap:.2f} USD would be exceeded "
                                 f"(measured {spent:.6f} + worst case {worst:.6f}); paid call refused")
        try:
            prior = _amount(self.ledger.phase_total(self.phase))
        except Exception:  # noqa: BLE001 - fail closed
            raise BudgetExceeded("spend ledger unreadable; paid call refused") from None
        if prior + spent + worst > self.phase_cap:
            raise BudgetExceeded(f"{self.phase} phase cap {self.phase_cap:.2f} USD would be exceeded "
                                 f"(ledger {prior:.6f} + run {spent:.6f} + worst case {worst:.6f}); paid call refused")
        self.estimated += worst
        self.paid_calls += 1
        return worst

    def finish(self, model: str = PAID_MODEL) -> dict[str, Any]:
        """Final reading, then one ledger row. If the final read fails the last known delta is kept and flagged."""
        measured_ok = self.usage_start is not None
        if measured_ok:
            try:
                self.spent()
            except BudgetExceeded:
                measured_ok = False
        self.ledger.record_spend(self.run_id, model, self.last_delta, self.estimated, self.paid_calls, self.phase)
        try:
            total: float | None = _amount(self.ledger.phase_total(self.phase))
        except Exception:  # noqa: BLE001
            total = None
        return {"measured_usd": self.last_delta, "estimated_usd": self.estimated, "paid_calls": self.paid_calls,
                "phase": self.phase, "phase_total_usd": total, "phase_cap_usd": self.phase_cap,
                "phase_remaining_usd": None if total is None else max(self.phase_cap - total, 0.0),
                "run_cap_usd": self.run_cap, "measured_ok": measured_ok}


class BudgetGuard:
    """Free ids always pass. A paid id needs allow_paid, a budget above 0 and an exact allow-list match."""

    def __init__(self, budget_usd: float = 0.0, *, allow_paid: bool = False, meter: SpendMeter | None = None,
                 router: Any = None) -> None:
        v = valid_budget(budget_usd)
        if v != 0 and not allow_paid:
            raise ValueError(f"budget_usd must be 0 without --allow-paid: {PAID_NEEDS_APPROVAL}")
        if allow_paid and v > 0 and meter is None:
            raise ValueError("a paid budget needs a spend meter (the /key counter)")
        self.budget_usd, self.allow_paid, self.meter, self.router = v, bool(allow_paid), meter, router

    def check(self, model: str) -> None:
        """Raise before any network call for a model that is not permitted."""
        if is_free_model(model):
            return
        if is_allowed_paid_model(model) and self.allow_paid and self.budget_usd > 0:
            return
        raise BudgetExceeded(f"model {model!r} refused: {PAID_NEEDS_APPROVAL}")

    def authorize_paid(self, model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any) -> None:
        """Called on a cache miss, before the request. Free ids need nothing more."""
        self.check(model)
        if is_free_model(model):
            return
        if self.meter is None:
            raise BudgetExceeded("no spend meter; paid call refused")
        self.meter.authorize(model, messages, schema, max_tokens)


# ---------------------------------------------------------------------------------------------- routing


class ModelRouter:
    """free-first: first free model; after ``switch_after`` consecutive 429s the second free model; after that,
    only if a paid model was given (needs --allow-paid), the paid model; else the run stops (PolicyExhausted).
    Every switch is kept in ``switches``; calls per system and model in ``mix``."""

    def __init__(self, free_models: Sequence[str], paid_model: str | None = None, *, switch_after: int = 3,
                 system_getter: Callable[[], str | None] | None = None) -> None:
        if isinstance(switch_after, bool) or not isinstance(switch_after, int) or switch_after < 1:
            raise ValueError("switch_after must be an integer >= 1")
        self.chain = list(free_models) + ([paid_model] if paid_model else [])
        if not self.chain:
            raise ValueError("at least one model is needed")
        self.n_free = len(free_models)
        self.switch_after, self.system_getter = switch_after, system_getter
        self.idx = 0
        self.consecutive_429 = 0
        self.switches: list[dict[str, Any]] = []
        self.mix: dict[str, dict[str, int]] = {}
        self.stopped_reason: str | None = None
        self._lock = threading.Lock()

    def current(self) -> str:
        with self._lock:
            if self.idx >= len(self.chain):
                raise PolicyExhausted(self.stopped_reason or "all permitted models are rate limited")
            return self.chain[self.idx]

    def note_use(self, model: str) -> None:
        with self._lock:
            sysname = (self.system_getter() if self.system_getter else None) or "unattributed"
            per = self.mix.setdefault(sysname, {})
            per[model] = per.get(model, 0) + 1

    def on_success(self) -> None:
        with self._lock:
            self.consecutive_429 = 0

    def on_429(self, model: str) -> None:
        with self._lock:
            if self.idx >= len(self.chain) or self.chain[self.idx] != model:
                return  # a stale report about a model already left behind
            self.consecutive_429 += 1
            if self.consecutive_429 < self.switch_after:
                return
            n, self.consecutive_429 = self.consecutive_429, 0
            nxt = self.idx + 1
            if nxt >= len(self.chain):
                self.stopped_reason = (f"{model} rate limited {n} times in a row and no fallback is left "
                                       f"(pass --allow-paid to permit the paid model); stopping with partial results")
                self.switches.append({"from": model, "to": None, "after_429s": n, "reason": self.stopped_reason})
            else:
                reason = "rate_limited" if nxt < self.n_free else "all free models rate limited; paid model (--allow-paid)"
                self.switches.append({"from": model, "to": self.chain[nxt], "after_429s": n, "reason": reason})
            self.idx = nxt

    def summary(self) -> dict[str, Any]:
        with self._lock:
            mixed = sorted(s for s, per in self.mix.items() if len(per) > 1)
            return {"policy": "free-first", "chain": list(self.chain), "switch_after_429": self.switch_after,
                    "switches": list(self.switches),
                    "calls_by_system": {s: dict(p) for s, p in sorted(self.mix.items())},
                    "systems_with_mixed_models": mixed, "stopped": self.stopped_reason}


def is_rate_limited(exc: BaseException) -> bool:
    """llm.py raises ``RuntimeError('OpenRouter request failed (429); ...')`` once its retries are spent."""
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(exc, RuntimeError) and "request failed (429)" in str(exc))
