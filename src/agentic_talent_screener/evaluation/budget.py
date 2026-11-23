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
from collections import deque
import re
import threading
from urllib.parse import urlsplit
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


#: Fixed, key-free text per refusal cause. These (not exception text, which may carry amounts) are what the console,
#: manifest and report show. Nothing here is built from a key, a URL with credentials or provider output.
REFUSAL_MESSAGES: dict[str, str] = {
    "free_only": "free-only run: a paid model needs --allow-paid and a budget above 0",
    "not_allow_listed": "model is not on the paid allow-list (only qwen/qwen3.8-flash)",
    "baseline_not_set": "M1 baseline not set: run phase-open, then commit PHASE_BASELINES_USD",
    "phase_not_opened": "M1 phase not opened for this key in the spend store: run phase-open",
    "usage_below_baseline": "/key usage is below the M1 baseline (replaced key?): the baseline needs a reviewed change",
    "run_cap": "per-run spend cap reached",
    "phase_cap": "M1 phase spend cap reached",
    "key_unreadable": "OpenRouter /key usage could not be read",
    "base_url_not_pinned": "OpenRouter base URL is not exactly https://openrouter.ai/api/v1",
    "schema_check_failed": "spend store schema check failed",
    "ledger_unreadable": "spend ledger could not be read",
    "max_tokens_required": "a paid call needs max_tokens",
    "rate_limited": "every permitted model is rate limited (add --allow-paid to permit the paid fallback)",
    "refused": "refused by the spend guard",
}


class BudgetExceeded(RuntimeError):
    """A spend-guard refusal. ``code`` is one of ``REFUSAL_MESSAGES``; ``str(exc)`` may carry amounts for the log."""

    def __init__(self, message: str = "", code: str = "refused") -> None:
        super().__init__(message)
        self.code = code if code in REFUSAL_MESSAGES else "refused"


def _refuse(code: str, message: str) -> BudgetExceeded:
    return BudgetExceeded(message, code=code)


def refusal_code(exc: BaseException | None) -> str | None:
    """The code of the first BudgetExceeded in an exception chain (systems wrap it in InfrastructureError)."""
    seen = 0
    while exc is not None and seen < 10:
        if isinstance(exc, BudgetExceeded):
            return exc.code
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return None


class PolicyExhausted(BudgetExceeded):
    """Every allowed model in the routing chain is rate limited; the run stops with partial results."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message, code="rate_limited")


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
KEY_HOST = "openrouter.ai"
TOKENS_PER_MESSAGE_OVERHEAD = 16
DEFAULT_RETRIES = 1  # runner caps llm.py retries at 1
FREE_FIRST_MODELS = ("google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free")


def is_free_model(model: str) -> bool:
    return (isinstance(model, str) and _FREE_MODEL.fullmatch(model) is not None
            and not model.startswith("openrouter/"))  # openrouter/auto may route to a paid model


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

#: /key usage (USD) when each phase opened, anchored in REVIEWED CODE so every worktree and branch shares it through
#: git and deleting or reverting the store cannot move it. Empty until the PO commits the value that
#: ``phase-open`` prints (a one-line reviewed change, logged in decisions.log). While a phase is absent every paid
#: call is refused ("baseline not set"). The store's per-key baseline can only LOWER the effective value.
PHASE_BASELINES_USD: dict[str, float] = {"M1": 0.0}  # phase-open 2026-11-24: OpenRouter /key usage 0.000000 USD (decisions.log)


class SpendLedger(Protocol):
    """What the meter needs from the store (``EvalStore`` through ``StoreLedger``)."""

    def record_spend(self, run_id: str, model: str, usd_measured: float, usd_estimated: float, requests: int,
                     phase: str) -> Any: ...

    def phase_total(self, phase: str) -> float: ...

    def phase_baseline(self, phase: str, key_fp: str) -> float | None: ...

    def check_schema(self) -> None: ...


def _amount(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError("amounts must be finite numbers >= 0")
    return float(v)


def key_fingerprint() -> str:
    """Non-reversible fingerprint of the configured OpenRouter key (a salted sha256, 16 hex chars). The key itself is
    never stored or logged. Raises BudgetExceeded when no key is configured."""
    import hashlib

    from agentic_talent_screener.llm import LLMSettings

    key = LLMSettings().openrouter_api_key
    if key is None:
        raise _refuse("key_unreadable", "no OpenRouter key configured; paid call refused")
    return hashlib.sha256(b"eval-phase:" + key.get_secret_value().encode("utf-8")).hexdigest()[:16]


class StoreLedger:
    """The M1 ledger: E15b's ``EvalStore``, the only place spend rows live. Uses only the store's public API;
    ``phase_total`` is the store's float (sum over rows of max(measured, estimated))."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def record_spend(self, run_id: str, model: str, usd_measured: float, usd_estimated: float, requests: int,
                     phase: str) -> None:
        self.store.record_spend(run_id, model, usd_measured, usd_estimated, requests, phase)

    def phase_total(self, phase: str) -> float:
        return _amount(self.store.phase_total(phase))

    def phase_baseline(self, phase: str, key_fp: str) -> float | None:
        return self.store.phase_baseline(phase, key_fp)

    def open_phase(self, phase: str, key_fp: str, baseline_usd: float, cap_usd: float) -> float:
        return self.store.open_phase(phase, key_fp, baseline_usd, cap_usd)

    def check_schema(self) -> None:
        self.store.check_ledger_schema()


# ---------------------------------------------------------------------------------------------- /key usage


def fetch_key_usage(get: Callable[..., Any] | None = None, *, timeout: float = 10.0) -> float:
    """Provider-reported cumulative USD usage for the key (GET https://openrouter.ai/api/v1/key, ``data.usage``).

    Fails closed. The request goes only to https://openrouter.ai, never follows a redirect, and its errors never
    include request details, so the key cannot leak through them or to another host."""
    try:
        from agentic_talent_screener.llm import LLMSettings

        settings = LLMSettings()
        if settings.openrouter_api_key is None:
            raise _refuse("key_unreadable", "OpenRouter key usage unreadable: no API key configured; paid call refused")
        parts = urlsplit(settings.openrouter_base_url)
        if parts.scheme != "https" or parts.hostname != KEY_HOST or parts.port not in (None, 443) or parts.username:
            raise _refuse("base_url_not_pinned", "OpenRouter key usage refused: the base URL is not https://openrouter.ai")
        if get is None:
            import requests

            get = requests.get
        resp = get(f"{settings.openrouter_base_url.rstrip('/')}/key",
                   headers={"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"},
                   timeout=timeout, allow_redirects=False)
        status = getattr(resp, "status_code", 200)
        if isinstance(status, int) and 300 <= status < 400:
            raise _refuse("key_unreadable", "OpenRouter key usage refused: the server redirected")
        resp.raise_for_status()
        usage = resp.json()["data"]["usage"]
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed; the message deliberately omits exc text
        raise _refuse("key_unreadable", f"OpenRouter key usage unreadable ({type(exc).__name__}); paid call refused") from None
    return _reading(usage)


EXPECTED_BASE_URL = "https://openrouter.ai/api/v1"


def check_base_url() -> None:
    """F6: the call that carries the key and spends must go to exactly the official endpoint (llm.py reads the URL
    from the environment, so a poisoned ATS_OPENROUTER_BASE_URL would redirect the key and the spend)."""
    from agentic_talent_screener.llm import LLMSettings

    if LLMSettings().openrouter_base_url != EXPECTED_BASE_URL:
        raise _refuse("base_url_not_pinned", f"the OpenRouter base URL is not exactly {EXPECTED_BASE_URL}; paid call refused")


def _reading(usage: Any) -> float:
    if isinstance(usage, bool) or not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0:
        raise _refuse("key_unreadable", "OpenRouter key usage was not a finite number >= 0; paid call refused")
    return float(usage)


def worst_case_usd(model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any,
                   max_retries: int = DEFAULT_RETRIES) -> float:
    """Upper bound for one logical call: (input token bound x input price + max_tokens x output price) x (retries + 1).

    The input bound is the UTF-8 byte count of the contents and schema (one token per byte at worst, which covers
    byte-level BPE on non-ASCII text) plus a fixed overhead per message."""
    if model not in PAID_PRICES_USD_PER_TOKEN:
        raise _refuse("not_allow_listed", "no price known for this model; paid call refused")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise _refuse("max_tokens_required", "max_tokens is required (a positive integer) for a paid call")
    n_bytes = sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages)
    n_bytes += len(json.dumps(schema, default=str, ensure_ascii=False).encode("utf-8"))
    tokens_in = n_bytes + TOKENS_PER_MESSAGE_OVERHEAD * len(messages)
    p_in, p_out = PAID_PRICES_USD_PER_TOKEN[model]
    return (tokens_in * p_in + max_tokens * p_out) * (max_retries + 1)


#: A well-formed usage block can never settle an attempt below this (USD): a block that reports 0 is not believed.
MIN_BILLED_PER_ATTEMPT_USD = 1e-6
#: Headroom rule (F2). The last HOLDBACK_CALLS settled paid calls stay at their worst case on top of the run's counted
#: spend, so charges the lagging /key counter has not shown yet, or that a usage block under-reports, are covered.
#: Trust boundary: the per-run cap is strict unless OpenRouter both under-reports usage and /key lags by more than about
#: 125 calls (worst seen in testing: about $0.62). The phase cap (M1, $1.50) and a provider-side key limit (the user can set
#: about $1.60 in OpenRouter) are the backstops. Sized on the T67 lag probes (usage block lying low, /key charged $0.0015 per call, pool 150): with
#: 20 calls held a /key lagging by 100 reads still ended at $0.572 billed; with 40 it stays at or below $0.50 for lags of
#: 0, 10 and 100 reads. The hold costs about $0.25 of headroom on the doc example ($0.23 expected + hold < $0.50).
#: The bound is a heuristic: it assumes the real cost per call is well below the worst case, which the price table gives.
HOLDBACK_CALLS = 40


def billed_cost(usages: Sequence[Any], model: str = PAID_MODEL) -> float | None:
    """Total billed USD of one logical call from the provider ``usage`` blocks of its HTTP attempts, or None when any
    attempt has no usable block (the caller then keeps the worst case)."""
    if not usages or model not in PAID_PRICES_USD_PER_TOKEN:
        return None
    p_in, p_out = PAID_PRICES_USD_PER_TOKEN[model]
    total = 0.0
    for u in usages:
        if not isinstance(u, Mapping):
            return None
        figures: list[float] = []
        cost = u.get("cost")
        if cost is not None:
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                return None
            figures.append(float(cost))
        pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
        if pt is not None or ct is not None:
            ok = all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in (pt, ct))
            if not ok:
                return None
            figures.append(pt * p_in + ct * p_out)
        if not figures:
            return None
        total += max(max(figures), MIN_BILLED_PER_ATTEMPT_USD)
    return total


class SpendMeter:
    """Authorises each paid call.

    Per run: counted = max(measured /key delta, reservations). A worst-case reservation is taken before every paid
    attempt; when the call ends it is settled to the billed cost read from the provider's ``usage`` block (see
    ``settle``), and stays at the worst case when that block is missing or malformed. The /key delta can only raise
    the figure, so a usage block that lies low is still bounded by the provider's own counter.

    Per phase (M1), two views, and the larger one binds:
      * ledger view = store phase total + this run's counted spend;
      * key view    = (/key usage now - phase baseline) + this run's reservations not yet visible on /key.
    The key view is the provider's own counter, so it bounds spend even when the ledger is lost, reverted or forked
    (separate worktrees). It also counts any non-eval paid use of the key, which is conservative.
    Residual risk: two processes running at once cannot see each other's in-flight reservations until /key settles,
    so the overshoot is bounded by one worst-case call per concurrent process. Nothing here reads model output."""

    def __init__(self, reader: Callable[[], float], ledger: SpendLedger, *, run_id: str, run_cap_usd: float,
                 phase_cap_usd: float = PHASE_CAP_USD, phase: str = PHASE, max_retries: int = DEFAULT_RETRIES,
                 baselines: Mapping[str, float] | None = None, key_fp: Callable[[], str] | None = None) -> None:
        self.run_cap = valid_budget(run_cap_usd)
        self.phase_cap = valid_budget(phase_cap_usd, ceiling=PHASE_CAP_USD)
        self._read, self.ledger, self.run_id, self.phase, self.max_retries = reader, ledger, run_id, phase, max_retries
        self._baselines = baselines  # default: the module's PHASE_BASELINES_USD, read at use time
        self._key_fp = key_fp or key_fingerprint
        self.usage_start: float | None = None
        self.usage_now: float | None = None
        self.last_delta = 0.0  # measured /key delta for this run
        self.reserved = 0.0  # counted reservations: settled billed costs + worst cases of unsettled calls
        self.worst_total = 0.0  # sum of the worst cases authorised (what each call reserved before settling)
        self.settled = 0.0  # billed cost of the settled calls (provider usage metadata)
        self.settled_calls = 0
        self.unsettled_calls = 0  # usage missing or malformed: the full worst case stays reserved
        self._pending: float | None = None  # worst case of the call in flight (calls are serialised by ``lock``)
        self._holds: deque[float] = deque(maxlen=HOLDBACK_CALLS)  # worst - billed of the last settled calls
        self.held = 0.0  # sum of ``_holds``: the headroom kept on top of the counted spend
        self.paid_calls = 0
        self.views: dict[str, Any] = {}
        self.lock = threading.RLock()  # callers hold it across authorize + the call so checks cannot interleave
        self.start_error: str | None = None
        self.start_code = "key_unreadable"

    @property
    def estimated(self) -> float:  # the reserved total is the ledger's usd_estimated
        return self.reserved

    def start(self) -> None:
        """Read /key and check the ledger's shape BEFORE any paid call. Any failure refuses every paid call."""
        steps: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("base_url_not_pinned", check_base_url), ("schema_check_failed", self.ledger.check_schema))
        for code, step in steps:
            try:
                step()
            except BudgetExceeded as exc:
                self.usage_start, self.start_error, self.start_code = None, str(exc), exc.code
                return
            except Exception:  # noqa: BLE001 - an unusable store: fail closed
                self.usage_start, self.start_error, self.start_code = None, REFUSAL_MESSAGES[code], code
                return
        try:
            self.usage_start = _reading(self._read())
            self.usage_now = self.usage_start
        except BudgetExceeded as exc:
            self.usage_start, self.start_error, self.start_code = None, str(exc), exc.code
        except Exception:  # noqa: BLE001 - a broken reader is an unreadable /key
            self.usage_start, self.start_error, self.start_code = None, REFUSAL_MESSAGES["key_unreadable"], "key_unreadable"

    def spent(self) -> float:
        """Read /key (validated) and return this run's measured delta; settles only upward."""
        if self.usage_start is None:
            raise _refuse(self.start_code, self.start_error or "usage before the run was never read; paid call refused")
        try:
            now = _reading(self._read())
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001
            raise _refuse("key_unreadable", "OpenRouter key usage unreadable; paid call refused") from None
        delta = now - self.usage_start
        if delta < -1e-12:
            raise _refuse("key_unreadable", "OpenRouter usage went backwards; paid call refused")
        self.usage_now = now
        self.last_delta = max(self.last_delta, delta, 0.0)
        return self.last_delta

    def counted(self) -> float:
        """The run's counted spend: the larger of the provider's /key delta and the reservations, plus the headroom
        held back for the last settled calls (they may not be visible on /key yet, or their usage may be low)."""
        return max(self.spent(), self.reserved) + self.held

    def baseline(self) -> float:
        """Effective phase baseline: the reviewed constant, lowered (never raised) by this key's store row."""
        table = PHASE_BASELINES_USD if self._baselines is None else self._baselines
        code = table.get(self.phase)
        if code is None:
            raise _refuse("baseline_not_set", f"{self.phase} baseline not set (run phase-open, then commit PHASE_BASELINES_USD); "
                                 "paid call refused")
        try:
            code = _amount(code)
            stored = self.ledger.phase_baseline(self.phase, self._key_fp())
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001
            raise _refuse("ledger_unreadable", "phase baseline unreadable; paid call refused") from None
        if stored is None:
            raise _refuse("phase_not_opened", f"phase {self.phase} is not opened for this key in the spend store "
                                 "(run phase-open); paid call refused")
        return min(code, _amount(stored))

    def phase_views(self, counted: float) -> dict[str, float]:
        base = self.baseline()
        usage_now = self.usage_now if self.usage_now is not None else 0.0
        if usage_now < base - 1e-12:
            raise _refuse("usage_below_baseline", f"/key usage {usage_now:.6f} is below the {self.phase} baseline {base:.6f} "
                                 "(replaced key?): the baseline needs a reviewed change; paid call refused")
        try:
            prior = _amount(self.ledger.phase_total(self.phase))
        except Exception:  # noqa: BLE001 - fail closed
            raise _refuse("ledger_unreadable", "spend ledger unreadable; paid call refused") from None
        ledger_view = prior + counted
        key_view = (usage_now - base) + max(0.0, self.reserved - self.last_delta) + self.held
        self.views = {"baseline_usd": base, "ledger_view_usd": ledger_view, "key_view_usd": key_view,
                      "binding": "key" if key_view > ledger_view else "ledger"}
        return {"ledger": ledger_view, "key": key_view, "used": max(ledger_view, key_view)}

    def authorize(self, model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any) -> float:
        worst = worst_case_usd(model, messages, schema, max_tokens, self.max_retries)
        counted = self.counted()  # one /key read; also sets usage_now
        if counted + worst > self.run_cap:
            raise _refuse("run_cap", f"per-run cap {self.run_cap:.2f} USD would be exceeded "
                                 f"(counted {counted:.6f} + worst case {worst:.6f}); paid call refused")
        v = self.phase_views(counted)
        if v["used"] + worst > self.phase_cap:
            raise _refuse("phase_cap", f"{self.phase} phase cap {self.phase_cap:.2f} USD would be exceeded "
                                 f"(ledger view {v['ledger']:.6f}, key view {v['key']:.6f} + worst case {worst:.6f}); "
                                 "paid call refused")
        self.reserved += worst  # reserved before the attempt; settle() may lower it to the billed cost
        self.worst_total += worst
        self._pending = worst
        self.paid_calls += 1
        return worst

    def settle(self, usages: Sequence[Any], model: str = PAID_MODEL) -> None:
        """Replace the reservation of the call that just ended by its billed cost.

        ``usages`` is the ``usage`` block of every HTTP response of that call (retries included), taken from the
        provider's response metadata, never from the model's message. The cost is ``usage.cost`` or the token counts
        times the allow-listed price (the larger of the two when both are present). If any attempt has no usable
        usage, the call keeps its full worst case (fail closed). A usage block that lies low is still bounded by the
        /key delta (``counted``) and the phase key view."""
        with self.lock:
            worst, self._pending = self._pending, None
            if worst is None:
                return
            cost = billed_cost(usages, model)
            if cost is None:
                self.unsettled_calls += 1
                return
            self.reserved = max(0.0, self.reserved - worst + cost)
            self.settled += cost
            self.settled_calls += 1
            self._holds.append(max(worst - cost, 0.0))  # the oldest hold drops out after HOLDBACK_CALLS newer calls
            self.held = sum(self._holds)

    def finish(self, model: str = PAID_MODEL) -> dict[str, Any]:
        """Final reading, then one ledger row holding BOTH figures. If the final read fails, the measured figure is
        raised to the reservations so the ledger never undercounts."""
        measured_ok = self.usage_start is not None
        if measured_ok:
            try:
                self.spent()
            except BudgetExceeded:
                measured_ok = False
        measured = self.last_delta if measured_ok else max(self.last_delta, self.reserved)
        self.ledger.record_spend(self.run_id, model, measured, self.reserved, self.paid_calls, self.phase)
        views: dict[str, Any] = {}
        total: float | None
        try:
            total = _amount(self.ledger.phase_total(self.phase))
            if measured_ok:  # the new row is in the store total now, so only the key view needs this run's gap
                base = self.baseline()
                key_view = (max(0.0, (self.usage_now or 0.0) - base) + max(0.0, self.reserved - self.last_delta)
                            + self.held)
                views = {"baseline_usd": base, "ledger_view_usd": total, "key_view_usd": key_view,
                         "binding": "key" if key_view > total else "ledger"}
        except Exception:  # noqa: BLE001
            total = None
        phase_used = max(total, views.get("key_view_usd", 0.0)) if total is not None else None
        return {"measured_usd": measured, "reserved_usd": self.reserved, "estimated_usd": self.reserved,
                "reserved_worst_usd": self.worst_total, "headroom_held_usd": self.held, "settled_usd": self.settled,
                "settled_calls": self.settled_calls, "unsettled_calls": self.unsettled_calls,
                "counted_usd": max(measured, self.reserved), "paid_calls": self.paid_calls,
                "phase": self.phase, "phase_total_usd": total, "phase_cap_usd": self.phase_cap,
                "phase_remaining_usd": None if phase_used is None else max(self.phase_cap - phase_used, 0.0),
                "run_cap_usd": self.run_cap, "measured_ok": measured_ok, **views}


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
        code = "free_only" if is_allowed_paid_model(model) else "not_allow_listed"
        raise _refuse(code, f"model {model!r} refused: {PAID_NEEDS_APPROVAL}")

    def authorize_paid(self, model: str, messages: Sequence[Mapping[str, Any]], schema: Any, max_tokens: Any) -> None:
        """Called on a cache miss, before the request. Free ids need nothing more."""
        self.check(model)
        if is_free_model(model):
            return
        if self.meter is None:
            raise _refuse("free_only", "no spend meter; paid call refused")
        self.meter.authorize(model, messages, schema, max_tokens)


# ---------------------------------------------------------------------------------------------- routing


MODEL_POLICIES = ("free-first", "paid-direct")


class ModelRouter:
    """free-first: first free model; after ``switch_after`` consecutive 429s the second free model; after that,
    only if a paid model was given (needs --allow-paid), the paid model; else the run stops (PolicyExhausted).
    Every switch is kept in ``switches``; calls per system and model in ``mix``."""

    def __init__(self, free_models: Sequence[str], paid_model: str | None = None, *, switch_after: int = 3,
                 system_getter: Callable[[], str | None] | None = None, policy: str = "free-first") -> None:
        if isinstance(switch_after, bool) or not isinstance(switch_after, int) or switch_after < 1:
            raise ValueError("switch_after must be an integer >= 1")
        if policy not in MODEL_POLICIES:
            raise ValueError(f"model policy must be one of {', '.join(MODEL_POLICIES)}")
        if paid_model is not None and not is_allowed_paid_model(paid_model):
            raise ValueError("the paid fallback must be on the allow-list")
        if policy == "paid-direct":
            if paid_model is None:
                raise ValueError("paid-direct routing needs the allow-listed paid model (--allow-paid)")
            free_models = []  # the free models are skipped on purpose
        elif not free_models or not all(is_free_model(m) for m in free_models):
            raise ValueError("free-first routing starts from a non-empty list of ':free' models only")
        self.policy = policy
        self.ignored_default_model: str | None = None
        self.chain = list(free_models) + ([paid_model] if paid_model else [])
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
            return {"policy": self.policy, "chain": list(self.chain), "switch_after_429": self.switch_after,
                    "switches": list(self.switches),
                    "calls_by_system": {s: dict(p) for s, p in sorted(self.mix.items())},
                    "systems_with_mixed_models": mixed, "stopped": self.stopped_reason,
                    "ignored_default_model": self.ignored_default_model}


def is_rate_limited(exc: BaseException) -> bool:
    """llm.py raises ``RuntimeError('OpenRouter request failed (429); ...')`` once its retries are spent."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc)
    if "request failed (429)" in text:
        return True
    # OpenRouter reports an upstream error of a free model as HTTP 200 with an ``error`` body and no ``choices``;
    # llm.py words that as "did not contain chat content". Only an error code of 429 is a rate limit.
    return "did not contain chat content" in text and _body_error_code(exc) == 429


def _body_error_code(exc: BaseException) -> int | None:
    """The numeric ``error.code`` of the provider body that ``llm.chat_json`` was parsing when it raised.

    STOPGAP: llm.py keeps the body only in a local of the ``chat_json`` frame, so this reads it from the exception's
    traceback. A rename of that function or variable breaks tests/evaluation/test_t66_live_harness.py rather than
    silently widening the match. The proper fix is a contract change in llm.py (numeric ``error_code`` in the message
    or a ``status_code`` attribute, which ``is_rate_limited`` already honours). Anything unreadable gives None, which
    counts as not rate limited, so the run fails closed and no spend opens."""
    tb, seen = exc.__traceback__, 0
    while tb is not None and seen < 50:
        frame = tb.tb_frame
        if frame.f_code.co_name == "chat_json" and frame.f_globals.get("__name__") == "agentic_talent_screener.llm":
            body = frame.f_locals.get("body")
            err = body.get("error") if isinstance(body, dict) else None
            code = err.get("code") if isinstance(err, dict) else None
            if isinstance(code, str) and code.strip().isdigit():
                code = int(code.strip())
            return code if isinstance(code, int) and not isinstance(code, bool) else None
        tb, seen = tb.tb_next, seen + 1
    return None
