"""E15a: paid-model guard. Everything offline; the OpenRouter /key endpoint is a mock."""

from __future__ import annotations

import json
import math

import pytest

from agentic_talent_screener.evaluation.budget import (
    BudgetExceeded, BudgetGuard, JsonLedger, ModelRouter, PolicyExhausted, SpendMeter, fetch_key_usage, valid_budget,
    worst_case_usd,
)
from agentic_talent_screener.evaluation.cache import CachedChat, LLMCache
from agentic_talent_screener.evaluation.cli import main
from agentic_talent_screener.evaluation.config import EvalConfig

PAID = "qwen/qwen3.8-flash"
F1, F2 = "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"
MSGS = [{"role": "user", "content": "hello"}]
SCHEMA = {"type": "object"}
SECRET = "sk-or-SECRET-KEY-VALUE"


class Usage:
    """Mock of GET /key: a counter the test moves."""

    def __init__(self, v=0.0):
        self.v, self.fail = v, False

    def __call__(self):
        if self.fail:
            raise BudgetExceeded("OpenRouter key usage unreadable (X); paid call refused")
        return self.v


class Ledger:
    def __init__(self, total=0.0):
        self.total, self.rows = total, []

    def record_spend(self, *a):
        self.rows.append(a)
        self.total += a[2]

    def phase_total(self, phase):
        return self.total


def meter(usage, ledger=None, cap=0.50):
    m = SpendMeter(usage, ledger or Ledger(), run_id="r1", run_cap_usd=cap)
    m.start()
    return m


def chat(tmp_path, call, *, allow_paid=True, budget=0.5, usage=None, ledger=None, free=(F1, F2), switch=3):
    m = meter(usage or Usage(), ledger) if allow_paid else None
    router = ModelRouter(list(free), PAID if allow_paid else None, switch_after=switch)
    g = BudgetGuard(budget if allow_paid else 0.0, allow_paid=allow_paid, meter=m, router=router)
    return CachedChat(LLMCache(tmp_path), g, call=call), router, m


def r429(*a, **k):
    raise RuntimeError("OpenRouter request failed (429); rate limited")


def test_paid_refused_without_allow_paid():
    g = BudgetGuard()
    with pytest.raises(BudgetExceeded):
        g.check(PAID)
    with pytest.raises(ValueError):
        BudgetGuard(0.1)  # a budget without --allow-paid


@pytest.mark.parametrize("model", ["openai/gpt-4o", "qwen/qwen3.8-flash:nitro", "qwen/qwen3.8-flash ",
                                   "QWEN/qwen3.8-flash", "qwen/qwen3.8", "x/" + PAID, ""])
def test_non_allowlisted_paid_id_refused(model):
    g = BudgetGuard(0.5, allow_paid=True, meter=meter(Usage()))
    with pytest.raises(BudgetExceeded):
        g.check(model)
    g.check(PAID)


def test_allow_paid_with_zero_cap_refuses():
    g = BudgetGuard(0.0, allow_paid=True)
    with pytest.raises(BudgetExceeded):
        g.check(PAID)


def test_per_run_cap_blocks_on_usage_delta_plus_worst_case():
    u = Usage(10.0)
    m = meter(u)
    w = worst_case_usd(PAID, MSGS, SCHEMA, 1000)
    assert 0 < w < 0.01
    m.authorize(PAID, MSGS, SCHEMA, 1000)
    u.v = 10.0 + 0.50 - w / 2  # delta + worst case now exceeds 0.50
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_phase_cap_blocks_at_1_50():
    led = Ledger(1.4995)
    m = meter(Usage(), led)
    with pytest.raises(BudgetExceeded, match="phase cap"):
        m.authorize(PAID, MSGS, SCHEMA, 1000)
    led.total = 1.0
    m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_phase_cap_counts_run_spend():
    u, led = Usage(), Ledger(1.30)
    m = meter(u, led)
    u.v = 0.21  # run delta 0.21 + ledger 1.30 > 1.50 though the run is under 0.50
    with pytest.raises(BudgetExceeded, match="phase cap"):
        m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_unreadable_key_refuses_at_start_and_later():
    u = Usage()
    u.fail = True
    m = meter(u)  # start() could not read
    u.fail = False
    with pytest.raises(BudgetExceeded):
        m.authorize(PAID, MSGS, SCHEMA, 100)
    u2 = Usage()
    m2 = meter(u2)
    u2.fail = True
    with pytest.raises(BudgetExceeded):
        m2.authorize(PAID, MSGS, SCHEMA, 100)


def test_max_tokens_required():
    m = meter(Usage())
    for bad in (None, 0, -1, True, 1.5):
        with pytest.raises(BudgetExceeded):
            m.authorize(PAID, MSGS, SCHEMA, bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1, 0.51, True, "0.1", None])
def test_bad_budgets_refused(bad):
    with pytest.raises(ValueError):
        valid_budget(bad)
    with pytest.raises(ValueError):
        BudgetGuard(bad, allow_paid=True)


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "0.51"])
def test_cli_budget_refused(bad):
    with pytest.raises(SystemExit):
        main(["run", "--allow-paid", "--budget-usd", bad])


def test_cli_budget_needs_allow_paid(capsys):
    assert main(["run", "--budget-usd", "0.1"]) == 2
    assert "allow-paid" in capsys.readouterr().err


def test_config_ceiling():
    assert EvalConfig(budget_usd=0.5).budget_usd == 0.5
    for bad in (0.51, math.nan, -1):
        with pytest.raises(ValueError):
            EvalConfig(budget_usd=bad)


def test_billed_then_failed_calls_count(tmp_path):
    """A paid call that bills and then raises still moves the provider counter, so the next call is refused."""
    u = Usage()
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        u.v += 0.30  # billed ...
        raise RuntimeError("OpenRouter request failed (500); provider error")  # ... then failed

    c, router, m = chat(tmp_path, call, usage=u, free=())
    for _ in range(2):
        with pytest.raises(RuntimeError):
            c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=1000)
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=1000)
    assert seen == [PAID, PAID]
    assert m.last_delta >= 0.5


def test_paid_call_needs_max_tokens_in_chat(tmp_path):
    c, *_ = chat(tmp_path, lambda *a, **k: {}, free=())
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        c(MSGS, schema=SCHEMA, schema_name="s")


def test_free_first_fallback_order(tmp_path):
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        if model != PAID:
            r429()
        return {"ok": model}

    c, router, m = chat(tmp_path, call)
    out = c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)
    assert out == {"ok": PAID}
    assert seen == [F1] * 3 + [F2] * 3 + [PAID]
    assert [(s["from"], s["to"]) for s in router.switches] == [(F1, F2), (F2, PAID)]
    assert router.summary()["calls_by_system"]["unattributed"] == {PAID: 1}


def test_free_only_stops_when_both_rate_limited(tmp_path):
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        r429()

    c, router, _ = chat(tmp_path, call, allow_paid=False)
    with pytest.raises(PolicyExhausted):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)
    assert seen == [F1] * 3 + [F2] * 3
    assert router.summary()["stopped"]


def test_success_resets_429_streak(tmp_path):
    n = {"i": 0}

    def call(*a, model, **k):
        n["i"] += 1
        if n["i"] in (1, 2, 4, 5):
            r429()
        return {"model": model}

    c, router, _ = chat(tmp_path, call, allow_paid=False)
    assert c(MSGS + [{"role": "user", "content": "a"}], schema=SCHEMA, schema_name="s") == {"model": F1}
    assert c(MSGS + [{"role": "user", "content": "b"}], schema=SCHEMA, schema_name="s") == {"model": F1}
    assert router.switches == []


def test_paid_blocked_after_fallback_stops_run(tmp_path):
    u = Usage(0.0)

    def call(*a, model, **k):
        if model != PAID:
            r429()
        return {}

    c, router, m = chat(tmp_path, call, usage=u)
    u.v = 0.499999
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)


def test_key_fetch_parses_and_never_leaks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)

    class Resp:
        def __init__(self, body, ok=True):
            self.body, self.ok = body, ok

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"401 for Bearer {SECRET}")

        def json(self):
            return self.body

    calls = []

    def get(url, headers, timeout):
        calls.append((url, headers))
        return Resp({"data": {"usage": 1.25}})

    assert fetch_key_usage(get) == 1.25
    assert calls[0][0].endswith("/key")
    for body in ({"data": {"usage": "x"}}, {"data": {"usage": float("nan")}}, {"data": {"usage": -1}},
                 {"data": {"usage": True}}, {"data": {}}, {}):
        with pytest.raises(BudgetExceeded) as e:
            fetch_key_usage(lambda *a, **k: Resp(body))
        assert SECRET not in str(e.value)
    with pytest.raises(BudgetExceeded) as e:
        fetch_key_usage(lambda *a, **k: Resp({}, ok=False))
    assert SECRET not in str(e.value) and "RuntimeError" in str(e.value)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("ATS_OPENROUTER_API_KEY", "")
    with pytest.raises(BudgetExceeded):
        fetch_key_usage(get)


def test_key_never_in_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    u = Usage()
    led = JsonLedger(tmp_path / "spend_ledger.json")
    m = SpendMeter(u, led, run_id="r1", run_cap_usd=0.5)
    m.start()
    u.v = 0.02
    m.authorize(PAID, MSGS, SCHEMA, 100)
    fin = m.finish()
    assert fin["measured_usd"] == pytest.approx(0.02)
    assert SECRET not in json.dumps(fin) + (tmp_path / "spend_ledger.json").read_text()


def test_json_ledger_roundtrip_and_corruption(tmp_path):
    led = JsonLedger(tmp_path / "l.json")
    assert led.phase_total("M1") == 0
    led.record_spend("a", PAID, 0.1, 0.2, 3, "M1")
    led.record_spend("b", PAID, 0.2, 0.2, 1, "M2")
    assert led.phase_total("M1") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        led.record_spend("c", PAID, math.nan, 0, 0, "M1")
    (tmp_path / "l.json").write_text("{not json")
    m = meter(Usage(), led)
    with pytest.raises(BudgetExceeded, match="ledger"):
        m.authorize(PAID, MSGS, SCHEMA, 100)


def test_finish_records_ledger_row():
    led, u = Ledger(), Usage(5.0)
    m = meter(u, led)
    u.v = 5.07
    m.authorize(PAID, MSGS, SCHEMA, 100)
    fin = m.finish()
    assert led.rows[0][:2] == ("r1", PAID) and led.rows[0][2] == pytest.approx(0.07) and led.rows[0][5] == "M1"
    assert fin["phase_remaining_usd"] == pytest.approx(1.50 - 0.07)
