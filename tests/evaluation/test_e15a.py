"""E15a: paid-model guard. Everything offline; the OpenRouter /key endpoint is a mock."""

from __future__ import annotations

import importlib.util
import json
import math
import socket
import traceback
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation import budget as B
from agentic_talent_screener.evaluation.budget import (
    BudgetExceeded, BudgetGuard, ModelRouter, PolicyExhausted, SpendMeter, StoreLedger, fetch_key_usage,
    valid_budget, worst_case_usd,
)
from agentic_talent_screener.evaluation.cache import CachedChat, LLMCache
from agentic_talent_screener.evaluation.cli import main
from agentic_talent_screener.evaluation.config import EvalConfig
from agentic_talent_screener.evaluation.runner import CoreSettings, execute
from agentic_talent_screener.evaluation.store import EvalStore

_spec = importlib.util.spec_from_file_location("e14_helpers", Path(__file__).with_name("test_e14.py"))
e14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e14)

PAID = "qwen/qwen3.8-flash"
F1, F2 = "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"
MSGS = [{"role": "user", "content": "hello"}]
SCHEMA = {"type": "object"}
SECRET = "sk-or-v1-SECRETKEYVALUE0123456789"
TINY = dict(pool_size=10, n_bases=3, n_jds=2, families=("A1",), min_n=5, n_resamples=200, seed=0, max_llm_calls=10_000)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise AssertionError("real network attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.chdir(tmp_path)  # LLMSettings reads .env from cwd: keep any real key out
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.delenv("ATS_LLM_MODEL", raising=False)
    monkeypatch.delenv("ATS_OPENROUTER_BASE_URL", raising=False)


class Usage:
    """Mock of the /key counter. ``lag`` = number of reads before a charge becomes visible."""

    def __init__(self, v=0.0, lag=0):
        self.v, self.lag, self.pending, self.fail = v, lag, [], False

    def charge(self, usd):
        self.pending.append([usd, self.lag])

    def __call__(self):
        if self.fail:
            raise BudgetExceeded("OpenRouter key usage unreadable (X); paid call refused")
        keep = []
        for item in self.pending:
            if item[1] <= 0:
                self.v += item[0]
            else:
                item[1] -= 1
                keep.append(item)
        self.pending = keep
        return self.v

    def true_total(self):
        return self.v + sum(p[0] for p in self.pending)


class Ledger:
    """Test double with the store's behaviour: phase must be open, a row counts as max(measured, estimated)."""

    def __init__(self, total=0.0, open_=True):
        self.total, self.rows, self.open_ = total, [], open_

    def record_spend(self, run_id, model, measured, estimated, requests, phase):
        self.rows.append((run_id, model, measured, estimated, requests, phase))
        self.total += max(measured, estimated)

    def phase_total(self, phase):
        return self.total

    def phase_open(self, phase):
        return self.open_


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


class Recorder(e14.FakeLLM):
    def __init__(self, fail_free_on=None):
        super().__init__()
        self.models, self.fail_free_on = [], fail_free_on or (lambda schema_name: False)

    def __call__(self, messages, *, schema, schema_name, model=None, **kw):
        self.models.append(model)
        if model != PAID and self.fail_free_on(schema_name):
            r429()
        return super().__call__(messages, schema=schema, schema_name=schema_name, model=model, **kw)


def open_store(base):
    with EvalStore.for_root(base) as st:
        StoreLedger(st).open_phase("M1")


# ---------------------------------------------------------------- allow-list, flag, caps


def test_paid_refused_without_allow_paid():
    with pytest.raises(BudgetExceeded):
        BudgetGuard().check(PAID)
    with pytest.raises(ValueError):
        BudgetGuard(0.1)


@pytest.mark.parametrize("model", ["openai/gpt-4o", "qwen/qwen3.8-flash:nitro", "qwen/qwen3.8-flash ", "QWEN/qwen3.8-flash",
                                   "qwen/qwen3.8", "x/" + PAID, "", "qwen/qwen3.8-flash\n", "qwen/qwen3.8-flash​",
                                   "openrouter/auto", "openrouter/auto:free", "qwen/qwen3.8-flash:free:nitro"])
def test_non_allowlisted_paid_id_refused(model):
    g = BudgetGuard(0.5, allow_paid=True, meter=meter(Usage()))
    with pytest.raises(BudgetExceeded):
        g.check(model)
    g.check(PAID)


def test_allow_paid_with_zero_cap_refuses():
    with pytest.raises(BudgetExceeded):
        BudgetGuard(0.0, allow_paid=True).check(PAID)


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


# ---------------------------------------------------------------- caps, reservations, lag


def test_per_run_cap_blocks_on_counted_spend():
    u = Usage(10.0)
    m = meter(u)
    w = worst_case_usd(PAID, MSGS, SCHEMA, 1000)
    assert 0 < w < 0.01
    m.authorize(PAID, MSGS, SCHEMA, 1000)
    u.v = 10.0 + 0.50 - w / 2  # measured delta + worst case now exceeds 0.50
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_reservations_alone_stop_the_run_when_key_reads_zero():
    """/key never moves; reserved worst cases still reach the cap (counted = max(measured, reserved))."""
    m = meter(Usage())
    n = 0
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        for _ in range(10_000):
            m.authorize(PAID, MSGS, SCHEMA, 20_000)
            n += 1
    assert 0 < n < 100 and m.reserved <= 0.50 + 1e-9


def test_reservation_is_never_released_on_failure(tmp_path):
    def call(*a, **k):
        raise RuntimeError("OpenRouter request failed (500); provider error")

    c, router, m = chat(tmp_path, call, free=(F1,), switch=1)
    r = []
    for i in range(3):
        try:
            c([{"role": "user", "content": f"q{i}"}], schema=SCHEMA, schema_name="s", max_tokens=1000)
        except RuntimeError:
            pass
        r.append(m.reserved)
    # the free model raised a 500 (not a 429), so no paid reservation was taken
    assert r == [0, 0, 0]

    def r429_then_fail(*a, model, **k):
        if model == PAID:
            raise RuntimeError("OpenRouter request failed (500); provider error")
        r429()

    c, router, m = chat(tmp_path / "b", r429_then_fail, free=(F1,), switch=1)
    for i in range(2):
        with pytest.raises(RuntimeError):
            c([{"role": "user", "content": f"z{i}"}], schema=SCHEMA, schema_name="s", max_tokens=1000)
    assert m.reserved == pytest.approx(2 * worst_case_usd(PAID, [{"role": "user", "content": "z0"}], SCHEMA, 1000), rel=0.1)
    assert m.paid_calls == 2


def test_cap_holds_when_key_usage_lags(tmp_path):
    """A lagging /key (charges visible 2 reads late) must not let true spend pass the cap."""
    u = Usage(lag=2)
    big = 480_000  # one call's worst case is about 0.45 USD

    def call(messages, **kw):
        u.charge(worst_case_usd(PAID, messages, SCHEMA, big))
        return {}

    c, _router, m = chat(tmp_path, call, free=(F1,), switch=1, usage=u)
    c2 = CachedChat(LLMCache(tmp_path / "x"), BudgetGuard(0.5, allow_paid=True, meter=m,
                                                         router=ModelRouter([F1], PAID, switch_after=1)),
                    call=lambda messages, model, **kw: r429() if model != PAID else call(messages, **kw))
    for i in range(3):
        try:
            c2([{"role": "user", "content": f"q{i}"}], schema=SCHEMA, schema_name="s", max_tokens=big)
        except BudgetExceeded:
            break
    assert u.true_total() <= 0.5 + 1e-9, f"true spend {u.true_total():.3f} exceeds the 0.50 cap"


def test_billed_then_failed_calls_count(tmp_path):
    u = Usage()
    seen = []

    def call(*a, model, **k):
        if model != PAID:
            r429()
        seen.append(model)
        u.v += 0.30  # billed ...
        raise RuntimeError("OpenRouter request failed (500); provider error")  # ... then failed

    c, router, m = chat(tmp_path, call, usage=u, free=(F1,), switch=1)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=1000)
    with pytest.raises(BudgetExceeded, match="per-run cap"):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=1000)
    assert seen == [PAID, PAID] and m.last_delta >= 0.5 - 1e-9


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
    u.v = 0.21
    with pytest.raises(BudgetExceeded, match="phase cap"):
        m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_closed_phase_refuses():
    m = meter(Usage(), Ledger(open_=False))
    with pytest.raises(BudgetExceeded, match="not open"):
        m.authorize(PAID, MSGS, SCHEMA, 100)


def test_unreadable_key_refuses_at_start_and_later():
    u = Usage()
    u.fail = True
    m = meter(u)
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


@pytest.mark.parametrize("readings", [[None], ["0.1"], [math.nan], [math.inf], [-1.0], [True], [{}],
                                      [0.0, None], [0.0, "x"], [0.0, math.nan], [0.0, math.inf], [0.3, 0.2], [0.0, -5.0],
                                      [0.0, True]])
def test_garbage_usage_never_authorises(readings):
    seq = iter(readings + [readings[-1]] * 5)
    m = SpendMeter(lambda: next(seq), Ledger(), run_id="r", run_cap_usd=0.5)
    m.start()
    with pytest.raises(BudgetExceeded):
        m.authorize(PAID, MSGS, SCHEMA, 100)


def test_token_bound_covers_non_ascii_input():
    text = "​" * 3000 + "\U0001F600" * 1000 + "漢" * 1000
    msgs = [{"role": "user", "content": text}]
    in_price = B.PAID_PRICES_USD_PER_TOKEN[PAID][0]
    out_price = B.PAID_PRICES_USD_PER_TOKEN[PAID][1]
    w0 = worst_case_usd(PAID, msgs, SCHEMA, 1, max_retries=0) - out_price
    assert w0 / in_price >= len(text.encode("utf-8"))


def test_retry_multiplier():
    w0 = worst_case_usd(PAID, MSGS, SCHEMA, 1000, max_retries=0)
    assert worst_case_usd(PAID, MSGS, SCHEMA, 1000) == pytest.approx(2 * w0)


def test_final_read_failure_does_not_undercount_ledger():
    led = Ledger()
    state = {"v": 0.0, "fail": False}

    def reader():
        if state["fail"]:
            raise BudgetExceeded("unreadable")
        return state["v"]

    m = SpendMeter(reader, led, run_id="r", run_cap_usd=0.5)
    m.start()
    m.authorize(PAID, MSGS, SCHEMA, 100_000)
    state["fail"] = True
    fin = m.finish()
    assert led.rows[0][2] >= m.reserved > 0 and led.rows[0][3] == m.reserved and fin["measured_ok"] is False


def test_finish_records_both_figures():
    led, u = Ledger(), Usage(5.0)
    m = meter(u, led)
    m.authorize(PAID, MSGS, SCHEMA, 100)
    u.v = 5.07
    fin = m.finish()
    row = led.rows[0]
    assert row[:2] == ("r1", PAID) and row[2] == pytest.approx(0.07) and row[3] == pytest.approx(m.reserved)
    assert row[5] == "M1" and fin["phase_remaining_usd"] == pytest.approx(1.50 - max(0.07, m.reserved))


# ---------------------------------------------------------------- routing


def test_free_first_fallback_order(tmp_path):
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        if model != PAID:
            r429()
        return {"ok": model}

    c, router, m = chat(tmp_path, call)
    assert c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100) == {"ok": PAID}
    assert seen == [F1] * 3 + [F2] * 3 + [PAID]
    assert [(s["from"], s["to"]) for s in router.switches] == [(F1, F2), (F2, PAID)]
    assert router.summary()["calls_by_system"]["unattributed"] == {PAID: 1}


def test_router_never_starts_with_a_paid_or_empty_chain():
    for free in ([], [PAID], [PAID, F1], ["openrouter/auto:free"]):
        with pytest.raises(ValueError):
            ModelRouter(free, PAID)
    with pytest.raises(ValueError):
        ModelRouter([F1], "openai/gpt-4o")


def test_free_only_stops_when_both_rate_limited(tmp_path):
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        r429()

    c, router, _ = chat(tmp_path, call, allow_paid=False)
    with pytest.raises(PolicyExhausted):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)
    assert seen == [F1] * 3 + [F2] * 3 and router.summary()["stopped"]


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


def test_non_429_error_never_falls_back_to_paid(tmp_path):
    seen = []

    def call(*a, model, **k):
        seen.append(model)
        raise RuntimeError("OpenRouter request failed (500); provider error")

    c, *_ = chat(tmp_path, call)
    with pytest.raises(RuntimeError):
        c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)
    assert seen == [F1]


def test_unrouted_path_authorises_paid(tmp_path):
    """T51 F3: without a router a paid id still goes through the meter (and is refused without max_tokens)."""
    m = meter(Usage())
    g = BudgetGuard(0.5, allow_paid=True, meter=m)
    called = []
    c = CachedChat(LLMCache(tmp_path), g, call=lambda *a, **k: called.append(1) or {}, default_model=lambda: PAID)
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        c(MSGS, schema=SCHEMA, schema_name="s")
    assert called == []
    c(MSGS, schema=SCHEMA, schema_name="s", max_tokens=100)
    assert called == [1] and m.paid_calls == 1 and m.reserved > 0


# ---------------------------------------------------------------- /key reader


class Resp:
    def __init__(self, body=None, status=200):
        self.body, self.status_code = body if body is not None else {"data": {"usage": 1.25}}, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} for Bearer {SECRET}")

    def json(self):
        return self.body


def test_key_fetch_parses_pins_host_and_never_redirects():
    calls = []

    def get(url, headers, timeout, allow_redirects):
        calls.append((url, timeout, allow_redirects))
        return Resp()

    assert fetch_key_usage(get) == 1.25
    assert calls[0][0] == "https://openrouter.ai/api/v1/key" and calls[0][1] <= 15 and calls[0][2] is False


@pytest.mark.parametrize("base", ["http://openrouter.ai/api/v1", "https://evil.example/api/v1",
                                  "https://openrouter.ai.evil.example/api/v1", "https://openrouter.ai@evil.example/x",
                                  "https://openrouter.ai:8443/api/v1", "ftp://openrouter.ai/api/v1"])
def test_key_never_sent_to_another_host(monkeypatch, base):
    monkeypatch.setenv("ATS_OPENROUTER_BASE_URL", base)
    sent = []
    with pytest.raises(BudgetExceeded) as e:
        fetch_key_usage(lambda *a, **k: sent.append(a) or Resp())
    assert sent == [] and SECRET not in str(e.value)


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_key_redirect_refused(status):
    with pytest.raises(BudgetExceeded, match="redirect"):
        fetch_key_usage(lambda *a, **k: Resp(status=status))


@pytest.mark.parametrize("body", [{"data": {"usage": "x"}}, {"data": {"usage": float("nan")}}, {"data": {"usage": -1}},
                                  {"data": {"usage": True}}, {"data": {}}, {}, {"data": {"usage": 1e400}}])
def test_key_garbage_body_refused(body):
    with pytest.raises(BudgetExceeded) as e:
        fetch_key_usage(lambda *a, **k: Resp(body))
    assert SECRET not in str(e.value)


def test_key_errors_never_leak(monkeypatch):
    with pytest.raises(BudgetExceeded) as e:
        fetch_key_usage(lambda *a, **k: Resp(status=401))
    assert SECRET not in "".join(traceback.format_exception(e.value))

    def boom(url, headers, **k):
        raise ConnectionError(f"failed {url} {headers}")

    with pytest.raises(BudgetExceeded) as e:
        fetch_key_usage(boom)
    assert SECRET not in "".join(traceback.format_exception(e.value)) or e.value.__suppress_context__
    assert SECRET not in str(e.value)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("ATS_OPENROUTER_API_KEY", "")
    with pytest.raises(BudgetExceeded):
        fetch_key_usage(lambda *a, **k: Resp())


# ---------------------------------------------------------------- the committed store is the only ledger


def test_store_ledger_satisfies_the_protocol_and_counts_max(tmp_path):
    with EvalStore.for_root(tmp_path) as st:
        led = StoreLedger(st)
        assert not led.phase_open("M1")
        assert led.open_phase("M1") and not led.open_phase("M1")
        led.record_spend("a", PAID, 0.10, 0.30, 3, "M1")  # reserved above measured
        led.record_spend("b", PAID, 0.40, 0.20, 1, "M1")  # measured above reserved
        led.record_spend("c", PAID, 9.0, 9.0, 1, "M2")
        assert led.phase_total("M1") == pytest.approx(0.70)
        u = Usage()
        m = SpendMeter(u, led, run_id="r", run_cap_usd=0.5)
        m.start()
        m.authorize(PAID, MSGS, SCHEMA, 1000)  # 0.70 + small < 1.50
        led.record_spend("d", PAID, 0.80, 0.0, 1, "M1")
        with pytest.raises(BudgetExceeded, match="phase cap"):
            m.authorize(PAID, MSGS, SCHEMA, 1000)


def test_missing_store_refuses_paid_calls(tmp_path):
    """A fresh or deleted store has no phase-open row, so paid calls are refused (the reset attack)."""
    with EvalStore.for_root(tmp_path) as st:
        m = meter(Usage(), StoreLedger(st))
        with pytest.raises(BudgetExceeded, match="not open"):
            m.authorize(PAID, MSGS, SCHEMA, 100)


def test_corrupt_store_refuses(tmp_path):
    with EvalStore.for_root(tmp_path) as st:
        led = StoreLedger(st)
        led.open_phase("M1")
        st.conn.execute("DROP TABLE spend_ledger")
    with EvalStore.for_root(tmp_path) as st2:
        pass  # migrations recreate the table, but the phase-open row is gone
    with EvalStore.for_root(tmp_path) as st3:
        m = meter(Usage(), StoreLedger(st3))
        with pytest.raises(BudgetExceeded):
            m.authorize(PAID, MSGS, SCHEMA, 100)


def test_runs_dir_cannot_reset_the_phase_total(tmp_path):
    open_store(tmp_path)
    with EvalStore.for_root(tmp_path) as st:
        StoreLedger(st).record_spend("old", PAID, 1.4999, 1.4999, 10, "M1")
    rec = Recorder(fail_free_on=lambda s: True)
    for rd in ("runs", "runs/alt"):  # a new --runs-dir must not matter
        d = execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, switch_after_429=1,
                                 runs_dir=rd, **TINY), base=tmp_path, runs_dir=rd, run_id=f"r-{len(rd)}", call=rec,
                    matcher=e14.TextMatcher(), key_usage=Usage())
        assert PAID not in rec.models
        raw = json.loads((d / "raw.json").read_text())
        assert raw["spend"]["paid_calls"] == 0
    assert not list(tmp_path.rglob("spend_ledger.json"))  # the JSON fallback is gone


# ---------------------------------------------------------------- runner end to end


def test_free_first_even_when_default_model_is_paid(tmp_path, monkeypatch):
    open_store(tmp_path)
    monkeypatch.setenv("ATS_LLM_MODEL", PAID)
    rec = Recorder()
    d = execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, **TINY), base=tmp_path,
                run_id="pf", call=rec, matcher=e14.TextMatcher(), key_usage=Usage())
    assert rec.models and rec.models[0] == F1 and PAID not in rec.models
    raw = json.loads((d / "raw.json").read_text())
    assert raw["model_routing"]["chain"][0] == F1 and raw["model_routing"]["ignored_default_model"] == PAID


def test_explicit_paid_model_arg_refused(tmp_path):
    with pytest.raises(ValueError, match="free"):
        execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, **TINY), base=tmp_path,
                run_id="pm", call=Recorder(), model=PAID, matcher=e14.TextMatcher(), key_usage=Usage())


@pytest.mark.parametrize("how", ["model_arg", "env"])
def test_runner_never_calls_paid_without_flag(tmp_path, monkeypatch, how):
    rec = Recorder(fail_free_on=lambda s: True)
    kw = {}
    if how == "env":
        monkeypatch.setenv("ATS_LLM_MODEL", PAID)
    else:
        kw["model"] = F1
    d = execute(CoreSettings(systems=("naive_llm", "full_engine"), **TINY), base=tmp_path, run_id="nf", call=rec,
                matcher=e14.TextMatcher(), **kw)
    assert PAID not in rec.models
    raw = json.loads((d / "raw.json").read_text())
    assert raw["spend"]["allow_paid"] is False and raw["spend"]["paid_calls"] == 0 and raw["model_routing"]["stopped"]


def test_paid_run_records_one_ledger_row_and_report(tmp_path):
    open_store(tmp_path)
    rec = Recorder(fail_free_on=lambda s: True)
    d = execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, switch_after_429=1, **TINY),
                base=tmp_path, run_id="paid1", call=rec, matcher=e14.TextMatcher(), key_usage=Usage(0.0))
    raw = json.loads((d / "raw.json").read_text())
    assert rec.models[:2] == [F1, F2] and PAID in rec.models
    with EvalStore.for_root(tmp_path) as st:
        rows = st.conn.execute("SELECT run_id, model, usd_measured, usd_estimated, requests, phase FROM spend_ledger "
                               "WHERE run_id=?", ("paid1",)).fetchall()
    assert len(rows) == 1 and rows[0][5] == "M1" and rows[0][3] > 0 and rows[0][4] == raw["spend"]["paid_calls"]
    assert raw["spend"]["reserved_usd"] <= 0.5 + 1e-9
    md = (d / "report.md").read_text(encoding="utf-8")
    assert "## Spend and models" in md and "Measured spend this run" in md and "remaining" in md
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["spend"]["phase"] == "M1" and manifest["allow_paid"] is True
    leaked = [str(p) for p in tmp_path.rglob("*") if p.is_file() and SECRET.encode() in p.read_bytes()]
    assert leaked == []


def test_cross_system_model_mix_is_disclosed(tmp_path):
    open_store(tmp_path)
    rec = Recorder(fail_free_on=lambda s: s != "naive_screener_score")
    d = execute(CoreSettings(systems=("naive_llm", "full_engine"), allow_paid=True, budget_usd=0.5, switch_after_429=1,
                             **TINY), base=tmp_path, run_id="mix", call=rec, matcher=e14.TextMatcher(),
                key_usage=Usage())
    raw = json.loads((d / "raw.json").read_text())
    by = {s: frozenset(p) for s, p in raw["model_routing"]["calls_by_system"].items()}
    assert by.get("naive_llm") == {F1} and PAID in by.get("full_engine", frozenset())
    results = json.loads((d / "results.json").read_text())
    assert any(c["kind"] == "model_mix" for c in results["caveats"])
    assert "different models" in (d / "report.md").read_text(encoding="utf-8")


def test_trace_records_the_model_actually_used(tmp_path):
    open_store(tmp_path)
    rec = Recorder(fail_free_on=lambda s: True)
    d = execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, switch_after_429=1, **TINY),
                base=tmp_path, run_id="tr", call=rec, model=F1, matcher=e14.TextMatcher(), key_usage=Usage())
    traces = [json.loads(line) for line in (d / "traces.jsonl").read_text().splitlines() if line.strip()]
    ok = [t for t in traces if t.get("type") == "call" and t.get("error_kind") is None]
    assert ok and all(t["model"] == PAID for t in ok)


def test_offline_allow_paid_makes_no_network_call(tmp_path, monkeypatch):
    import requests

    got = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: got.append(a))
    execute(CoreSettings(systems=("keyword_ats",), allow_paid=True, budget_usd=0.5, offline=True, **TINY),
            base=tmp_path, run_id="off", call=Recorder(), matcher=e14.TextMatcher())
    assert got == []


def test_default_key_reader_never_writes_key_and_does_not_redirect(tmp_path, monkeypatch):
    import requests

    open_store(tmp_path)
    seen = []

    def fake_get(url, headers, timeout, allow_redirects):
        seen.append((url, allow_redirects))
        return Resp({"data": {"usage": 1.0}})

    monkeypatch.setattr(requests, "get", fake_get)
    rec = Recorder(fail_free_on=lambda s: True)
    execute(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, switch_after_429=1, **TINY),
            base=tmp_path, run_id="k", call=rec, matcher=e14.TextMatcher())
    assert seen and all(u == "https://openrouter.ai/api/v1/key" and r is False for u, r in seen)
    assert [p for p in tmp_path.rglob("*") if p.is_file() and SECRET.encode() in p.read_bytes()] == []


def test_phase_open_cli(tmp_path, monkeypatch, capsys):
    from agentic_talent_screener.evaluation import cli

    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)
    assert main(["phase-open"]) == 0 and "opened" in capsys.readouterr().out
    assert main(["phase-open"]) == 0 and "already open" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["run", "--budget-usd", "0.01"], ["run", "--allow-paid", "--budget-usd", "0"],
                                  ["run", "--allow-paid", "--budget-usd", "0.6"]])
def test_cli_refuses_inconsistent_paid_flags(argv):
    try:
        rc = main(argv)
    except SystemExit as e:
        rc = e.code
    assert rc not in (0, None)
