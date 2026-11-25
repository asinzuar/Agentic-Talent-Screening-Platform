"""T65: E11 (claims suite) on top of E15a (paid guard) and E15b (store). Offline; the network is blocked."""

from __future__ import annotations

import json
import socket

import pytest

from agentic_talent_screener.evaluation import budget as B
from agentic_talent_screener.evaluation import cli
from agentic_talent_screener.evaluation.budget import BudgetGuard, ModelRouter
from agentic_talent_screener.evaluation.cache import CachedChat, LLMCache
from agentic_talent_screener.evaluation.claims_suite import ClaimsSettings, execute_claims
from agentic_talent_screener.evaluation.store import STORE_FILE, EvalStore
from agentic_talent_screener.evaluation.systems import EvalCandidate, NaiveLLMScreener
from agentic_talent_screener.evaluation.jds import synthetic_jds

from test_e11_claims import FREE, FakeLLM  # noqa: E402  (pytest puts tests/evaluation on sys.path)

TINY = dict(pool_size=3, n_bases=1, n_jds=1, min_n=5, seed=0, max_llm_calls=10_000)
PAID = "qwen/qwen3.8-flash"
F1, F2 = "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("real network attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_claims_cache_lives_in_the_shared_store_and_replays_from_it(tmp_path):
    execute_claims(ClaimsSettings(**TINY), base=tmp_path, run_id="a", call=FakeLLM(), model=FREE)
    assert (tmp_path / STORE_FILE).is_file()
    assert not (tmp_path / "data/eval/cache").exists()  # no per-suite JSON cache any more
    with EvalStore.for_root(tmp_path) as store:
        rows = store.conn.execute("SELECT model, schema_name FROM llm_cache").fetchall()
    assert rows and {r[0] for r in rows} == {FREE} and {"l2_resume_extraction", "l1b_security_findings"} <= {r[1] for r in rows}

    def boom(*a, **k):
        raise AssertionError("network used")

    d = execute_claims(ClaimsSettings(**{**TINY, "offline": True}), base=tmp_path, run_id="b", call=boom, model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["status"] == "complete" and res["live_calls_used"] == 0 and res["cache"]["misses"] == 0


def test_claims_runs_are_recorded_in_the_stores_runs_table(tmp_path):
    execute_claims(ClaimsSettings(**TINY), base=tmp_path, run_id="rec1", call=FakeLLM(), model=FREE)
    with EvalStore.for_root(tmp_path) as store:
        runs = store.list_runs()
        assert "rec1" in runs
    # compare reads the stored summaries by run id, with the run directories gone
    import shutil

    shutil.rmtree(tmp_path / "runs/rec1")
    execute_claims(ClaimsSettings(**TINY), base=tmp_path, run_id="rec2", call=FakeLLM(), model=FREE)
    shutil.rmtree(tmp_path / "runs/rec2")
    from agentic_talent_screener.evaluation.report import compare_runs

    diff = compare_runs(tmp_path / "runs/rec1", tmp_path / "runs/rec2", store_root=tmp_path)
    assert diff["n_regressions"] == 0 and diff["n_metrics"] > 5


def test_the_cache_put_carries_model_schema_and_messages(tmp_path):
    seen = {}

    class Spy(LLMCache):
        def put(self, key, value, *, model, schema_name, messages):
            seen["args"] = (model, schema_name, messages)
            super().put(key, value, model=model, schema_name=schema_name, messages=messages)

    store = EvalStore.for_root(tmp_path)
    chat = CachedChat(Spy(store, synthetic=True), BudgetGuard(0.0), call=lambda m, **k: {"ok": 1})
    chat([{"role": "user", "content": "x"}], schema={"type": "object"}, schema_name="s", model=FREE)
    assert seen["args"] == (FREE, "s", [{"role": "user", "content": "x"}])
    store.close()


def test_claims_refuses_allow_paid_with_a_clear_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)
    for extra in (["--allow-paid", "--budget-usd", "0.5"], ["--allow-paid"]):
        assert cli.main(["run", "--suite", "claims", "--offline", "--run-id", "np", *extra]) == 2
        err = capsys.readouterr().err
        assert "refused [free_only]" in err and "no paid path" in err
    assert not (tmp_path / "runs/np").exists()  # refused before anything was written


def test_claims_never_pays_even_when_a_paid_model_is_requested(tmp_path):
    d = execute_claims(ClaimsSettings(**TINY), base=tmp_path, run_id="paid", call=FakeLLM(), model=PAID)
    res = json.loads((d / "results.json").read_text())
    assert res["errors"]["extraction"] == B.REFUSAL_MESSAGES["free_only"] and res["live_calls_used"] == 0
    assert "paid" not in json.dumps(res["settings"]).lower().replace("allow_paid", "")  # no paid setting is recorded


# ----------------------------------------------------------------------------- NaiveLLMScreener: max_tokens and timeout


def _naive_call(seen):
    def call(messages, *, schema, schema_name, model=None, **kw):
        seen.append({"model": model, **kw})
        return {"score": 5.0}
    return call


def _candidate():
    return [EvalCandidate("S-0", text="Python, Docker")]


def test_naive_screener_max_tokens_and_timeout_reach_the_plain_free_path(tmp_path):
    seen = []
    store = EvalStore.for_root(tmp_path)
    chat = CachedChat(LLMCache(store, synthetic=True), BudgetGuard(0.0), call=_naive_call(seen), default_model=lambda: FREE)
    NaiveLLMScreener(chat, FREE, max_tokens=77, timeout=33).score(_candidate(), synthetic_jds(1, 0)[0])
    assert seen and seen[0]["max_tokens"] == 77 and seen[0]["timeout"] == 33
    store.close()


def test_naive_screener_max_tokens_and_timeout_reach_the_routed_path(tmp_path):
    seen = []
    store = EvalStore.for_root(tmp_path)
    router = ModelRouter([F1, F2], None)
    chat = CachedChat(LLMCache(store, synthetic=True), BudgetGuard(0.0, router=router), call=_naive_call(seen))
    NaiveLLMScreener(chat, None, max_tokens=88, timeout=44).score(_candidate(), synthetic_jds(1, 0)[0])
    assert seen[0]["model"] == F1 and seen[0]["max_tokens"] == 88 and seen[0]["timeout"] == 44
    store.close()


def test_naive_screener_max_tokens_reaches_the_paid_authorisation_and_the_call(tmp_path):
    """On the paid fallback the guard must see the same max_tokens the call gets (a paid call without one is refused)."""
    seen, authorised = [], []

    class Meter:
        import threading
        lock = threading.Lock()

        def authorize(self, model, messages, schema, max_tokens):
            authorised.append((model, max_tokens))

    def call(messages, *, schema, schema_name, model=None, **kw):
        seen.append({"model": model, **kw})
        if model != PAID:
            raise RuntimeError("OpenRouter request failed (429); rate limited")
        return {"score": 7.0}

    store = EvalStore.for_root(tmp_path)
    router = ModelRouter([F1, F2], PAID, switch_after=1)
    guard = BudgetGuard(0.5, allow_paid=True, meter=Meter(), router=router)
    chat = CachedChat(LLMCache(store, synthetic=True), guard, call=call)
    NaiveLLMScreener(chat, None, max_tokens=99, timeout=55).score(_candidate(), synthetic_jds(1, 0)[0])
    paid_calls = [s for s in seen if s["model"] == PAID]
    assert paid_calls and paid_calls[0]["max_tokens"] == 99 and paid_calls[0]["timeout"] == 55
    assert authorised == [(PAID, 99)]
    store.close()
