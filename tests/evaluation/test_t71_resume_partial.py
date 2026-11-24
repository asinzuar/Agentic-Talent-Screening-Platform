"""T71: the per-run hold no longer eats the cap, a lagging /key is not blamed on the next run, a run the cap stops is
scored on what finished, and the dry run predicts a resumed run from the cached answers. Offline, no network."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation import budget as B
from agentic_talent_screener.evaluation.cli import cached_answers, estimate_text
from agentic_talent_screener.evaluation.runner import CoreSettings, execute

_spec = importlib.util.spec_from_file_location("t66_helpers", Path(__file__).with_name("test_t66_live_harness.py"))
t66 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t66)
e15a, e14 = t66.e15a, t66.e14
PAID, Usage, TINY = e15a.PAID, e15a.Usage, e15a.TINY
isolate = e15a.isolate


# ---------------------------------------------------------------- 1. the live runs' early stops

def test_run2_early_stop_is_reproduced_and_fixed():
    """t41v3-run2: 24 matching-sized calls (worst about $0.0194, billed about $0.001) held $0.465 of the $0.50 cap
    and stopped the run at $0.025 billed. The hold per call is now bounded, so the run goes on."""
    m = e15a.meter(Usage())
    n = 0
    for _ in range(300):
        try:
            m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 20_000)  # worst case about $0.019 per call
        except B.BudgetExceeded:
            break
        m.settle([{"cost": 0.001}])
        n += 1
    assert m.held <= B.HOLDBACK_CALLS * B.HOLD_PER_CALL_MAX_USD + 1e-12
    assert n > 100 and m.settled == pytest.approx(0.001 * n)  # many calls, stopped near the cap by billed spend
    assert m.settled + m.held <= 0.5 + 1e-9


def test_the_hold_per_call_is_bounded_by_the_real_cost_scale():
    m = e15a.meter(Usage())
    for _ in range(3):
        w = m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 20_000)
        m.settle([{"cost": 0.0005}])
    assert w > 0.01 and m.held == pytest.approx(3 * B.HOLD_PER_CALL_MAX_USD)


class LedgerWithMeasured(e15a.Ledger):
    def __init__(self, measured, **kw):
        super().__init__(**kw)
        self.measured = measured

    def phase_measured_total(self, phase):
        return self.measured


def test_a_lagging_key_is_not_blamed_on_the_next_run():
    """Run 1 spent $0.20 but /key still shows $0.05 when run 2 starts. Run 2's start is floored at the earlier runs'
    end usage, so run 1's late charges are not attributed to run 2; the phase views still count everything."""
    u = Usage(v=0.05)
    B.PHASE_BASELINES_USD["M1"] = 0.0
    led = LedgerWithMeasured(0.20, total=0.20)
    m = B.SpendMeter(u, led, run_id="r2", run_cap_usd=0.5)
    m.start()
    assert m.usage_start == pytest.approx(0.20)  # floor = baseline + earlier measured spend
    m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 100)
    u.v = 0.20 + 0.0004  # run 1's charges settle on /key (plus a sliver of this run's)
    m.settle([{"cost": 0.0004}])
    assert m.spent() == pytest.approx(0.0004)  # not 0.15 of run 1
    u.v = 0.10  # still lagging: nothing of this run is visible, and that is not "usage went backwards"
    assert m.spent() == pytest.approx(0.0004)  # settles only upward
    v = m.phase_views(m.counted())
    assert v["key"] < 0.5 and v["ledger"] > 0.2  # the phase view still counts everything


def test_without_a_floor_the_raw_reading_stands():
    u = Usage(v=0.30)
    m = B.SpendMeter(u, e15a.Ledger(), run_id="r", run_cap_usd=0.5)  # this ledger cannot say what was measured
    B.PHASE_BASELINES_USD["M1"] = 0.0
    m.start()
    assert m.usage_start == pytest.approx(0.30)


# ---------------------------------------------------------------- 2. a stopped run scores what finished

def _stopped_run(tmp_path, limit, systems=("full_engine",), run_id="stop", **kw):
    rec = e15a.Recorder()
    cfg = dict(TINY, **kw)
    cfg["max_llm_calls"] = limit
    d = execute(CoreSettings(systems=systems, **cfg), base=tmp_path, run_id=run_id, call=rec, matcher=e14.TextMatcher())
    return d, json.loads((d / "raw.json").read_text()), json.loads((d / "results.json").read_text())


def test_a_run_stopped_by_the_cap_is_scored_on_what_finished(tmp_path):
    n_all = _count_calls(tmp_path / "full")
    limit = int(n_all * 0.8)  # stops inside the twins
    d, raw, res = _stopped_run(tmp_path, limit, pool_size=10, n_bases=4, families=("A1",), n_jds=1, min_n=2,
                               n_resamples=200, seed=0)
    rec = raw["systems"]["full_engine"]
    assert rec["status"] == "partial" and rec["partial"]["unfinished"] > 0
    assert rec["reason"].startswith("partial (cap reached after ") and " documents)" in rec["reason"]
    rb = rec["robustness"]
    assert rb is not None and rb["counts"].get("unfinished", 0) > 0
    n_used = rb["asr"]["overall"]["n"]
    assert 0 < n_used < rb["n_twins"]  # finished twins are scored, unfinished ones are not outcomes
    assert "unfinished" in rb["excluded_not_outcomes"]
    assert not res["metrics"]["robustness.asr.full_engine"]["verdict"].startswith("not measurable (full_engine did not run")
    md = (d / "report.md").read_text(encoding="utf-8")
    assert "partial (cap reached after" in md
    assert res["summary"]["fail"] == 0 or all("unfinished" not in r["verdict"] for r in res["slide9"])


def _count_calls(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rec = e15a.Recorder()
    execute(CoreSettings(systems=("full_engine",), **dict(TINY, pool_size=10, n_bases=4, families=("A1",), n_jds=1,
                                                         min_n=2, n_resamples=200)),
            base=tmp_path, run_id="all", call=rec, matcher=e14.TextMatcher())
    return len(rec.models)


def test_unfinished_documents_are_never_counted_as_fails(tmp_path):
    n_all = _count_calls(tmp_path / "full")
    d, raw, res = _stopped_run(tmp_path, int(n_all * 0.8), pool_size=10, n_bases=4, families=("A1",), n_jds=1, min_n=2,
                               n_resamples=200, run_id="nf")
    det = raw["systems"]["full_engine"]["detection"]
    assert det["clean"]["fpr"]["n"] <= 10 and det["clean"]["fpr"]["k"] <= det["clean"]["fpr"]["n"]
    items = [i for i in map(json.loads, (d / "per_item.jsonl").read_text().splitlines()) if i["type"] == "twin"]
    assert any(i["status"] == "unfinished" for i in items)
    assert all(i["success"] is not True for i in items if i["status"] == "unfinished")


def test_a_stop_in_the_clean_pool_skips_ranking_and_says_why(tmp_path):
    d, raw, res = _stopped_run(tmp_path, 12, pool_size=10, n_bases=4, families=("A1",), n_jds=1, min_n=2,
                               n_resamples=200, run_id="pool")
    rec = raw["systems"]["full_engine"]
    assert rec["partial"]["unfinished"] > 0 and rec["ranking"] is None
    assert "ranking skipped" in rec["errors"]["ranking"] and rec["status"] in ("partial", "not_run")
    row = next(r for r in res["slide9"] if r["layer"] == "Matching & ranking")
    assert row["verdict"].startswith("not measurable")


def test_naive_llm_stopped_midway_is_partial_too(tmp_path):
    d, raw, res = _stopped_run(tmp_path, 15, systems=("naive_llm",), pool_size=10, n_bases=4, families=("A1",),
                               n_jds=1, min_n=2, n_resamples=200, run_id="nv")
    rec = raw["systems"]["naive_llm"]
    assert rec["partial"] and rec["partial"]["unfinished"] > 0
    assert rec["status"] in ("partial", "not_run") and "cap reached" in rec["reason"]


def test_a_complete_run_has_no_partial_flag(tmp_path):
    rec = e15a.Recorder()
    d = execute(CoreSettings(systems=("full_engine",), **dict(TINY, families=("A1",), n_jds=1)), base=tmp_path,
                run_id="whole", call=rec, matcher=e14.TextMatcher())
    r = json.loads((d / "raw.json").read_text())["systems"]["full_engine"]
    assert r["partial"] is None and r["status"] == "ok"


def test_is_cap_stop_only_for_caps():
    from agentic_talent_screener.evaluation.cache import CacheMiss
    from agentic_talent_screener.evaluation.runner import CallLimitReached

    assert B.is_cap_stop(B.BudgetExceeded("x", code="run_cap")) and B.is_cap_stop(B.BudgetExceeded("x", code="phase_cap"))
    assert B.is_cap_stop(CallLimitReached("--max-llm-calls 1 reached")) and B.is_cap_stop(B.PolicyExhausted("x"))
    for bad in ("key_unreadable", "baseline_not_set", "base_url_not_pinned", "ledger_unreadable", "refused"):
        assert not B.is_cap_stop(B.BudgetExceeded("x", code=bad)), bad
    assert not B.is_cap_stop(CacheMiss("offline")) and not B.is_cap_stop(ValueError())


# ---------------------------------------------------------------- 3. the dry run for a resumed run

def _store_with(tmp_path, l1, l2, match):
    con = sqlite3.connect(tmp_path / "eval_store.sqlite3")
    con.execute("CREATE TABLE llm_cache (key TEXT PRIMARY KEY, model TEXT NOT NULL, schema_name TEXT NOT NULL, "
                "response TEXT NOT NULL, created_at TEXT NOT NULL) WITHOUT ROWID")
    n = 0
    for schema, count in (("l1b_security_findings", l1), ("l2_resume_extraction", l2), ("semantic_matches", match)):
        for _ in range(count):
            n += 1
            con.execute("INSERT INTO llm_cache VALUES (?,?,?,?,?)", (f"k{n}", PAID, schema, "{}", "t"))
    con.execute("INSERT INTO llm_cache VALUES ('f','x/y:free','l1b_security_findings','{}','t')")
    con.commit()
    con.close()


DOC = dict(systems=("full_engine",), allow_paid=True, budget_usd=0.5, model_policy="paid-direct", families=("A1",),
           pool_size=100, n_bases=20, n_jds=1, max_llm_calls=500)


def test_cached_answers_are_counted_read_only_per_stage(tmp_path):
    _store_with(tmp_path, 100, 200, 30)
    before = (tmp_path / "eval_store.sqlite3").read_bytes()
    assert cached_answers(tmp_path, PAID) == {"l1": 100, "l2": 200, "match": 30}  # the free model's row is not counted
    assert (tmp_path / "eval_store.sqlite3").read_bytes() == before
    assert cached_answers(tmp_path / "none", PAID) == {}


def test_dry_run_predicts_the_remaining_calls_of_a_resumed_run(tmp_path):
    _store_with(tmp_path, 100, 200, 30)  # what t41v3-run1 and run2 left behind
    text = estimate_text(CoreSettings(**DOC), tmp_path)
    assert "cached answers in the store for qwen/qwen3.8-flash: l1 100, l2 200, match 30" in text
    assert "resumed run: about 264 calls remain (L1 80, L2 160, matching 24)" in text
    assert "headroom held back" in text and "$0.20" in text  # 50 held calls x $0.004
    assert "resumed run under the $0.50 cap: it can finish" in text
    fresh = estimate_text(CoreSettings(**DOC))
    assert "resumed run" not in fresh and "cannot finish" in fresh  # a fresh plan needs more than the cap


def test_dry_run_flags_a_resumed_run_that_still_cannot_finish(tmp_path):
    _store_with(tmp_path, 10, 20, 3)
    text = estimate_text(CoreSettings(**dict(DOC, budget_usd=0.2)), tmp_path)
    assert "warning: it cannot finish" in text
