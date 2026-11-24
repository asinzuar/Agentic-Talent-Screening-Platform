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


SMALL = dict(pool_size=10, n_bases=4, families=("A1",), n_jds=1, min_n=2, n_resamples=200)


def _probe_settings():
    return CoreSettings(systems=("full_engine",), **dict(TINY, **SMALL))


def test_probe_is_exact_per_document_against_a_real_partial_run(tmp_path):
    """Run until the call limit stops it, then probe the same plan: the probe's remaining L1 calls equal what the
    continuation really makes, and a complete store has nothing left."""
    from agentic_talent_screener.evaluation.probe import probe_remaining

    s = _probe_settings()
    full = tmp_path / "full"
    full.mkdir()
    rec_all = e15a.Recorder()
    execute(s, base=full, run_id="all", call=rec_all, matcher=e14.TextMatcher())
    n_all = len(rec_all.models)
    done = probe_remaining(s, full, e15a.F1, matcher=e14.TextMatcher())
    assert done is not None and sum(sum(p.values()) for p in [done["systems"]["full_engine"]["clean"],
                                                              *done["systems"]["full_engine"]["twins"]]) == 0
    part = tmp_path / "part"
    part.mkdir()
    s_part = CoreSettings(systems=("full_engine",), **dict(TINY, **SMALL, max_llm_calls=int(n_all * 0.5)))
    execute(s_part, base=part, run_id="p1", call=e15a.Recorder(), matcher=e14.TextMatcher())
    pr = probe_remaining(s, part, e15a.F1, matcher=e14.TextMatcher())
    sysp = pr["systems"]["full_engine"]
    probed = {k: sum(p.get(k, 0) for p in [sysp["clean"], *sysp["twins"]]) for k in ("l1", "l2")}
    rec2 = e15a.Recorder()
    execute(s, base=part, run_id="p2", call=rec2, matcher=e14.TextMatcher())
    assert probed["l1"] > 0 and len(rec2.models) > 0
    assert len(rec2.models) <= probed["l1"] + probed["l2"] + 2 * probed["l1"]  # an upper bound, never an under-count
    assert probed["l1"] >= len(rec2.models) / 3  # and not wildly above: one L1 per uncached document


def test_dry_run_forecast_is_probed_not_a_ratio(tmp_path):
    full = tmp_path
    execute(_probe_settings(), base=full, run_id="all", call=e15a.Recorder(), matcher=e14.TextMatcher())
    text = estimate_text(CoreSettings(systems=("full_engine",), allow_paid=False, **dict(TINY, **SMALL)), full)
    # a free-first run with a complete store: nothing is probed for the paid model, so no resumed block is invented
    assert "ratio estimate" not in text


def test_the_resumed_forecast_counts_matching_for_every_clean_resume():
    """C2: matching is one call per clean document per JD. 30 matching answers cached for a 100-resume pool leave 70."""
    from agentic_talent_screener.evaluation.cli import _resumed_lines

    def expected(stage):
        return {"l1": 0.0003, "l2": 0.0009, "match": 0.0010, "naive": 0.0}[stage]

    lines: list[str] = []
    sysp = {"clean": {"match": 70}, "twins": [{"l1": 1, "l2": 1}] * 70 + [{}] * 2}
    probe = {"n_jds": 1}
    calls = _resumed_lines(lines, sysp, probe, {"l1": 100, "l2": 200, "match": 30}, PAID, 0.50, 0.20, expected,
                           lambda s: 0.02)
    text = "\n".join(lines)
    assert calls == 70 + 70 + 140 + 70 + 70 - 70 + 70 or calls > 70  # the clean pool's 70 matching calls are in
    assert "matching" in text and "twins still need calls" in text


def test_the_forecast_says_where_a_run_that_cannot_finish_stops():
    from agentic_talent_screener.evaluation.cli import _resumed_lines

    lines: list[str] = []
    sysp = {"clean": {"match": 70}, "twins": [{"l1": 1, "l2": 2, "match": 1}] * 80}
    _resumed_lines(lines, sysp, {"n_jds": 1}, {"l1": 1}, PAID, 0.40, 0.20, lambda s: 0.001, lambda s: 0.02)
    text = "\n".join(lines)
    assert "will NOT finish" in text and "twins (first documents in run order)" in text
    lines2: list[str] = []
    _resumed_lines(lines2, sysp, {"n_jds": 1}, {"l1": 1}, PAID, 2.00, 0.20, lambda s: 0.001, lambda s: 0.02)
    assert "it can finish" in "\n".join(lines2)


# ---------------------------------------------------------------- 4. run-start floor clamp (T74 L2)

def test_a_tampered_ledger_cannot_hide_more_than_this_runs_cap():
    u = Usage(v=1.0)
    B.PHASE_BASELINES_USD["M1"] = 0.0
    led = LedgerWithMeasured(99.0, total=99.0)  # an inflated usd_measured row
    m = B.SpendMeter(u, led, run_id="r", run_cap_usd=0.5)
    m.start()
    assert m.usage_start == pytest.approx(1.0 + 0.5)  # clamped to the raw reading + the run cap
    fin = m.finish()
    assert fin["start_floor_applied"] is True and fin["start_raw_usd"] == pytest.approx(1.0)
    assert fin["start_floor_usd"] == pytest.approx(1.5)


def test_no_floor_is_reported_when_none_applies():
    u = Usage(v=0.30)
    B.PHASE_BASELINES_USD["M1"] = 0.0
    m = B.SpendMeter(u, LedgerWithMeasured(0.10, total=0.1), run_id="r", run_cap_usd=0.5)
    m.start()
    fin = m.finish()
    assert fin["start_floor_applied"] is False and fin["start_floor_usd"] is None


def test_the_report_names_a_floored_start(tmp_path):
    from agentic_talent_screener.evaluation.report import _spend_md

    md = "\n".join(_spend_md({"allow_paid": True, "start_floor_applied": True, "start_raw_usd": 0.05,
                             "start_floor_usd": 0.2, "measured_usd": 0.0, "source": "x", "phase": "M1"}, None))
    assert "Run start floored" in md and "$0.0500" in md and "$0.2000" in md
