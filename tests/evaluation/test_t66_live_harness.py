"""T66 (E15d): paid fallback reachable, provider outage is not a FAIL, compare by bare run id, paid-direct policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation.budget import ModelRouter, is_rate_limited
from agentic_talent_screener.evaluation.cli import _run_ref, build_parser, estimate_text, main
from agentic_talent_screener.evaluation.report import compare_runs
from agentic_talent_screener.evaluation.runner import CoreSettings, execute

_spec = importlib.util.spec_from_file_location("e15a_helpers", Path(__file__).with_name("test_e15a.py"))
e15a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e15a)
e14 = e15a.e14
PAID, F1, F2, TINY, Usage = e15a.PAID, e15a.F1, e15a.F2, e15a.TINY, e15a.Usage
isolate = e15a.isolate  # no network, key kept out, baseline reset


def _paid_run(tmp_path, rec, run_id, **kw):
    e15a.open_store(tmp_path)
    return execute(CoreSettings(systems=("naive_llm", "full_engine"), allow_paid=True, budget_usd=0.5, **TINY, **kw),
                   base=tmp_path, run_id=run_id, call=rec, matcher=e14.TextMatcher(), key_usage=Usage())


# ------------------------------------------------------------------ 1. the paid fallback is reachable

@pytest.mark.parametrize("wording", ["status", "body"])
def test_46_straight_429s_from_both_free_models_reach_the_paid_model(tmp_path, wording):
    errors = {
        "status": "OpenRouter request failed (429); top_level_keys=['error']",
        "body": "OpenRouter response did not contain chat content; top_level_keys=['error', 'user_id']; "
                "error_keys=['code', 'message']",
    }
    free_calls = {"n": 0}

    class Rec(e15a.Recorder):
        def __call__(self, messages, *, schema, schema_name, model=None, **kw):
            if model != PAID:
                free_calls["n"] += 1
                self.models.append(model)
                raise RuntimeError(errors[wording])
            return super().__call__(messages, schema=schema, schema_name=schema_name, model=model, **kw)

    rec = Rec()
    d = _paid_run(tmp_path, rec, f"fb-{wording}")
    raw = json.loads((d / "raw.json").read_text())
    assert free_calls["n"] == 6 and PAID in rec.models  # both free models tripped once, then paid for the rest
    assert raw["spend"]["paid_calls"] > 0
    assert [s["to"] for s in raw["model_routing"]["switches"]] == [F2, PAID]
    used = {m for per in raw["model_routing"]["calls_by_system"].values() for m in per}
    assert used == {PAID}  # shared across systems and calls: nobody went back to a free model
    assert raw["spend"]["reserved_usd"] <= 0.5 + 1e-9


def test_46_failures_in_a_row_share_one_router_state():
    router = ModelRouter([F1, F2], PAID, switch_after=3)
    seen = []
    for _ in range(46):
        m = router.current()
        seen.append(m)
        if m != PAID:
            router.on_429(m)
    assert seen[:6] == [F1] * 3 + [F2] * 3 and set(seen[6:]) == {PAID}


def test_is_rate_limited_does_not_widen_to_other_errors():
    assert is_rate_limited(RuntimeError("OpenRouter request failed (429); x"))
    assert not is_rate_limited(RuntimeError("OpenRouter request failed (500); x"))
    assert not is_rate_limited(RuntimeError("OpenRouter response did not contain chat content; choices_count=0"))
    assert not is_rate_limited(ValueError("bad json"))


# ------------------------------------------------------------------ 2. an outage is not measurable, never fail

def test_total_outage_reads_not_measurable_never_fail_or_pass(tmp_path):
    class Down(e15a.Recorder):
        def __call__(self, messages, *, schema, schema_name, model=None, **kw):
            self.models.append(model)
            raise RuntimeError("OpenRouter request failed (503); provider error")

    d = execute(CoreSettings(systems=("naive_llm", "full_engine"), **TINY), base=tmp_path, run_id="down",
                call=Down(), matcher=e14.TextMatcher())
    res = json.loads((d / "results.json").read_text())
    raw = json.loads((d / "raw.json").read_text())
    out = raw["systems"]["full_engine"]["outage"]
    assert out["failed"] > 0 and out["failed"] == out["total"]
    for key, m in res["metrics"].items():
        if m["system"] in ("full_engine", "naive_llm"):
            assert m["verdict"].startswith("not measurable (") and "provider outage:" in m["verdict"], key
            assert m["value"] is None
    for row in res["slide9"]:
        if row["layer"] in ("Matching & ranking", "Robustness", "Detection", "Efficiency", "Agent reliability"):
            assert row["verdict"].startswith("not measurable") and "provider outage" in row["verdict"], row["layer"]
    assert res["summary"]["fail"] == 0 and res["summary"]["pass"] == 0
    assert f"provider outage: {out['failed']}/{out['total']} calls failed" in (d / "report.md").read_text(encoding="utf-8")


def test_healthy_run_has_no_outage_flag(tmp_path):
    d = execute(CoreSettings(systems=("naive_llm", "full_engine"), **TINY), base=tmp_path, run_id="ok",
                call=e15a.Recorder(), matcher=e14.TextMatcher())
    raw = json.loads((d / "raw.json").read_text())
    assert raw["systems"]["full_engine"]["outage"] is None and raw["systems"]["naive_llm"]["outage"] is None


def test_outage_rule_needs_failures_and_fewer_than_min_n_successes():
    from agentic_talent_screener.evaluation.runner import _outage

    def call(kind):
        return {"type": "call", "error_kind": kind}

    assert _outage([call("provider_error")] * 5, 30) == {"failed": 5, "total": 5, "ok": 0}
    assert _outage([call(None)] * 3 + [call("transport")], 30) == {"failed": 1, "total": 4, "ok": 3}
    assert _outage([call(None)] * 40 + [call("provider_error")] * 10, 30) is None  # enough usable output
    assert _outage([call(None)] * 5, 30) is None  # no failures: plain n<min handles it
    assert _outage([call("budget_guard")] * 5, 30) is None  # refusals are not an outage


# ------------------------------------------------------------------ 3. compare by bare run id

def test_compare_by_bare_run_id_from_another_cwd(tmp_path, monkeypatch):
    d = execute(CoreSettings(systems=("keyword_ats",), **TINY), base=tmp_path, run_id="bare-a")
    execute(CoreSettings(systems=("keyword_ats",), **TINY), base=tmp_path, run_id="bare-b")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    diff = compare_runs("bare-a", "bare-b", store_root=tmp_path)  # store first
    assert diff["n_regressions"] == 0 and diff["n_metrics"] > 0
    (tmp_path / "eval_store.sqlite3").unlink()  # then runs/<id>
    assert compare_runs("bare-a", "bare-b", store_root=tmp_path)["n_metrics"] == diff["n_metrics"]
    assert (d / "results.json").is_file()
    with pytest.raises(ValueError, match="not in the eval store"):
        compare_runs("nope", "bare-a", store_root=tmp_path)


def test_cli_compare_passes_bare_ids_through(tmp_path, monkeypatch):
    from agentic_talent_screener.evaluation import cli

    assert _run_ref("t41v2-core", tmp_path) == "t41v2-core"  # no "must stay under data/eval or runs" error
    assert isinstance(_run_ref("runs/x", tmp_path), Path)
    seen = {}
    monkeypatch.setattr("agentic_talent_screener.evaluation.report.compare_runs",
                        lambda a, b, tol, store_root=None: seen.update(a=a, b=b) or {"n_regressions": 0})
    monkeypatch.setattr("agentic_talent_screener.evaluation.report.compare_markdown", lambda d, show_all=False: "ok")
    assert cli.main(["compare", "idA", "idB"]) == 0 and (seen["a"], seen["b"]) == ("idA", "idB")


# ------------------------------------------------------------------ 4. paid-direct

def test_paid_direct_router_skips_free_models_and_needs_the_paid_model():
    r = ModelRouter([F1, F2], PAID, policy="paid-direct")
    assert r.chain == [PAID] and r.current() == PAID and r.summary()["policy"] == "paid-direct"
    with pytest.raises(ValueError, match="paid"):
        ModelRouter([F1], None, policy="paid-direct")
    with pytest.raises(ValueError):
        ModelRouter([F1], PAID, policy="bogus")


def test_paid_direct_run_uses_only_paid_and_records_policy(tmp_path):
    rec = e15a.Recorder()
    d = _paid_run(tmp_path, rec, "pd", model_policy="paid-direct")
    assert rec.models and set(rec.models) == {PAID}
    raw = json.loads((d / "raw.json").read_text())
    assert raw["model_routing"]["policy"] == "paid-direct" and raw["model_routing"]["chain"] == [PAID]
    assert raw["spend"]["paid_calls"] > 0 and raw["spend"]["reserved_usd"] <= 0.5 + 1e-9
    assert json.loads((d / "manifest.json").read_text())["model_policy"] == "paid-direct"
    assert "paid-direct" in (d / "report.md").read_text(encoding="utf-8")


def test_paid_direct_still_needs_allow_paid_and_a_live_run():
    with pytest.raises(ValueError, match="allow-paid"):
        CoreSettings(model_policy="paid-direct", **TINY).validate()
    with pytest.raises(ValueError, match="live"):
        CoreSettings(model_policy="paid-direct", allow_paid=True, budget_usd=0.5, offline=True, **TINY).validate()
    assert main(["run", "--model-policy", "paid-direct", "--run-id", "x", "--runs-dir", "runs"]) == 2


def test_paid_direct_respects_the_per_run_cap(tmp_path):
    e15a.open_store(tmp_path)
    rec = e15a.Recorder()
    d = execute(CoreSettings(systems=("naive_llm",), allow_paid=True, budget_usd=0.0005, model_policy="paid-direct",
                             **TINY), base=tmp_path, run_id="cap", call=rec, matcher=e14.TextMatcher(),
                key_usage=Usage())
    raw = json.loads((d / "raw.json").read_text())
    assert raw["spend"]["reserved_usd"] <= 0.0005 + 1e-12 and raw["refusals"]


# ------------------------------------------------------------------ 5. sizing options

def _run_actions():
    return {a.dest: a for a in build_parser()._subparsers._group_actions[0].choices["run"]._actions}


def test_run_help_documents_sizing_and_policy_options():
    acts = _run_actions()
    for dest in ("pool_size", "n_bases", "relevant_fraction", "n_jds", "model_policy", "dry_run"):
        assert len(acts[dest].help or "") > 30, dest
    assert "paid-direct" in acts["model_policy"].help
    doc = (Path(__file__).parents[2] / "src/agentic_talent_screener/evaluation/EFFICIENCY.md").read_text(encoding="utf-8")
    for token in ("--pool-size", "--n-bases", "--relevant-fraction", "--n-jds", "--model-policy paid-direct", "--dry-run"):
        assert token in doc


def test_relevant_fraction_changes_the_base_split(tmp_path):
    base = dict(TINY, pool_size=80, n_bases=10)
    outs = {}
    for frac in (0.2, 0.8):
        d = execute(CoreSettings(systems=("keyword_ats",), relevant_fraction=frac, **base), base=tmp_path,
                    run_id=f"f{int(frac * 10)}")
        outs[frac] = json.loads((d / "raw.json").read_text())["settings"]["relevance_coverage"]
    assert outs[0.2]["bases_relevant"] == 2 and outs[0.8]["bases_relevant"] == 8
    with pytest.raises(ValueError):
        CoreSettings(relevant_fraction=1.5, **TINY).validate()


def test_dry_run_prints_a_plan_without_calls():
    s = CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, model_policy="paid-direct",
                     families=("A1",), pool_size=80, n_bases=30, n_jds=1, max_llm_calls=650)
    text = estimate_text(s)
    assert "30 twin bases (15 relevant, 15 not)" in text and "estimate, not a cap" in text and "$0.50 per run" in text
    small = estimate_text(CoreSettings(pool_size=20, n_bases=4, n_jds=1, families=("A1",)))
    assert "below --min-verdict-n" in small
