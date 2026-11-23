"""T66 (E15d): paid fallback reachable, provider outage is not a FAIL, compare by bare run id, paid-direct policy."""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest
import requests

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


class Resp:
    def __init__(self, status, body):
        self.status_code, self.body, self.headers = status, body, {}

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeHTTP:
    """Mocked OpenRouter behind the REAL ``llm.chat_json``: ``requests.post`` is replaced, nothing else.

    ``free_failure``: None, ("status", 429) or ("body", code): what every free-model request answers.
    ``usage``: "tokens" (honest token counts), "cost" (usage.cost), "none" (no block), "bad" (malformed),
    or a float (a usage.cost that lies: the key is charged the honest price)."""

    def __init__(self, rec, usage_meter, free_failure=None, usage="tokens"):
        self.rec, self.key, self.free_failure, self.usage = rec, usage_meter, free_failure, usage
        self.posts, self.models, self.paid_posts = 0, [], 0

    def post(self, url, json=None, headers=None, timeout=None):
        import json as _json

        self.posts += 1
        model = json["model"]
        self.models.append(model)
        if model != PAID and self.free_failure:
            kind, code = self.free_failure
            if kind == "status":
                return Resp(code, {"error": {"code": code}})
            return Resp(200, {"error": {"code": code, "message": "upstream"}})
        fmt = json["response_format"]["json_schema"]
        result = self.rec(json["messages"], schema=fmt["schema"], schema_name=fmt["name"], model=model)
        content = _json.dumps(result)
        prompt = sum(len(m["content"]) for m in json["messages"]) // 4 + 1
        done = len(content) // 4 + 1
        honest = prompt * 0.15e-6 + done * 0.47e-6
        if model == PAID:
            self.paid_posts += 1
            self.key.charge(honest)
        body = {"choices": [{"message": {"content": content}}]}
        if self.usage == "tokens":
            body["usage"] = {"prompt_tokens": prompt, "completion_tokens": done}
        elif self.usage == "cost":
            body["usage"] = {"cost": honest}
        elif self.usage == "bad":
            body["usage"] = {"cost": "free", "prompt_tokens": -3}
        elif isinstance(self.usage, float):
            body["usage"] = {"cost": self.usage}
        return Resp(200, body)


@pytest.fixture
def http(monkeypatch):
    """Patch the module ``llm.requests`` (what chat_json calls) and its retry sleep."""
    from agentic_talent_screener import llm

    def install(fake):
        monkeypatch.setattr(llm, "requests", types.SimpleNamespace(
            post=fake.post, RequestException=requests.RequestException, HTTPError=requests.HTTPError))
        monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
        return fake
    return install


def _live_run(tmp_path, fake, run_id, systems=("full_engine",), cap=0.5, **kw):
    e15a.open_store(tmp_path)
    cfg = dict(TINY, **kw)
    policy = "free-first" if fake.free_failure else "paid-direct"
    d = execute(CoreSettings(systems=systems, allow_paid=True, budget_usd=cap, model_policy=policy, **cfg),
                base=tmp_path, run_id=run_id, call=None, matcher=e14.TextMatcher(), key_usage=fake.key)
    return d, json.loads((d / "raw.json").read_text())


# ------------------------------------------------------------------ 1. the paid fallback is reachable

@pytest.mark.parametrize("failure", [("status", 429), ("body", 429), ("body", "429")])
def test_46_straight_429s_from_both_free_models_reach_the_paid_model(tmp_path, http, failure):
    fake = http(FakeHTTP(e15a.Recorder(), Usage(), free_failure=failure))
    d, raw = _live_run(tmp_path, fake, "fb-" + str(failure[0]) + str(failure[1]), systems=("naive_llm", "full_engine"))
    free_posts = [m for m in fake.models if m != PAID]
    assert fake.paid_posts > 0 and set(free_posts) == {F1, F2}
    assert len(free_posts) >= 6  # both free models tripped (3 consecutive 429s each), then paid for the rest
    assert [x["to"] for x in raw["model_routing"]["switches"]] == [F2, PAID]
    used = {m for per in raw["model_routing"]["calls_by_system"].values() for m in per}
    assert used == {PAID}  # shared across systems and calls: nobody went back to a free model
    assert raw["spend"]["paid_calls"] > 0 and raw["spend"]["reserved_usd"] <= 0.5 + 1e-9


@pytest.mark.parametrize("failure", [("body", 502), ("body", 400), ("body", 402), ("body", None), ("status", 500)])
def test_a_non_429_error_never_opens_paid_spend(tmp_path, http, failure):
    """R2: a 502, 400 or an unreadable code is not a rate limit. The run stops without a paid call."""
    fake = http(FakeHTTP(e15a.Recorder(), Usage(), free_failure=failure))
    d, raw = _live_run(tmp_path, fake, "no429", systems=("full_engine",))
    assert fake.paid_posts == 0 and PAID not in fake.models and raw["spend"]["paid_calls"] == 0
    assert raw["model_routing"]["switches"] == []


def test_error_code_is_read_from_the_real_chat_json_frame(http):
    """A rename inside llm.chat_json (the stopgap reads its local ``body``) breaks this test, not the match."""
    from agentic_talent_screener import llm

    class Key:
        def charge(self, v):
            pass

    for code, expect in ((429, True), ("429", True), (502, False), (400, False), (None, False)):
        http(FakeHTTP(e15a.Recorder(), Key(), free_failure=("body", code)))
        with pytest.raises(RuntimeError) as ei:
            llm.chat_json([{"role": "user", "content": "x"}], schema={}, schema_name="s", model=F1, max_tokens=10,
                          timeout=1)
        assert is_rate_limited(ei.value) is expect, code
    http(FakeHTTP(e15a.Recorder(), Key(), free_failure=("status", 429)))
    with pytest.raises(RuntimeError) as ei:
        llm.chat_json([{"role": "user", "content": "x"}], schema={}, schema_name="s", model=F1, max_tokens=10, timeout=1)
    assert is_rate_limited(ei.value)


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
    assert not is_rate_limited(RuntimeError("OpenRouter response did not contain chat content; top_level_keys=['error']"))
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
    assert out["failed"] > 0 and out["failed"] == out["total"] and out["answered"] == 0
    for key, m in res["metrics"].items():
        if m["system"] in ("full_engine", "naive_llm"):
            assert m["verdict"].startswith("not measurable (") and "provider outage:" in m["verdict"], key
            assert m["value"] is None
    for row in res["slide9"]:
        if row["layer"] in ("Matching & ranking", "Robustness", "Detection", "Efficiency", "Agent reliability"):
            assert row["verdict"].startswith("not measurable") and "provider outage" in row["verdict"], row["layer"]
    assert res["summary"]["fail"] == 0 and res["summary"]["pass"] == 0
    assert f"provider outage: {out['failed']}/{out['total']} calls failed, 0 answered" in (d / "report.md").read_text(encoding="utf-8")


def test_healthy_run_has_no_outage_flag(tmp_path):
    d = execute(CoreSettings(systems=("naive_llm", "full_engine"), **TINY), base=tmp_path, run_id="ok",
                call=e15a.Recorder(), matcher=e14.TextMatcher())
    raw = json.loads((d / "raw.json").read_text())
    assert raw["systems"]["full_engine"]["outage"] is None and raw["systems"]["naive_llm"]["outage"] is None


def test_outage_rule_needs_a_failed_majority_and_few_answers():
    from agentic_talent_screener.evaluation.runner import _outage

    def call(kind):
        return {"type": "call", "error_kind": kind}

    assert _outage([call("provider_error")] * 5, 30) == {"failed": 5, "total": 5, "answered": 0}
    assert _outage([call(None)] * 3 + [call("transport")] * 4, 30) == {"failed": 4, "total": 7, "answered": 3}
    assert _outage([call(None)] * 40 + [call("provider_error")] * 50, 30) is None  # enough answers
    assert _outage([call(None)] * 5, 30) is None  # no failures: plain n<min handles it
    assert _outage([call("budget_guard")] * 5, 30) is None  # refusals are not an outage


def test_one_transient_error_is_not_an_outage():
    from agentic_talent_screener.evaluation.runner import _outage

    records = [{"type": "call", "error_kind": None}] * 25 + [{"type": "call", "error_kind": "transport"}]
    assert _outage(records, 30) is None  # R3: 25 answered, 1 failed, the system's own fail verdicts must survive


def test_answered_but_wrong_output_is_not_an_outage():
    from agentic_talent_screener.evaluation.runner import _outage

    records = ([{"type": "call", "error_kind": None}] * 4 + [{"type": "call", "error_kind": "invalid_output"}] * 95
               + [{"type": "call", "error_kind": "provider_error"}])
    assert _outage(records, 30) is None  # R3: that is an agent-reliability result, not an outage


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
    (other / "bare-z").mkdir()  # a directory of that name in the cwd is never used
    with pytest.raises(ValueError, match="not in the eval store"):
        compare_runs("bare-z", "bare-a", store_root=tmp_path)


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


def test_dry_run_on_the_doc_example_prints_the_labelled_cap_basis():
    s = CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, model_policy="paid-direct",
                     families=("A1",), pool_size=100, n_bases=20, n_jds=1, max_llm_calls=500)
    text = estimate_text(s)
    for label in ("reserved worst case", "expected billed (estimate)", "cap $0.50 per run", "80 twins",
                  "FPR n = 100 clean resumes", "10 relevant, 10 not"):
        assert label in text, label
    doc = (Path(__file__).parents[2] / "src/agentic_talent_screener/evaluation/EFFICIENCY.md").read_text(encoding="utf-8")
    assert text in doc  # the doc shows the dry run's own output for the example command
    assert "--pool-size 100 --n-bases 20" in doc and "--families A1" in doc and "naive_llm" in doc


def test_dry_run_warns_when_the_system_cannot_finish_under_the_cap():
    s = CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.10, model_policy="paid-direct",
                     families=("A1",), pool_size=100, n_bases=20, n_jds=1, max_llm_calls=500)
    text = estimate_text(s)
    assert "it cannot finish" in text and "warning: full_engine cannot finish under $0.10" in text
    full_grid = estimate_text(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, pool_size=90,
                                           n_bases=30, n_jds=1, max_llm_calls=300))
    assert "cannot finish" in full_grid and "960 twins" in full_grid


def test_dry_run_worst_case_matches_what_the_meter_reserves():
    """The estimate uses the cap's own worst-case formula: about $0.015 per document (reviewed probe: 0.0154)."""
    text = estimate_text(CoreSettings(systems=("full_engine",), allow_paid=True, budget_usd=0.5, families=("A1",),
                                      pool_size=100, n_bases=20, n_jds=1))
    worst = float(text.split("reserved worst case (what the cap reserves before a call settles): $")[1].split(" ")[0])
    assert 180 * 0.012 < worst < 180 * 0.019


def test_twins_per_base_limits_and_rotates_the_cells():
    from agentic_talent_screener.evaluation.attacks.twins import generate_twins, grid
    from agentic_talent_screener.evaluation.datasets.synthetic import make_resume

    bases = [make_resume(i, 0) for i in range(8)]
    all_cells = generate_twins(bases, "core", 0, ["A1"])
    one = generate_twins(bases, "core", 0, ["A1"], per_base=1)
    assert len(all_cells) == 8 * len(grid("core", ["A1"])) and len(one) == 8
    assert len({(t.label.position, t.label.visibility) for t in one}) > 1  # rotates through the cells
    with pytest.raises(ValueError):
        generate_twins(bases, "core", 0, ["A1"], per_base=0)
    assert "30 twins" in estimate_text(CoreSettings(pool_size=60, n_bases=30, n_jds=1, families=("A1",),
                                                    twins_per_base=1, systems=("keyword_ats",)))


def test_twins_per_base_in_a_run(tmp_path):
    d = execute(CoreSettings(systems=("keyword_ats",), twins_per_base=1, **dict(TINY, n_bases=3)), base=tmp_path,
                run_id="tpb")
    raw = json.loads((d / "raw.json").read_text())
    assert raw["settings"]["n_twins"] == 3 and raw["settings"]["twins_per_base"] == 1
    assert _run_actions()["twins_per_base"].help


# ------------------------------------------------------------------ settlement of reservations (R1)

def test_settle_lowers_the_reservation_to_the_billed_cost():
    m = e15a.meter(Usage())
    worst = m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)
    assert m.reserved == pytest.approx(worst)
    m.settle([{"prompt_tokens": 1000, "completion_tokens": 500}])
    billed = 1000 * 0.15e-6 + 500 * 0.47e-6
    assert m.reserved == pytest.approx(billed) and billed < worst / 10
    assert m.settled == pytest.approx(billed) and m.worst_total == pytest.approx(worst) and m.settled_calls == 1
    m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)  # the next call reserves on top of the settled cost
    m.settle([{"cost": 0.002}])
    assert m.reserved == pytest.approx(billed + 0.002)
    fin = m.finish()
    assert fin["reserved_worst_usd"] == pytest.approx(2 * worst) and fin["settled_usd"] == pytest.approx(billed + 0.002)


@pytest.mark.parametrize("usages", [[], [None], [{}], [{"cost": "x"}], [{"cost": float("nan")}], [{"cost": -1}],
                                    [{"prompt_tokens": -1, "completion_tokens": 5}], [{"prompt_tokens": 5}],
                                    [{"cost": 0.001}, None], [{"cost": True}], ["usage"]])
def test_missing_or_malformed_usage_keeps_the_worst_case(usages):
    m = e15a.meter(Usage())
    worst = m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)
    m.settle(usages)
    assert m.reserved == pytest.approx(worst) and m.unsettled_calls == 1 and m.settled_calls == 0
    m.settle([{"cost": 0.0}])  # no call in flight: nothing happens
    assert m.reserved == pytest.approx(worst)


def test_a_lying_usage_block_is_bounded_by_the_key_counter():
    u = Usage()
    m = e15a.meter(u)
    m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)
    u.charge(0.40)  # the provider really billed 0.40
    m.settle([{"cost": 0.0000001}])  # usage claims almost nothing
    assert m.reserved < 1e-6
    assert m.counted() == pytest.approx(0.40 + m.held)  # max(key delta, reservations) + the headroom
    with pytest.raises(e15a.BudgetExceeded, match="per-run cap"):
        m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 120_000)  # 0.40 counted + a big worst case > 0.50


def test_the_phase_key_view_still_binds_when_usage_lies():
    u = Usage()
    m = e15a.meter(u, cap=0.50)
    m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 100)
    u.charge(1.45)  # real spend on the key far above what the usage block says
    m.settle([{"cost": 0.0}])
    with pytest.raises(e15a.BudgetExceeded):
        m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 100)
    m.run_cap = 2.0  # even with a larger run cap the phase key view refuses
    with pytest.raises(e15a.BudgetExceeded, match="phase cap"):
        m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 100_000)


def test_a_500_run_completes_many_more_calls_once_reservations_settle(tmp_path_factory, http):
    kw = dict(pool_size=30, n_bases=8, n_jds=1, max_llm_calls=100_000)
    out = {}
    for usage in ("tokens", "none"):
        tp = tmp_path_factory.mktemp(usage)
        fake = http(FakeHTTP(e15a.Recorder(), Usage(), usage=usage))
        d, raw = _live_run(tp, fake, "bud-" + usage, systems=("full_engine",), cap=0.5, **kw)
        out[usage] = (raw["spend"], raw["systems"]["full_engine"]["status"], raw["refusals"])
    settled, unsettled = out["tokens"][0], out["none"][0]
    assert unsettled["paid_calls"] < settled["paid_calls"]  # worst-case-only stops the run early
    assert "run_cap" in out["none"][2] and not out["tokens"][2]
    assert out["tokens"][1] == "ok" and settled["settled_calls"] == settled["paid_calls"]
    assert settled["reserved_usd"] < settled["reserved_worst_usd"] / 5 and settled["reserved_usd"] <= 0.5
    assert settled["measured_usd"] == pytest.approx(settled["settled_usd"], rel=0.05)


def test_the_meter_settles_from_the_http_response_usage_block(tmp_path, http):
    for usage in ("cost", "bad", 0.0):
        tp = tmp_path / str(usage)
        fake = http(FakeHTTP(e15a.Recorder(), Usage(), usage=usage))
        d, raw = _live_run(tp, fake, "u", systems=("naive_llm",), cap=0.5)
        sp = raw["spend"]
        if usage == "bad":
            assert sp["settled_calls"] == 0 and sp["unsettled_calls"] == sp["paid_calls"] > 0
        else:
            assert sp["settled_calls"] == sp["paid_calls"] > 0
        if usage == 0.0:  # a lying zero is bounded by the key's own counter
            assert sp["counted_usd"] >= sp["measured_usd"] > 0


# ------------------------------------------------------------------ round 3: dropped attempts, lag headroom, zero usage

class Timeouts(FakeHTTP):
    """Every paid call: a first attempt that is billed (0.00125) but times out, then a good second attempt."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._flip = False

    def post(self, url, json=None, headers=None, timeout=None):
        if json["model"] == PAID:
            self._flip = not self._flip
            if self._flip:
                self.key.charge(0.00125)
                raise requests.ConnectionError("read timed out")
        return super().post(url, json=json, headers=headers, timeout=timeout)


def test_a_timed_out_attempt_before_a_good_retry_keeps_the_worst_case(tmp_path, http):
    """F1: the billed-but-unanswered attempt has no usage block, so the call stays unsettled at its worst case."""
    u = Usage()
    fake = http(Timeouts(e15a.Recorder(), u, usage="cost"))
    d, raw = _live_run(tmp_path, fake, "to", pool_size=4, n_bases=1, max_llm_calls=10_000)
    sp = raw["spend"]
    assert sp["paid_calls"] > 0 and sp["settled_calls"] == 0 and sp["unsettled_calls"] == sp["paid_calls"]
    assert sp["reserved_usd"] >= u.true_total()  # counted covers what was billed (the probe: 0.0035 vs 0.0175)
    assert sp["settled_usd"] == 0.0


def test_a_raised_attempt_is_recorded_as_an_unsettled_entry():
    from agentic_talent_screener.evaluation.runner import _CountingRequests

    class Real:
        n = 0

        def post(self, *a, **k):
            Real.n += 1
            if Real.n == 1:
                raise requests.ConnectionError("x")
            return Resp(200, {"usage": {"cost": 0.001}})

    c = _CountingRequests(Real(), lambda: None)
    with pytest.raises(requests.ConnectionError):
        c.post("u")
    c.post("u")
    assert c.usages == [None, {"cost": 0.001}]
    assert e15a.B.billed_cost(c.usages) is None


class HonestLie(FakeHTTP):
    """The key is charged 0.0015 per paid post (below the worst case) while the usage block says almost nothing."""

    def post(self, url, json=None, headers=None, timeout=None):
        resp = super().post(url, json=json, headers=headers, timeout=timeout)
        if json["model"] == PAID:
            self.key.charge(0.0015)
        return resp


@pytest.mark.parametrize("lag", [0, 10, 100])
def test_a_lying_usage_block_with_a_lagging_key_cannot_pass_the_cap(tmp_path, http, lag):
    """F2: the reviewer's probe (pool 150). Before the headroom rule: 0.513 at lag 10 and 0.618 at lag 100."""
    u = Usage(lag=lag)
    fake = http(HonestLie(e15a.Recorder(), u, usage=1e-9))
    d, raw = _live_run(tmp_path, fake, f"h{lag}", pool_size=150, n_bases=8, max_llm_calls=10_000)
    assert u.true_total() <= 0.50 + 1e-9, (lag, u.true_total())
    assert raw["spend"]["headroom_held_usd"] > 0


def test_headroom_is_held_for_the_last_calls_only():
    m = e15a.meter(Usage())
    for _ in range(e15a.B.HOLDBACK_CALLS + 5):
        w = m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)
        m.settle([{"cost": 0.0001}])
    assert m.held == pytest.approx(e15a.B.HOLDBACK_CALLS * (w - 0.0001))  # older holds were released
    assert m.counted() == pytest.approx(m.reserved + m.held)


@pytest.mark.parametrize("usage", [{"cost": 0}, {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0},
                                   {"prompt_tokens": 0, "completion_tokens": 0}])
def test_a_zero_usage_block_never_settles_below_the_floor(usage):
    m = e15a.meter(Usage())
    m.authorize(PAID, e15a.MSGS, e15a.SCHEMA, 4096)
    m.settle([usage])
    assert m.reserved >= e15a.B.MIN_BILLED_PER_ATTEMPT_USD and m.settled_calls == 1


def test_zero_cost_with_tokens_settles_at_the_token_price():
    assert e15a.B.billed_cost([{"cost": 0, "prompt_tokens": 1000, "completion_tokens": 100}]) == pytest.approx(
        1000 * 0.15e-6 + 100 * 0.47e-6)
