"""E12: traces, efficiency metrics. Offline; CachedChat(call=fake)."""

import json

import pytest

from agentic_talent_screener.evaluation.budget import BudgetExceeded, BudgetGuard
from agentic_talent_screener.evaluation.cache import CacheMiss, CachedChat, LLMCache
from agentic_talent_screener.evaluation.metrics.efficiency import PriceTable, percentile, summarize, summarize_all, wilson
from agentic_talent_screener.evaluation.systems import EvalCandidate, InfrastructureError, KeywordATS, build_systems
from agentic_talent_screener.evaluation.tracing import (
    TraceSink, TracedChat, TracedSystem, check_schema, error_kind, load_traces, ref, trace_systems,
)
from agentic_talent_screener.ranking.sample_data import sample_requirement_profile

FREE = "meta/test-model:free"
JD = sample_requirement_profile()
SCHEMA = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"], "additionalProperties": False}
SECRET = "TOP-SECRET-RESUME-TEXT jane@example.com sk-or-KEY123"


def make(tmp_path, call, offline=False):
    cache = LLMCache(tmp_path / "cache")
    inner = CachedChat(cache, BudgetGuard(0.0), offline=offline, call=call, default_model=lambda: FREE)
    sink = TraceSink("runs/t", base=tmp_path)
    sink.system = "t"
    return TracedChat(inner, sink), sink


def ask(chat, text="hello", name="site_a"):
    return chat([{"role": "user", "content": text}], schema=SCHEMA, schema_name=name)


def test_call_record_fields_and_no_text(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"score": 3})
    ask(chat, SECRET)
    ask(chat, SECRET)  # hit
    raw = (tmp_path / "runs" / "t" / "traces.jsonl").read_text(encoding="utf-8")
    assert "TOP-SECRET" not in raw and "jane@" not in raw and "sk-or" not in raw
    a, b = load_traces(tmp_path / "runs" / "t")
    assert (a["cache"], b["cache"]) == ("miss", "hit")
    assert a["call_site"] == "site_a" and a["model"] == FREE and a["attempts"] == 1 and b["attempts"] == 0
    assert a["schema_valid"] is True and a["error_kind"] is None and a["latency_ms"] >= 0
    assert a["est_prompt_tokens"] > 0 and a["tokens_basis"] == "estimated_from_chars"


def test_errors_classified_and_reraised_without_message(tmp_path):
    def boom(*a, **k):
        raise ValueError("secret prompt echo " + SECRET)

    chat, sink = make(tmp_path, boom)
    with pytest.raises(ValueError):
        ask(chat)
    rec = sink.records[0]
    assert rec["error_kind"] == "invalid_output" and rec["schema_valid"] is False
    assert "secret" not in json.dumps(sink.records)


def test_infrastructure_failures_are_not_schema_failures(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"score": 1}, offline=True)
    with pytest.raises(CacheMiss):
        ask(chat)
    with pytest.raises(BudgetExceeded):
        chat([{"role": "user", "content": "x"}], schema=SCHEMA, schema_name="s", model="paid/model")
    assert [r["error_kind"] for r in sink.records] == ["cache_miss", "budget_guard"]
    assert all(r["schema_valid"] is None for r in sink.records)
    s = summarize(sink.records, system="t")
    assert s["infrastructure_failures"] == 2 and s["schema_valid"]["n"] == 0
    assert s["verdicts"]["schema_valid_ge_99pct"]["status"] == "not measurable"


def test_schema_invalid_output_flagged(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"wrong": 1})
    ask(chat)
    assert sink.records[0]["schema_valid"] is False and sink.records[0]["error_kind"] is None


def test_check_schema_cases():
    assert check_schema({"score": 1.5}, SCHEMA)
    assert not check_schema({"score": "x"}, SCHEMA)
    assert not check_schema({"score": True}, SCHEMA)
    assert not check_schema({"score": 1, "extra": 2}, SCHEMA)
    assert not check_schema([1], SCHEMA)


def test_error_kind_mapping():
    assert error_kind(TimeoutError()) == "timeout"
    assert error_kind(ConnectionError()) == "transport"
    assert error_kind(RuntimeError()) == "provider_error"
    assert error_kind(KeyError()) == "other"


def test_attempts_counts_underlying_invocations(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"score": 1})
    ask(chat)
    assert sink.records[0]["attempts"] == 1
    assert summarize(sink.records, system="t")["retry_rate"]["value"] is None


def test_traced_chat_is_cachedchat_and_safe_dir(tmp_path):
    chat, _ = make(tmp_path, lambda *a, **k: {"score": 1})
    assert isinstance(chat, CachedChat)
    with pytest.raises(ValueError):
        TraceSink("../outside", base=tmp_path)


def test_naive_system_per_resume_records(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"score": 7})
    systems = trace_systems(build_systems(chat), sink)
    cands = [EvalCandidate("c1", text="python sql"), EvalCandidate("c2", text="baking")]
    out = systems["naive_llm"].score(cands, JD)
    assert set(out.scores) == {"c1", "c2"}
    resumes = [r for r in sink.records if r["type"] == "resume"]
    assert len(resumes) == 2 and all(r["timing_basis"] == "exact" and r["calls"] == 1 for r in resumes)
    assert all(r["system"] == "naive_llm" for r in resumes)
    assert resumes[0]["stage_ms"] and '"c1"' not in json.dumps(sink.records)


def test_pooled_system_pool_mean_and_error_path(tmp_path):
    _, sink = make(tmp_path, lambda *a, **k: {"score": 1})
    out = TracedSystem(KeywordATS(), sink).score([EvalCandidate("a", text="python"), EvalCandidate("b", text="sql")], JD)
    assert len(out.scores) + len(out.unranked) == 2
    assert [r["timing_basis"] for r in sink.records if r["type"] == "resume"] == ["pool_mean"] * 2

    class Bad:
        name = "bad"

        def score(self, c, jd):
            raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        TracedSystem(Bad(), sink).score([EvalCandidate("a", text="t")], JD)
    last = sink.records[-1]
    assert last["status"] == "error" and last["error_kind"] == "provider_error"


def test_per_candidate_error_recorded(tmp_path):
    _, sink = make(tmp_path, lambda *a, **k: {})

    class Bad:
        name = "bad"

        def score(self, c, jd):
            raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        TracedSystem(Bad(), sink, per_candidate=True).score([EvalCandidate("a", text="t")], JD)
    assert sink.records[-1]["error_kind"] == "transport" and sink.resume_ref is None


def test_empty_pool(tmp_path):
    _, sink = make(tmp_path, lambda *a, **k: {})
    assert TracedSystem(KeywordATS(), sink).score([], JD).scores == {}


def _call(site, ms, hit=False, valid=True, err=None, attempts=1, pt=1000, ct=500, system="s"):
    return {"type": "call", "system": system, "call_site": site, "latency_ms": ms, "cache": "hit" if hit else "miss",
            "schema_valid": valid, "error_kind": err, "attempts": attempts,
            "est_prompt_tokens": pt, "est_completion_tokens": ct}


def _resume(ms, basis="exact", status="ranked", hits=0, err=None, system="s"):
    return {"type": "resume", "system": system, "wall_ms": ms, "timing_basis": basis, "status": status,
            "cache_hits": hits, "live_calls": 1 - min(hits, 1), "error_kind": err}


def test_summary_metrics_and_verdicts():
    recs = [_call("a", 100), _call("a", 200, attempts=2), _call("b", 300, valid=False), _call("b", 50, hit=True),
            _resume(1000), _resume(2000), _resume(21000)]
    s = summarize(recs, PriceTable(usd_per_m_input=1.0, usd_per_m_output=2.0, inr_per_usd=100.0), system="s", min_n=1)
    lat = s["latency_s_per_resume"]
    assert lat["p50"] == 2.0 and lat["max"] == 21.0 and lat["p95"] == pytest.approx(19.1) and lat["n_qualified"] == 3
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "fail"
    assert s["schema_valid"]["value"] == pytest.approx(2 / 3)  # live calls only: 100, 200, 300
    assert s["verdicts"]["schema_valid_ge_99pct"]["status"] == "fail"
    assert s["cache_hit_rate"]["value"] == 0.25
    m = s["cost"]["modelled_not_measured"]
    assert s["cost"]["measured_usd"] == 0.0 and m["label"] == "modelled, not measured"
    assert m["usd_per_resume"] == pytest.approx(0.008 / 3) and m["inr_per_resume"] == pytest.approx(0.008 / 3 * 100)
    assert s["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "pass"
    assert s["verdicts"]["tool_call_correctness"]["status"] == "n/a"
    assert summarize(recs, system="other")["n_calls"] == 0
    with pytest.raises(ValueError):
        summarize(recs, system="")


def test_r1_pool_mean_latency_is_not_measurable():
    """Reviewer repro: one 30 s resume and nine 1 s resumes, pooled, gave ten 3.9 s means and a pass."""
    recs = [_resume(3900, basis="pool_mean") for _ in range(10)]
    s = summarize(recs, system="s")
    v = s["verdicts"]["latency_max_lt_20s"]
    assert v["status"] == "not measurable" and "max" not in s["latency_s_per_resume"]
    assert s["latency_s_per_resume"]["n_qualified"] == 0 and s["latency_s_per_resume"]["n_pool_mean"] == 10


def test_r1_mostly_replay_latency_is_not_measurable():
    """Reviewer repro: one live resume plus 99 replays got a real verdict."""
    recs = [_resume(900)] + [_resume(2, hits=3) for _ in range(99)]
    s = summarize(recs, system="s")
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "not measurable"
    assert s["latency_s_per_resume"]["n_qualified"] == 1 and s["latency_s_per_resume"]["n_with_cache_replay"] == 99


def test_r1_timeout_resume_fails_latency():
    recs = [_resume(1000), _resume(1000), _resume(60000, status="error", err="timeout")]
    s = summarize(recs, system="s")
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "fail"
    assert s["latency_s_per_resume"]["n_errored_excluded"] == 1


def test_r1_qualified_count_reported_when_judged():
    recs = [_resume(1000), _resume(2000), _resume(5, hits=1)]
    v = summarize(recs, system="s")["verdicts"]["latency_max_lt_20s"]
    assert v["status"] == "not measurable (n<min)" and v["n_qualified"] == 2 and v["n_resumes"] == 3
    v = summarize(recs, system="s", min_n=2)["verdicts"]["latency_max_lt_20s"]
    assert v["status"] == "pass" and v["n"] == 2 and v["ci95"] is not None


def _sv(k, n, min_n=30):
    recs = [_call("a", 1, valid=(i < k)) for i in range(n)]
    return summarize(recs, system="s", min_n=min_n)["verdicts"]["schema_valid_ge_99pct"]


def test_r8_schema_valid_n1_is_not_a_pass():
    v = _sv(1, 1)
    assert v["status"] == "not measurable (n<min)" and v["n"] == 1
    assert v["ci95"][0] == pytest.approx(0.2065, abs=1e-3) and v["ci95"][1] == 1.0


def test_r8_schema_valid_below_at_min_and_proven_fail():
    assert _sv(29, 29)["status"] == "not measurable (n<min)"
    assert _sv(30, 30)["status"] == "inconclusive"  # n at min, but the CI still spans the 99% target
    assert _sv(400, 400)["status"] == "pass"  # Wilson lower bound >= 0.99
    # proven fail below min: 0 of 5 valid, Wilson upper bound < 0.99
    v = _sv(0, 5)
    assert v["status"] == "fail" and v["n"] == 5 and v["ci95"][1] < 0.99


def test_r8_latency_min_n_and_proven_fail_below_min():
    few = summarize([_resume(1000)] * 29, system="s")["verdicts"]["latency_max_lt_20s"]
    assert few["status"] == "not measurable (n<min)" and few["n"] == 29
    ok = summarize([_resume(1000)] * 30, system="s")["verdicts"]["latency_max_lt_20s"]
    assert ok["status"] == "pass" and ok["n"] == 30 and ok["ci95"][0] > 0.85
    slow = summarize([_resume(25000)], system="s")["verdicts"]["latency_max_lt_20s"]
    assert slow["status"] == "fail" and slow["n"] == 1


def test_r8_cost_min_n_and_proven_fail_below_min():
    cheap = [_call("a", 1, pt=1000, ct=100) for _ in range(3)] + [_resume(1) for _ in range(3)]
    v = summarize(cheap, system="s")["verdicts"]["modelled_cost_lt_1_inr"]
    assert v["status"] == "not measurable (n<min)" and v["n"] == 3 and len(v["ci95"]) == 2
    many = [_call("a", 1, pt=1000, ct=100) for _ in range(30)] + [_resume(1) for _ in range(30)]
    assert summarize(many, system="s")["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "pass"
    dear = [_call("a", 1, pt=10_000_000, ct=0)] + [_resume(1)]  # about INR 85 per resume, n=1
    assert summarize(dear, system="s")["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "fail"


def test_r8_min_n_configurable_in_evalconfig():
    from agentic_talent_screener.evaluation.config import EvalConfig

    assert EvalConfig().min_verdict_n == 30
    assert EvalConfig(min_verdict_n=5).min_verdict_n == 5
    with pytest.raises(ValueError):
        EvalConfig(min_verdict_n=0)


def test_r2_cost_is_per_system_not_diluted():
    """Reviewer repro: 5 systems on 100 resumes; pooled denominator turned INR 0.90 into 0.18."""
    # 100 LLM resumes at 0.9 INR each (usd 0.9/85 each); 4 other systems with 100 resumes and no calls
    usd_each = 0.9 / 85.0
    pt = int(usd_each * 1e6 / 0.10)
    recs = [_call("a", 1, pt=pt, ct=0, system="full") for _ in range(100)]
    for n in ("full", "bm25", "tfidf", "ats", "naive"):
        recs += [_resume(1, system=n) for _ in range(100)]
    per = summarize_all(recs)
    assert set(per) == {"full", "bm25", "tfidf", "ats", "naive"}
    full = per["full"]["cost"]["modelled_not_measured"]["inr_per_resume"]
    assert full == pytest.approx(0.9, rel=0.01)
    assert per["bm25"]["cost"]["modelled_not_measured"]["inr_per_resume"] == 0
    # judged per system: a system at 1.2 INR fails even though a pooled average would pass
    heavy = [_call("a", 1, pt=int(1.2 / 85 * 1e6 / 0.10), ct=0, system="x") for _ in range(100)]
    heavy += [_resume(1, system="x") for _ in range(100)]
    assert summarize_all(heavy + recs)["x"]["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "fail"


def test_r3_replay_only_schema_valid_is_not_measured():
    """Reviewer repro: 1 live call plus 99 hits passed at 100 percent; replay-only now reads not measurable."""
    s = summarize([_call("a", 1, hit=True) for _ in range(99)], system="s")
    v = s["verdicts"]["schema_valid_ge_99pct"]
    assert v["status"] == "not measurable" and "replay only" in v["note"]
    assert s["schema_valid_replayed_biased"]["n"] == 99 and "biased" in s["schema_valid_replayed_biased"]["basis"]
    mixed = summarize([_call("a", 1)] + [_call("a", 1, hit=True) for _ in range(99)], system="s")
    assert mixed["schema_valid"]["n"] == 1  # the verdict rests on the single live call


def test_r4_transport_failures_are_not_schema_failures(tmp_path):
    def down(*a, **k):
        raise ConnectionError("net")

    chat, sink = make(tmp_path, down)
    with pytest.raises(ConnectionError):
        ask(chat)
    rec = sink.records[0]
    assert rec["error_kind"] == "transport" and rec["schema_valid"] is None
    recs = sink.records + [_call("a", 1, system="t") for _ in range(100)]
    s = summarize(recs, system="t")
    assert s["schema_valid"]["value"] == 1.0 and s["schema_valid"]["n"] == 100
    assert s["failure_kinds"] == {"transport": 1} and s["terminal_failure_rate"]["n"] == 101


def test_r5_retry_rate_not_measurable_until_c1():
    s = summarize([_call("a", 1, attempts=2)], system="s")
    assert s["retry_rate"]["value"] is None and "C1" in s["retry_rate"]["status"]


def test_r6_infrastructure_error_kind_on_resume_record(tmp_path):
    """FullPipeline raises InfrastructureError (a RuntimeError); it must not be labelled provider_error."""
    assert error_kind(InfrastructureError("x")) == "infrastructure"
    p = tmp_path / "g.txt"
    p.write_text("Jane Roe\nSenior engineer\nSkills: Python, SQL, Spark, Airflow, Kafka\n"
                 "Built production data pipelines in Python and Spark for four years.\n", encoding="utf-8")
    chat, sink = make(tmp_path, lambda *a, **k: {}, offline=True)
    with pytest.raises(InfrastructureError):
        trace_systems(build_systems(chat, model=FREE), sink)["full_engine"].score([EvalCandidate("g", path=p)], JD)
    resume = [r for r in sink.records if r["type"] == "resume"][-1]
    assert resume["error_kind"] == "infrastructure" and resume["status"] == "error"
    calls = [r for r in sink.records if r["type"] == "call"]
    assert calls and all(c["error_kind"] == "cache_miss" and c["est_prompt_tokens"] == 0 for c in calls)
    s = summarize(sink.records, system="full_engine")
    assert s["infrastructure_failures"] == len(calls)
    assert s["cost"]["modelled_not_measured"]["est_prompt_tokens_total"] == 0


def test_t32_salted_ref_and_runs_only(tmp_path):
    chat1, sink1 = make(tmp_path, lambda *a, **k: {"score": 1})
    sink2 = TraceSink("runs/u", base=tmp_path)
    assert ref("S-0000", sink1.salt) != ref("S-0000", sink2.salt)
    assert sink1.salt not in sink1.path.read_text(encoding="utf-8")
    (tmp_path / "data" / "eval").mkdir(parents=True)
    with pytest.raises(ValueError):
        TraceSink("data/eval/x", base=tmp_path)


def test_sink_truncates_existing_trace_and_replaces_symlink(tmp_path):
    d = tmp_path / "runs" / "z"
    d.mkdir(parents=True)
    (d / "traces.jsonl").write_text("stale\n", encoding="utf-8")
    TraceSink("runs/z", base=tmp_path)
    assert (d / "traces.jsonl").read_text(encoding="utf-8") == ""
    target = tmp_path / "victim.txt"
    target.write_text("keep", encoding="utf-8")
    (d / "traces.jsonl").unlink()
    try:
        (d / "traces.jsonl").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    TraceSink("runs/z", base=tmp_path)
    assert target.read_text(encoding="utf-8") == "keep"


def test_resolved_model_recorded(tmp_path):
    chat, sink = make(tmp_path, lambda *a, **k: {"score": 1})
    ask(chat)
    assert sink.records[0]["model"] == FREE


def test_failure_kinds_counted():
    s = summarize([_call("a", 1, valid=None, err="transport")] * 2, system="s")
    assert s["failure_kinds"] == {"transport": 2} and s["terminal_failure_rate"]["value"] == 1.0


def test_price_table_validation_and_helpers():
    with pytest.raises(ValueError):
        PriceTable(inr_per_usd=float("nan"))
    with pytest.raises(ValueError):
        PriceTable(usd_per_m_input=-1)
    with pytest.raises(ValueError):
        percentile([], 50)
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(99, 100) == pytest.approx((0.9455, 0.9982), abs=1e-3)
    assert wilson(0, 100)[1] == pytest.approx(0.0370, abs=1e-3)
