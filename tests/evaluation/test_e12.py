"""E12: traces, efficiency metrics. Offline; CachedChat(call=fake)."""

import json

import pytest

from agentic_talent_screener.evaluation.budget import BudgetExceeded, BudgetGuard
from agentic_talent_screener.evaluation.cache import CacheMiss, CachedChat, LLMCache
from agentic_talent_screener.evaluation.metrics.efficiency import PriceTable, percentile, summarize, wilson
from agentic_talent_screener.evaluation.systems import EvalCandidate, KeywordATS, build_systems
from agentic_talent_screener.evaluation.tracing import (
    TraceSink, TracedChat, TracedSystem, check_schema, error_kind, load_traces, trace_systems,
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
    assert a["call_site"] == "site_a" and a["model"] == "default" and a["attempts"] == 1 and b["attempts"] == 0
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
    s = summarize(sink.records)
    assert s["infrastructure_failures"] == 2 and s["schema_valid"]["n"] == 0
    assert s["verdicts"]["schema_valid_ge_99pct"]["status"] == "no data"


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
    assert summarize(sink.records)["retry_rate_boundary"]["value"] == 0


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


def _call(site, ms, hit=False, valid=True, err=None, attempts=1, pt=1000, ct=500):
    return {"type": "call", "system": "s", "call_site": site, "latency_ms": ms, "cache": "hit" if hit else "miss",
            "schema_valid": valid, "error_kind": err, "attempts": attempts,
            "est_prompt_tokens": pt, "est_completion_tokens": ct}


def _resume(ms):
    return {"type": "resume", "system": "s", "wall_ms": ms, "timing_basis": "exact", "status": "ranked"}


def test_summary_metrics_and_verdicts():
    recs = [_call("a", 100), _call("a", 200, attempts=2), _call("b", 300, valid=False), _call("b", 50, hit=True),
            _resume(1000), _resume(2000), _resume(21000)]
    s = summarize(recs, PriceTable(usd_per_m_input=1.0, usd_per_m_output=2.0, inr_per_usd=100.0))
    lat = s["latency_s_per_resume"]
    assert lat["p50"] == 2.0 and lat["max"] == 21.0 and lat["p95"] == pytest.approx(19.1)
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "fail"
    assert s["schema_valid"]["value"] == 0.75 and s["verdicts"]["schema_valid_ge_99pct"]["status"] == "fail"
    assert s["retry_rate_boundary"]["value"] == 0.25 and s["cache_hit_rate"]["value"] == 0.25
    m = s["cost"]["modelled_not_measured"]
    assert s["cost"]["measured_usd"] == 0.0 and m["label"] == "modelled, not measured"
    assert m["usd_per_resume"] == pytest.approx(0.008 / 3) and m["inr_per_resume"] == pytest.approx(0.008 / 3 * 100)
    assert s["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "pass"
    assert s["verdicts"]["tool_call_correctness"]["status"] == "n/a"
    assert summarize(recs, system="other")["n_calls"] == 0


def test_cost_target_fail_and_latency_pass():
    recs = [_call("a", 100, pt=10_000_000, ct=0), _resume(1000)]
    s = summarize(recs)  # defaults: $0.10/M in -> $1 per resume = 85 INR
    assert s["verdicts"]["modelled_cost_lt_1_inr"]["status"] == "fail"
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "pass"


def test_cache_replay_latency_no_data():
    recs = [_call("a", 1, hit=True, attempts=0), _resume(5)]
    s = summarize(recs)
    assert s["verdicts"]["latency_max_lt_20s"]["status"] == "no data"
    assert "not representative" in s["latency_basis"]


def test_failure_kinds_counted():
    s = summarize([_call("a", 1, valid=False, err="transport")] * 2)
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
    lo, hi = wilson(99, 100)
    assert 0.9 < lo < 0.99 < hi <= 1.0
