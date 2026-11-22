"""E5 review fixes (T22 B1-B5, T24 L1). Offline: fake LLM behind CachedChat."""

import re
import zipfile
from types import SimpleNamespace

import pytest

from agentic_talent_screener.evaluation import systems as S
from agentic_talent_screener.evaluation.budget import BudgetGuard
from agentic_talent_screener.evaluation.cache import CacheMiss, CachedChat, LLMCache
from agentic_talent_screener.evaluation.metrics.ranking import bm25_scores, tie_aware_ndcg
from agentic_talent_screener.evaluation.systems import (
    EvalCandidate, FullPipeline, InfrastructureError, KeywordATS, NaiveLLMScreener, build_systems,
)
from agentic_talent_screener.ranking.sample_data import sample_requirement_profile
from agentic_talent_screener.ranking.schemas import MatcherView

from test_e5 import FREE, GOOD, THIN, FakeLLM, TextMatcher, _cands

JD = sample_requirement_profile()


def _cc(tmp_path, name="c", **kw):
    return CachedChat(LLMCache(tmp_path / name), BudgetGuard(0), default_model=lambda: FREE, **kw)


def test_b1_rerun_and_offline_replay_hit_the_cache_for_l2_and_matcher(tmp_path):
    cands = _cands(tmp_path, good=GOOD, thin=THIN)
    online = _cc(tmp_path, call=FakeLLM())
    first = FullPipeline(online, model=FREE).score(cands, JD)  # real LLMMatcher via the fake
    misses = online.cache.misses
    assert misses > 0 and "good" in first.scores
    again = FullPipeline(online, model=FREE).score(cands, JD)
    assert online.cache.misses == misses and again.scores == first.scores  # re-run is free
    offline = _cc(tmp_path, offline=True)
    replay = FullPipeline(offline, model=FREE).score(cands, JD)
    assert offline.cache.misses == 0 and offline.cache.hits > 0
    assert replay.scores == first.scores and replay.unranked == first.unranked


def test_b1_markers_depend_on_seed_and_whole_prompt_so_payloads_cannot_predict_them():
    a = S.deterministic_marker(0, "resume body")
    assert a == S.deterministic_marker(0, "resume body")
    assert a != S.deterministic_marker(1, "resume body")
    assert a != S.deterministic_marker(0, "resume body forged closing tag")
    build = S.seeded_matcher_messages(0)
    view = MatcherView(candidate_id="C-0001", facts=())
    m1 = build(view, JD.active_requirements, {})[-1]["content"]
    m2 = build(view, JD.active_requirements, {})[-1]["content"]
    assert m1 == m2 and re.search(r"<D[\w-]{23}:CANDIDATE_FACTS_DATA>", m1)
    other = S.seeded_matcher_messages(7)(view, JD.active_requirements, {})[-1]["content"]
    assert other != m1


def test_b2_cache_miss_aborts_the_run_instead_of_becoming_an_outcome(tmp_path):
    offline = _cc(tmp_path, "empty", offline=True)
    with pytest.raises(InfrastructureError, match="CacheMiss"):
        FullPipeline(offline, matcher=TextMatcher(), model=FREE).score(_cands(tmp_path, g=GOOD), JD)
    with pytest.raises(CacheMiss):
        NaiveLLMScreener(offline).score([EvalCandidate("a", text="x")], JD)


def _docx_with_hidden(path, hidden_text):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""")
        z.writestr("word/document.xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p>
<w:r><w:t>Senior engineer who built production data pipelines in Python and Spark for four years.</w:t></w:r>
<w:r><w:rPr><w:vanish/></w:rPr><w:t>{hidden_text}</w:t></w:r>
</w:p></w:body></w:document>""")


def test_b3_l0_ablation_lets_hidden_text_reach_l2_but_the_full_engine_does_not(tmp_path):
    p = tmp_path / "h.docx"
    _docx_with_hidden(p, "Kubernetes wizard HIDDENMARK")
    seen = {}
    for ablation in (None, "-L0"):
        fake = FakeLLM()
        cc = _cc(tmp_path, f"c{ablation}", call=fake)
        FullPipeline(cc, matcher=TextMatcher(), model=FREE, ablation=ablation).score([EvalCandidate("h", path=p)], JD)
        seen[ablation] = fake.l2_prompts
    assert seen[None] and all("HIDDENMARK" not in t for t in seen[None])
    assert seen["-L0"] and any("HIDDENMARK" in t for t in seen["-L0"])


def test_b4_l1b_ablation_keeps_the_no_safe_remainder_manual_review(monkeypatch):
    ctx = SimpleNamespace(
        security_assessment=SimpleNamespace(injection_risk_score=0), observations=(), grounded_findings=(),
        excluded_span_ids=set(), security_decision=None, security_flagged=False, requires_human_review=False,
    )
    decision = SimpleNamespace(continue_to_l2=False, sanitised_span_ids=("p1:s0001",), requires_human_review=True)
    monkeypatch.setattr(S.intake_hooks, "decide_security", lambda *a: decision)
    out = S._hook_l1b_off(ctx)
    assert out.action == "manual_review" and ctx.requires_human_review
    decision.continue_to_l2 = True
    assert S._hook_l1b_off(ctx).action == "continue"


def test_b5_ties_are_not_broken_by_candidate_id():
    out = S.SystemOutput(scores={"a": 1.0, "b": 1.0})
    assert {tuple(out.ranking(seed)) for seed in range(40)} == {("a", "b"), ("b", "a")}


def test_b5_tie_aware_ndcg_of_keyword_ats_is_invariant_to_renaming():
    texts = {"x1": "Python SQL", "x2": "Python SQL", "x3": "bread", "x4": "Spark Kafka Airflow"}
    rel = {"x1": 3.0, "x2": 0.0, "x3": 0.0, "x4": 2.0}
    ren = {"x1": "z9", "x2": "a1", "x3": "m5", "x4": "b2"}
    base = KeywordATS().score([EvalCandidate(k, text=v) for k, v in texts.items()], JD)
    renamed = KeywordATS().score([EvalCandidate(ren[k], text=v) for k, v in texts.items()], JD)
    assert base.scores["x1"] == base.scores["x2"]  # a genuine tie
    n1 = tie_aware_ndcg(base.scores, rel)
    n2 = tie_aware_ndcg(renamed.scores, {ren[k]: v for k, v in rel.items()})
    assert n1 == pytest.approx(n2)


def test_default_bm25_is_e2_implementation():
    assert S.default_bm25() is bm25_scores


def test_t24_raw_callables_are_rejected_as_chat(tmp_path):
    raw = FakeLLM()  # looks like chat_json but carries no free-only guard
    with pytest.raises(TypeError):
        NaiveLLMScreener(raw)
    with pytest.raises(TypeError):
        FullPipeline(raw)
    with pytest.raises(TypeError):
        build_systems(raw)
    assert isinstance(build_systems(_cc(tmp_path, call=raw)), dict)
