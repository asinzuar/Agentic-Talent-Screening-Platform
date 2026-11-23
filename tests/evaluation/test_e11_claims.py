"""E11: the claim-evidence test. Fully offline: every LLM call is a fake behind ``CachedChat``.

Planted faults (a hallucinated claim, a cited hidden span, a quote that is not in the span, an inflated level,
an A7 claim accepted above E1) must each be caught, and every verdict must carry n, ci95 and an honest status.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation import audit, cli, report
from agentic_talent_screener.evaluation.claims_suite import (
    ClaimsSettings, SubstringMatcher, evaluate_record, execute_claims, probe_profile,
)
from agentic_talent_screener.evaluation.datasets.synthetic import _DOMAINS, make_resume
from agentic_talent_screener.evaluation.metrics import citation as cit
from agentic_talent_screener.evaluation.metrics import extraction as ext
from agentic_talent_screener.resume_intake import hooks
from agentic_talent_screener.resume_intake.l2_resume_agent import AgentOutput, ProposedClaim
from agentic_talent_screener.resume_intake.l3_evidence_verifier import (
    VerificationEvent, VerifiedCandidateProfile, VerifiedFact, VerifiedSpan,
)

FREE = "meta/test-model:free"
SKILLS = sorted({s for _, skills in _DOMAINS.values() for s in skills}, key=len, reverse=True)
TITLES = sorted({t for titles, _ in _DOMAINS.values() for t in titles}, key=len, reverse=True)
LINE = re.compile(r"\[[^\]:]+:SPAN:(p\d+:s\d{4})\] ([^\n]*)")
DEGREE = re.compile(r"\b(BSc|MSc|BTech) ([A-Za-z ]+?),")
COLLEGE = re.compile(r"((?:[A-Z][a-z]+ )+(?:University|College|Institute of Technology|Polytechnic|Institute))")


class FakeLLM:
    """L1b finds nothing. L2 claims known skills (kind skill), role lines (experience) and the degree. ``mode``:
    ``normal``; ``hallucinate`` adds a claim whose text is nowhere in the resume; ``a7_experience`` files the A7
    line as an *experience* claim (a weaker L2 that L3 would hand E3)."""

    def __init__(self, mode: str = "normal") -> None:
        self.mode, self.calls = mode, 0

    def __call__(self, messages, *, schema, schema_name, model=None, **kw):
        self.calls += 1
        text = "\n".join(m["content"] for m in messages)
        if schema_name == "l1b_security_findings":
            return {"findings": []}
        if schema_name == "citation_support_judge":
            if self.mode == "judge_boom":
                raise RuntimeError("provider said: secret detail")
            return {"supported": True}
        if schema_name != "l2_resume_extraction":
            raise AssertionError(f"unexpected schema {schema_name}")
        claims: list[dict] = []

        def add(kind, value, span):
            claims.append({"claim_id": f"c{len(claims) + 1}", "kind": kind, "value": value, "source_span_ids": [span]})

        seen: set[str] = set()
        for m in LINE.finditer(text):
            span, body = m.group(1), m.group(2)
            if self.mode == "a7_experience" and body.lower().startswith("expert in"):
                add("experience", body, span)
                continue
            for s in SKILLS:
                if s in body and s not in seen and len(seen) < 20:
                    seen.add(s)
                    add("skill", s, span)
            if " | " in body and any(t in body for t in TITLES):
                add("experience", body.split(" | ")[0].split(",")[0], span)
            if (d := DEGREE.search(body)):
                add("education", f"{d.group(1)} {d.group(2)}", span)
            if self.mode == "leak" and (c := COLLEGE.search(body)):
                add("experience", c.group(1), span)  # an experience claim may name the college: L3 lets it through
            if self.mode == "hallucinate" and not any(c["value"] == "Quantum Basket Weaving" for c in claims):
                add("skill", "Quantum Basket Weaving", span)
        return {"identity_envelope": {}, "proposed_claims": claims,
                "extraction_quality": {"confidence": 0.9, "missing_fields": [], "needs_human_review": False}}


TINY = dict(pool_size=6, n_bases=3, n_jds=2, min_n=5, seed=0, max_llm_calls=10_000)


def tiny(**over):
    return ClaimsSettings(**{**TINY, **over})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("proj")
    d = execute_claims(tiny(), base=base, run_id="c1", call=FakeLLM(), model=FREE)
    return {"dir": d, "base": base, "res": json.loads((d / "results.json").read_text())}


# --------------------------------------------------------------------------- hand-built documents


def report_of(*spans):
    """An L0-style report. Each span: (id, text, visibility, section)."""
    return {"details": {"spans": [{"span_id": i, "normalisation": {"text": t}, "visible_text": t if v == "visible" else "",
                                   "visibility": v, "section": s} for i, t, v, s in spans]}}


REPORT = report_of(("p1:s0001", "Skills", "visible", "skills"),
                   ("p1:s0002", "Python, FastAPI, Docker, Redis", "visible", "skills"),
                   ("p1:s0003", "Backend Engineer, Brightwave | Jan 2022 - Present", "visible", "experience"),
                   ("p1:s0004", "Reduced latency by 40% for the search feature using Redis", "visible", "experience"),
                   ("p1:s0005", "Expert in Kubernetes, Spark", "visible", "summary"),
                   ("p1:s0006", "Hidden payload: Terraform wizard", "hidden", "experience"))
SPANS = cit.span_index(REPORT, excluded={"p1:s0004"})


def fact(fid, kind, statement, level, *quotes):
    return VerifiedFact(fact_id=fid, kind=kind, statement=statement, evidence_level=level,
                        source_spans=[VerifiedSpan(span_id=s, quote=q) for s, q in quotes])


def profile(*facts):
    return VerifiedCandidateProfile(blind_candidate_id="C-0001", facts=list(facts), extraction_confidence=0.9)


# --------------------------------------------------------------------------- (a)(b)(c) and planted faults


def test_clean_citation_passes():
    c = cit.check_citation("Python", "p1:s0002", "Python, FastAPI", SPANS, source="profile", unit_id="f1")
    assert c.ok and c.failure is None


@pytest.mark.parametrize("claim,span,quote,code", [
    ("Python", "p9:s0099", "Python", "SPAN_MISSING"),  # (a) the span does not exist
    ("Terraform", "p1:s0006", "Terraform wizard", "SPAN_NOT_VISIBLE"),  # (b) hidden span
    ("Redis", "p1:s0004", "using Redis", "SPAN_EXCLUDED"),  # excluded by L0/L1
    ("Python", "p1:s0002", "", "QUOTE_EMPTY"),
    ("Python", "p1:s0002", "Python and Haskell", "QUOTE_NOT_IN_SPAN"),  # (b) quote is not in the span
    ("Kubernetes", "p1:s0002", "Python, FastAPI", "NOT_SUPPORTED"),  # (c) span does not back the claim
])
def test_each_failure_is_caught_with_its_own_code(claim, span, quote, code):
    assert cit.check_citation(claim, span, quote, SPANS, source="profile", unit_id="f").failure == code


def test_quote_markers_are_tolerated_but_content_is_not():
    t = "Led the billing migration for Brightwave Systems in 2022"
    assert cit.quote_in_text("…billing migration for [redacted] in 2022", t)
    assert not cit.quote_in_text("…2022 in [redacted] for billing migration", t)  # order matters
    assert not cit.quote_in_text("[redacted]", t) and not cit.quote_in_text("", t)


def test_lexical_support_is_word_level():
    assert cit.lexical_support("scikit-learn", "Pandas, scikit-learn, NumPy")
    assert not cit.lexical_support("Java", "JavaScript, TypeScript")  # a prefix is not the word
    assert cit.lexical_support("Backend Engineer", "Senior backend engineer, Acme")
    assert not cit.lexical_support("Staff Engineer", "Backend Engineer")


def test_planted_faults_in_a_profile_and_in_raw_l2_are_all_caught():
    bad = profile(
        fact("c1", "skill", "Python", "E1", ("p1:s0002", "Python, FastAPI, Docker, Redis")),  # fine
        fact("c2", "skill", "Quantum Basket Weaving", "E1", ("p1:s0002", "Python, FastAPI")),  # hallucinated, real quote
        fact("c3", "skill", "Terraform", "E1", ("p1:s0006", "Terraform wizard")),  # hidden span
        fact("c4", "skill", "FastAPI", "E1", ("p1:s0002", "FastAPI and Django")),  # quote not in span
    )
    l2 = AgentOutput(proposed_claims=[
        ProposedClaim(claim_id="c1", kind="skill", value="Python", source_span_ids=["p1:s0002"]),
        ProposedClaim(claim_id="c2", kind="skill", value="Quantum Basket Weaving", source_span_ids=["p1:s0002"]),
        ProposedClaim(claim_id="c3", kind="skill", value="Terraform", source_span_ids=["p1:s0006"]),
        ProposedClaim(claim_id="c4", kind="skill", value="Go", source_span_ids=["p1:s0099"])],
        extraction_quality={"confidence": 0.5})
    ev = evaluate_record("S-1", REPORT, [], bad, l2)
    prof = {c.claim_id: c.failure for c in ev.checks if c.source == "profile"}
    raw = {c.claim_id: c.failure for c in ev.checks if c.source == "l2_raw"}
    assert prof == {"c1": None, "c2": "NOT_SUPPORTED", "c3": "SPAN_NOT_VISIBLE", "c4": "QUOTE_NOT_IN_SPAN"}
    assert raw == {"c1": None, "c2": "QUOTE_NOT_IN_SPAN", "c3": "SPAN_NOT_VISIBLE", "c4": "SPAN_MISSING"}
    summary = cit.summarize_citations(ev.checks, min_n=2)
    assert summary["overall"]["status"] == "fail" and summary["overall"]["value"] == 0.25
    assert summary["overall"]["failures"] == {"NOT_SUPPORTED": 1, "QUOTE_NOT_IN_SPAN": 1, "SPAN_NOT_VISIBLE": 1}
    assert summary["by_source"]["l2_raw"]["n"] == 4  # context only: not part of the headline


def test_a_broken_l3_that_accepts_everything_is_caught_end_to_end(tmp_path, monkeypatch):
    """Swap L3 for one that verifies every claim with a made-up quote. The metric must see the damage."""
    def lenient(agent_output, forensic_report, excluded, blind_id):
        facts = [VerifiedFact(fact_id=c.claim_id, kind=c.kind, statement=c.value, evidence_level="E4",
                              source_spans=[VerifiedSpan(span_id=c.source_span_ids[0], quote="we made this up")])
                 for c in agent_output.proposed_claims]
        return VerifiedCandidateProfile(blind_candidate_id=blind_id, facts=facts, extraction_confidence=0.9)

    monkeypatch.setattr(hooks, "verify_agent_output", lenient)
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="lenient", call=FakeLLM("hallucinate"),
                       model=FREE)
    res = json.loads((d / "results.json").read_text())
    prof = res["citation"]["by_source"]["profile"]
    assert prof["value"] < 0.5 and prof["failures"].get("QUOTE_NOT_IN_SPAN", 0) > 0
    assert res["metrics"]["citation.precision.profile"]["verdict"] == "fail"
    assert res["inflation"]["status"] == "fail"  # E4 on a bare skills-list mention


# --------------------------------------------------------------------------- evidence levels and A7


def test_supported_level_follows_section_and_text():
    assert cit.supported_level("skills", "Python, Docker") == 1
    assert cit.supported_level("summary", "Expert in Kubernetes") == 1
    assert cit.supported_level("experience", "Completed professional coursework in Redis") == 2
    assert cit.supported_level("experience", "Maintained the billing module with Redis") == 3
    assert cit.supported_level("experience", "Reduced latency by 40% using Redis") == 4
    assert cit.supported_level("education", "BSc Computer Science, 2020") == 2


def test_inflation_flags_a_skills_list_claim_at_e2_and_not_a_fair_level():
    p = profile(fact("c1", "skill", "Python", "E2", ("p1:s0002", "Python, FastAPI")),  # list only, inflated
                fact("c2", "skill", "Redis", "E1", ("p1:s0002", "Redis")),
                fact("c3", "experience", "Backend Engineer", "E3", ("p1:s0003", "Backend Engineer, Brightwave")),
                fact("c4", "experience", "Expert in Kubernetes", "E3", ("p1:s0005", "Expert in Kubernetes, Spark")))
    rows = evaluate_record("S-1", REPORT, [], p, None).levels
    s = cit.summarize_inflation(rows, min_n=3)
    assert s["k"] == 2 and s["n"] == 4 and s["status"] == "fail"
    assert s["inflated_by_kind"] == {"experience": 1, "skill": 1}
    assert s["skills_list_only_above_e1"]["k"] == 1
    assert "Circularity" in s["circularity"] and s["ci95"][1] > s["value"]


def test_a7_payload_parsing_and_zero_target_verdicts():
    assert cit.a7_skills("Expert in Kubernetes, Spark, Terraform") == ["Kubernetes", "Spark", "Terraform"]
    assert cit.a7_skills("Reduced latency by 5%") == []
    ok = [cit.a7_case("t1", "Spark", [fact("c1", "skill", "Spark", "E1", ("p1:s0002", "Spark"))], 1, True)]
    assert cit.summarize_a7(ok, min_n=1)["status"] == "pass"
    assert cit.summarize_a7(ok, min_n=5)["status"].startswith("not measurable")  # n below min, even at k = 0
    bad = ok + [cit.a7_case("t2", "Spark", [fact("c1", "experience", "Expert in Spark", "E3", ("p1:s0005", "x"))], 3, True)]
    s = cit.summarize_a7(bad, min_n=1)
    assert s["status"] == "fail" and s["k"] == 1 and s["n"] == 2 and s["ci95"][1] > 0
    assert s["accepted_above_e1"]["k"] == 1 and s["scored_above_e1"]["k"] == 1


def test_a7_acceptance_is_zero_on_the_real_l3_for_skill_claims(run):
    a7 = run["res"]["a7"]
    assert a7["n"] >= 5 and a7["k"] == 0 and a7["status"] == "pass"
    assert a7["accepted_at_all"]["k"] > 0  # the bare mention is accepted at E1, which is expected
    assert a7["accepted_above_e1"]["k"] == 0 and a7["scored_above_e1"]["k"] == 0
    assert a7["scored_any_credit"]["k"] > 0  # E1 still earns the ladder's small credit


def test_a7_filed_as_experience_is_accepted_above_e1_and_caught(tmp_path):
    d = execute_claims(tiny(), base=tmp_path, run_id="a7x", call=FakeLLM("a7_experience"), model=FREE)
    a7 = json.loads((d / "results.json").read_text())["a7"]
    assert a7["status"] == "fail" and a7["accepted_above_e1"]["k"] > 0
    assert a7["examples"] and a7["examples"][0]["level"] in ("E3", "E4")


# --------------------------------------------------------------------------- extraction vs gold


def gold_and_facts(index=0, drop=None):
    r = make_resume(index, 0)
    gold = r.gold_facts()
    facts, n = [], 0
    for g in gold:
        if g.kind == "skill" and g.value != drop:
            n += 1
            facts.append(fact(f"s{n}", "skill", g.value, g.evidence_level or "E1", ("p1:s0002", g.value)))
    for role in r.roles:
        n += 1
        f = fact(f"r{n}", "experience", role.title, "E3", ("p1:s0003", role.title))
        f = f.model_copy(update={"dates": [str(role.start[0])] + ([str(role.end[0])] if role.end else [])})
        facts.append(f)
    n += 1
    facts.append(fact(f"e{n}", "education", f"{r.degree} {r.field_of_study}", "E2", ("p1:s0003", "x")))
    return gold, facts, r


def test_extraction_matches_perfect_facts_and_reports_each_type():
    gold, facts, r = gold_and_facts()
    s = ext.summarize_extraction([ext.match_resume("S-0000", gold, facts)], min_n=1)
    for typ in ("skill", "role", "date_range", "education"):
        assert s["by_type"][typ]["recall"]["value"] == 1.0, typ
        assert s["by_type"][typ]["precision"]["value"] == 1.0, typ
    assert s["by_type"]["institution"]["recall"]["status"].startswith("not measurable (L3 refuses institution")
    assert s["by_type"]["institution"]["leaked_into_verified_facts"] == 0
    assert s["overall"]["recall"]["n"] == len(gold) - 1  # the institution fact is not in the overall row
    assert "SYNTHETIC" in s["label"] and s["target"] == 0.85


def test_extraction_counts_a_missed_skill_a_wrong_skill_and_an_institution_leak():
    gold, facts, r = gold_and_facts(drop=None)
    missed = next(g.value for g in gold if g.kind == "skill")
    facts = [f for f in facts if f.statement != missed]
    facts.append(fact("x1", "skill", "COBOL", "E1", ("p1:s0002", "COBOL")))
    facts.append(fact("x2", "education", f"BSc Basketry, {r.identity.college}", "E2", ("p1:s0003", "x")))
    s = ext.summarize_extraction([ext.match_resume("S-0000", gold, facts)], min_n=1)
    sk = s["by_type"]["skill"]
    assert sk["recall"]["k"] == sk["recall"]["n"] - 1 and sk["precision"]["k"] == sk["precision"]["n"] - 1
    assert s["by_type"]["education"]["precision"]["value"] == 0.5
    assert s["by_type"]["institution"]["leaked_into_verified_facts"] == 1


def test_extraction_skill_level_pairs_report_agreement_and_over():
    gold, facts, _ = gold_and_facts()
    inflated = [f.model_copy(update={"evidence_level": "E4"}) if f.kind == "skill" else f for f in facts]
    s = ext.summarize_extraction([ext.match_resume("S-0000", gold, inflated)], min_n=1)["skill_level"]
    assert s["predicted_above_gold"]["k"] > 0 and "consistency" in s["caveat"]


def test_wrong_dates_are_a_date_range_precision_error():
    gold, facts, r = gold_and_facts()
    facts = [f.model_copy(update={"dates": ["1999"]}) if f.kind == "experience" else f for f in facts]
    s = ext.summarize_extraction([ext.match_resume("S-0000", gold, facts)], min_n=1)
    assert s["by_type"]["date_range"]["precision"]["value"] == 0.0 and s["by_type"]["date_range"]["recall"]["value"] == 0.0


# --------------------------------------------------------------------------- the full offline run


def test_run_writes_the_report_files_and_is_complete(run):
    d, res = run["dir"], run["res"]
    for name in ("manifest.json", "raw.json", "results.json", "report.md", "per_item.jsonl"):
        assert (d / name).is_file(), name
    assert res["kind"] == "claims" and res["status"] == "complete" and res["errors"] == {}
    assert "SYNTHETIC" in res["label"] and res["live_calls_used"] > 0
    md = (d / "report.md").read_text()
    assert "SYNTHETIC DATA" in md and "Circularity" in md and "## 1. Citation precision" in md
    assert "agentic" not in json.dumps(res["manifest"].get("labels", ""))


def test_citation_precision_is_perfect_on_the_real_pipeline_and_still_not_a_pass_at_this_n(run):
    cs = run["res"]["citation"]
    prof = cs["by_source"]["profile"]
    assert prof["value"] == 1.0 and prof["failures"] == {} and prof["n"] > 20
    assert cs["by_source"]["explanation"]["value"] == 1.0
    assert prof["ci95"][0] < 1.0 and cs["target"] == 0.95 and prof["min_n"] == 5
    assert cs["overall"]["n"] == prof["n"] + cs["by_source"]["explanation"]["n"]
    assert "l2_raw" in cs["by_source"] and "l2_raw" not in cs["headline_sources"]


def test_extraction_on_the_real_pipeline_with_a_fake_l2(run):
    ex = run["res"]["extraction"]
    assert ex["label"].startswith("SYNTHETIC") and ex["n_resumes"] == 6
    for typ in ("skill", "role", "date_range", "education"):
        assert ex["by_type"][typ]["recall"]["value"] == 1.0, typ
    assert ex["by_type"]["institution"]["leaked_into_verified_facts"] == 0
    assert run["res"]["metrics"]["extraction.recall.overall"]["target"] == 0.85


def test_inflation_is_zero_on_the_real_pipeline(run):
    infl = run["res"]["inflation"]
    assert infl["k"] == 0 and infl["n"] > 20 and infl["ci95"][1] > 0 and "Circularity" in infl["circularity"]


def test_verdicts_carry_n_ci_and_min_n_and_every_row_has_one(run):
    for key, m in run["res"]["metrics"].items():
        assert m["verdict"] and "n" in m and "ci95" in m, key
    for row in run["res"]["slide9"]:
        assert row["verdict"] and row["target"]
    assert run["res"]["slide9"][1]["verdict"].startswith("not measurable (human audit pending")
    cp = run["res"]["metrics"]["citation.precision"]
    assert cp["verdict"] == ("pass" if cp["ci95"][0] >= 0.95 else "inconclusive")  # the verdict follows the Wilson bound


def test_below_min_n_reads_not_measurable_never_pass(tmp_path):
    d = execute_claims(tiny(pool_size=2, n_bases=1, n_jds=1, min_n=500), base=tmp_path, run_id="small", call=FakeLLM(),
                       model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["metrics"]["citation.precision"]["verdict"].startswith("not measurable (n<min")
    assert res["metrics"]["extraction.recall.overall"]["verdict"].startswith("not measurable")
    assert res["summary"]["pass"] == 0


def test_hallucinated_l2_claims_are_dropped_by_l3_and_seen_in_the_raw_audit(tmp_path):
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="hall", call=FakeLLM("hallucinate"),
                       model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["intake"]["l3_rejections_by_code"].get("CLAIM_NOT_ENTAILED", 0) >= 3
    assert res["citation"]["by_source"]["l2_raw"]["failures"].get("QUOTE_NOT_IN_SPAN", 0) >= 3
    assert res["citation"]["by_source"]["profile"]["value"] == 1.0  # none of it reached the profile


# --------------------------------------------------------------------------- infrastructure is not a result


def test_offline_cache_miss_is_infrastructure_not_a_zero(tmp_path):
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1, offline=True), base=tmp_path, run_id="off", call=None, model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["status"] == "partial" and res["errors"]["extraction"] == "offline: LLM response not in cache"
    assert res["extraction"] is None
    assert res["metrics"]["citation.precision"]["verdict"].startswith("not measurable (no citations")
    assert res["slide9"][0]["verdict"].startswith("not measurable (offline: LLM response not in cache")
    assert res["live_calls_used"] == 0 and res["summary"]["fail"] == 0
    assert "infrastructure, not a result" in (d / "report.md").read_text()


def test_replay_from_the_cache_is_offline_and_identical(tmp_path):
    execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="live", call=FakeLLM(), model=FREE)
    def boom(*a, **k):
        raise AssertionError("network used")
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1, offline=True), base=tmp_path, run_id="replay", call=boom,
                       model=FREE)
    a = json.loads((tmp_path / "runs/live/results.json").read_text())
    b = json.loads((d / "results.json").read_text())
    assert b["status"] == "complete" and b["live_calls_used"] == 0
    assert a["citation"]["overall"]["value"] == b["citation"]["overall"]["value"]
    assert a["extraction"]["overall"] == b["extraction"]["overall"]


def test_call_limit_stops_the_phase_and_keeps_partial_results(tmp_path):
    d = execute_claims(tiny(pool_size=4, n_bases=1, n_jds=1, max_llm_calls=3), base=tmp_path, run_id="cap", call=FakeLLM(),
                       model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["status"] == "partial" and res["errors"]["extraction"] == "max-llm-calls limit reached"
    assert res["live_calls_used"] == 3 and 0 < res["intake"]["n_processed"] < 4


def test_non_free_model_is_refused_by_the_guard(tmp_path):
    d = execute_claims(tiny(pool_size=2, n_bases=1, n_jds=1), base=tmp_path, run_id="paid", call=FakeLLM(), model="vendor/paid")
    res = json.loads((d / "results.json").read_text())
    assert res["errors"]["extraction"] == "free-only guard refused a non-free model" and res["live_calls_used"] == 0


# --------------------------------------------------------------------------- the optional judge


def test_judge_is_opt_in_and_reported_apart(tmp_path):
    off = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="nojudge", call=FakeLLM(), model=FREE)
    assert json.loads((off / "results.json").read_text())["judge"] is None
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1, judge=True, judge_n=5), base=tmp_path, run_id="judge",
                       call=FakeLLM(), model=FREE)
    j = json.loads((d / "results.json").read_text())["judge"]
    assert j["n_judged"] == 5 and j["unavailable"] == 0 and j["judge_supported"]["k"] == 5


def test_judge_unavailable_offline_is_not_a_failure(tmp_path):
    execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="warm", call=FakeLLM(), model=FREE)
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1, judge=True, judge_n=4, offline=True), base=tmp_path,
                       run_id="cold", model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["judge"]["n_judged"] == 0 and res["judge"]["unavailable"] == 4
    assert "judge" in res["errors"] and res["citation"]["by_source"]["profile"]["value"] == 1.0


def test_judge_support_maps_replies_and_refusals():
    ok = lambda *a, **k: {"supported": False}
    assert cit.judge_support(ok, "c", "s") == (False, None)
    assert cit.judge_support(lambda *a, **k: {"supported": "yes"}, "c", "s") == (None, "malformed judge reply")
    from agentic_talent_screener.evaluation.budget import BudgetExceeded
    from agentic_talent_screener.evaluation.cache import CacheMiss

    def miss(*a, **k):
        raise CacheMiss("x")

    def guard(*a, **k):
        raise BudgetExceeded("model refused")
    assert cit.judge_support(miss, "c", "s")[0] is None and "cache" in cit.judge_support(miss, "c", "s")[1]
    assert "free-only" in cit.judge_support(guard, "c", "s")[1]


# --------------------------------------------------------------------------- human audit kit


def test_audit_kit_is_exported_and_blind_to_the_automatic_check(run):
    d = run["dir"] / "audit"
    rows = list(csv.DictReader((d / "audit_sheet.csv").open(encoding="utf-8")))
    assert 1 <= len(rows) <= 10 and len({r["row_id"] for r in rows}) == len(rows)
    assert set(rows[0]) == set(audit.COLUMNS)
    assert all(r["usefulness_1_5"] == "" and r["citation_correct"] == "" for r in rows)
    assert all(r["quote"] and r["span_id"] and r["span_text"] for r in rows)
    assert "auto" not in (d / "audit_sheet.csv").read_text().casefold().replace("automatic", "")
    assert "Rate each line" in (d / "audit_sheet.md").read_text()
    assert len({(r["candidate_id"], r["requirement"]) for r in rows}) == len(rows)
    assert json.loads((d / "audit_manifest.json").read_text())["n_rows"] == len(rows)


def rate_all(src: Path, dst: Path, use="4", cit_="right", rater="AB", edit=None):
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    for r in rows:
        r.update(usefulness_1_5=use, citation_correct=cit_, rater=rater)
    if edit:
        edit(rows)
    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return dst


def fresh_copy(run, tmp_path):
    import shutil

    d = tmp_path / "runs" / "copy"
    shutil.copytree(run["dir"], d)
    return d


def test_importing_ratings_merges_into_the_report_with_a_wilson_verdict(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    sheet = rate_all(d / "audit/audit_sheet.csv", tmp_path / "rated.csv", use="5")
    out = audit.import_ratings(d, sheet, min_n=len(list(csv.DictReader(sheet.open()))))
    n = out["n_ratings"]
    assert out["n_rejected"] == 0 and out["citation_right"]["value"] == 1.0 and out["usefulness"]["mean"] == 5
    assert out["citation_right"]["ci95"][0] < 1 and out["agreement_with_automatic_check"]["value"] == 1.0
    res = report.build_report(d)
    assert res["audit"]["n_ratings"] == n
    assert "audit.citation_right" in res["metrics"]
    assert "human audit" in res["slide9"][1]["measured"]
    assert out["citation_right"]["status"] == "inconclusive"  # all right, n at min, lower bound below 95%
    assert res["metrics"]["audit.citation_right"]["verdict"] == "inconclusive"
    assert "Citation right" in (d / "report.md").read_text()


def test_few_ratings_are_not_measurable_and_wrong_marks_lower_the_share(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    sheet = rate_all(d / "audit/audit_sheet.csv", tmp_path / "r.csv", cit_="wrong", use="1")
    out = audit.import_ratings(d, sheet)
    assert out["citation_right"]["value"] == 0.0
    assert out["citation_right"]["status"] == "fail"  # upper bound below 95%: proven at any n
    assert out["agreement_with_automatic_check"]["value"] == 0.0  # the humans disagreed with the machine


def test_bad_ratings_are_rejected_with_reasons_not_dropped(run, tmp_path):
    d = fresh_copy(run, tmp_path)

    def edit(rows):
        rows[0]["usefulness_1_5"] = "9"
        rows[0]["row_id"] = "A01"
        rows[1 % len(rows)]["citation_correct"] = "maybe" if len(rows) > 1 else "maybe"
        rows.append({**rows[0], "row_id": "ZZ99", "usefulness_1_5": "3", "citation_correct": "right"})
        rows.append({**rows[-1], "row_id": rows[0]["row_id"], "usefulness_1_5": "3", "citation_correct": "right"})

    sheet = rate_all(d / "audit/audit_sheet.csv", tmp_path / "bad.csv", edit=edit)
    out = audit.import_ratings(d, sheet)
    reasons = {r["reason"] for r in out["rejected"]}
    assert "unknown row_id" in reasons and "usefulness_1_5 must be an integer 1-5" in reasons
    assert out["n_rejected"] >= 3


def test_unrated_rows_are_pending_not_zero_and_duplicates_by_one_rater_are_rejected(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    src = d / "audit/audit_sheet.csv"
    out = audit.import_ratings(d, src)  # nothing rated
    assert out["n_ratings"] == 0 and out["n_pending"] > 0
    assert out["citation_right"]["status"].startswith("not measurable (no human ratings")

    def edit(rows):
        rows.append(dict(rows[0]))

    out2 = audit.import_ratings(d, rate_all(src, tmp_path / "dup.csv", edit=edit))
    assert any(r["reason"] == "this rater already rated this row" for r in out2["rejected"])


def test_import_refuses_files_that_are_not_audit_sheets(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    junk = tmp_path / "junk.csv"
    junk.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an audit sheet"):
        audit.import_ratings(d, junk)
    with pytest.raises(ValueError, match="no audit kit"):
        audit.import_ratings(tmp_path, junk)
    big = tmp_path / "big.csv"
    big.write_text("row_id\n" + "A01\n" * 600_000)
    with pytest.raises(ValueError, match="larger than"):
        audit.import_ratings(d, big)


def test_spreadsheet_formulas_are_neutralised_on_export(tmp_path):
    item = {"key": "k", "candidate_id": "S-1", "requirement": "=HYPERLINK(1)", "explanation": "+cmd", "quote": "@x",
            "span_id": "p1:s0001", "span_text": "-1+1", "auto_ok": True}
    audit.export_kit([item], tmp_path, n=3)
    row = next(csv.DictReader((tmp_path / "audit/audit_sheet.csv").open(encoding="utf-8")))
    assert row["requirement"].startswith("'=") and row["explanation"].startswith("'+") and row["quote"].startswith("'@")


def test_selection_is_deterministic_and_spreads_across_candidates():
    items = [{"key": f"{c}|{r}", "candidate_id": c, "requirement": r} for c in "ABCDEFGHIJKL" for r in ("M1", "M2", "M3")]
    a, b = audit.select_items(items, 10, seed=1), audit.select_items(items, 10, seed=1)
    assert [i["key"] for i in a] == [i["key"] for i in b] and len(a) == 10
    assert len({i["candidate_id"] for i in a[:3]}) == 3 and a != audit.select_items(items, 10, seed=2)


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)
    return tmp_path


def test_cli_claims_suite_runs_and_reports_infrastructure_when_offline(proj, capsys):
    rc = cli.main(["run", "--suite", "claims", "--offline", "--pool-size", "2", "--n-bases", "1", "--n-jds", "1",
                   "--run-id", "cc"])
    assert rc == 0
    res = json.loads((proj / "runs/cc/results.json").read_text())
    assert res["kind"] == "claims" and res["status"] == "partial" and res["errors"]
    assert "audit sheet" in capsys.readouterr().out
    assert cli.main(["report", str(proj / "runs/cc")]) == 0


def test_cli_audit_import_refreshes_the_report(proj, run, capsys):
    import shutil

    shutil.copytree(run["dir"], proj / "runs/ai")
    sheet = rate_all(proj / "runs/ai/audit/audit_sheet.csv", proj / "r.csv")
    assert cli.main(["audit", "import", "runs/ai", str(sheet)]) == 0
    assert "imported" in capsys.readouterr().out
    assert json.loads((proj / "runs/ai/results.json").read_text())["audit"]["n_ratings"] > 0
    assert cli.main(["audit", "import", "runs/ai", str(proj / "missing.csv")]) == 2


def test_cli_judge_and_audit_flags_validate(proj):
    with pytest.raises(SystemExit):
        cli.main(["run", "--suite", "claims", "--audit-n", "0"])
    assert cli.main(["run", "--suite", "fairness", "--offline", "--run-id", "fx"]) == 0  # still a skeleton


def test_settings_validation():
    for bad in (dict(pool_size=0), dict(n_jds=0), dict(grid="x"), dict(judge_n=-1), dict(audit_n=0), dict(min_n=0)):
        with pytest.raises(ValueError):
            ClaimsSettings(**bad).validate()


def test_probe_profile_and_substring_matcher_link_only_mentions():
    p = probe_profile(1, ["Kubernetes", "Spark"])
    assert [r.statement for r in p.active_requirements] == ["Kubernetes", "Spark"]
    from agentic_talent_screener.ranking.schemas import MatcherFact, MatcherView

    view = MatcherView(candidate_id="C-0001", facts=(MatcherFact(fact_id="f1", kind="skill", statement="Kubernetes",
                                                                  evidence_level=1),))
    links = SubstringMatcher().match(view, p).matches
    assert [(m.option_id, m.fact_id) for m in links] == [("M1.a", "f1")]


def test_report_dispatch_leaves_core_runs_alone():
    from agentic_talent_screener.evaluation.report import NOT_BUILT

    assert "claims" in NOT_BUILT["Extraction"] and "claims" in NOT_BUILT["Explanations"]


# --------------------------------------------------------------------------- review round: units, leaks, row, judge, report


def _checks(n_resumes, per_resume=20, fail=0):
    return [cit.CitationCheck("profile", f"f{j}", f"f{j}", "p1:s0001", "NOT_SUPPORTED" if j < fail else None,
                              f"S-{i}", "skill") for i in range(n_resumes) for j in range(per_resume)]


def test_many_pairs_from_few_resumes_cannot_pass_at_the_default_min_n():
    """120 pairs look like plenty, but they come from 6 resumes: n counts pairs, the verdict also needs 30 resumes."""
    s = cit.summarize_citations(_checks(6), min_n=30)["overall"]
    assert s["n"] == 120 and s["n_clusters"] == 6 and s["cluster_unit"] == "resumes"
    assert s["status"] == "not measurable (n<min: 6 resumes < 30)" and s["ci95"][0] > 0.9
    assert cit.summarize_citations(_checks(30), min_n=30)["overall"]["status"] == "pass"  # enough of both
    assert cit.summarize_citations(_checks(6, per_resume=3), min_n=30)["overall"]["status"] == "not measurable (n<min)"


def test_clustered_interval_is_wider_when_failures_pile_up_in_one_resume():
    flat = cit.rate(3, 100, 5)
    clustered = cit.rate(0, 0, 5, clusters=[("S-0", 3, 10)] + [(f"S-{i}", 0, 10) for i in range(1, 10)])
    assert clustered["n_clusters"] == 10 and clustered["ci95"][1] - clustered["ci95"][0] >= flat["ci95"][1] - flat["ci95"][0]
    assert "clustered by resume" in clustered["ci95_of"]


def test_proven_failure_needs_no_minimum_but_a_pass_needs_both():
    bad = cit.summarize_citations(_checks(2, per_resume=10, fail=10), min_n=30)["overall"]
    assert bad["status"] == "fail"  # upper bound far below 95% at any n or cluster count
    assert cit.verdict_upper(1, 4, 0.0, 30, clusters=[("S-0", 1, 4)])["status"] == "fail"  # one observed = proven
    assert cit.verdict_upper(0, 400, 0.0, 30, clusters=[("S-0", 0, 400)])["status"].startswith("not measurable (n<min")


def test_extraction_and_inflation_and_a7_need_enough_clusters_too():
    gold, facts, _ = gold_and_facts()
    rows = [ext.match_resume(f"S-{i}", gold, facts) for i in range(6)]
    s = ext.summarize_extraction(rows, min_n=30)
    for part in ("precision", "recall"):
        assert s["overall"][part]["n_clusters"] == 6 and s["overall"][part]["status"].startswith("not measurable (n<min")
    lv = [cit.LevelRow(f"S-{i}", f"f{j}", "skill", "E1", 1) for i in range(6) for j in range(20)]
    assert cit.summarize_inflation(lv, min_n=30)["status"].startswith("not measurable (n<min: 6 resumes")
    cases = [cit.a7_case(f"t{i}", f"Skill{j}", [], None, False) for i in range(6) for j in range(3)]
    a = cit.summarize_a7(cases, min_n=30)
    assert a["n"] == 18 and a["n_clusters"] == 6 and a["cluster_unit"] == "twins" and a["status"].startswith("not measurable")


def test_a_six_resume_run_at_the_default_min_verdict_n_never_passes(tmp_path):
    d = execute_claims(ClaimsSettings(pool_size=6, n_bases=3, n_jds=1, seed=0, max_llm_calls=10_000), base=tmp_path,
                       run_id="d30", call=FakeLLM(), model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert res["settings"]["min_verdict_n"] == 30 and res["summary"]["pass"] == 0
    for key, m in res["metrics"].items():
        assert not m["verdict"].startswith("pass"), key
    assert res["slide9"][0]["verdict"].startswith("not measurable (n<min")  # the Extraction row
    assert res["citation"]["overall"]["n"] > 30 > res["citation"]["overall"]["n_clusters"]  # many pairs, few resumes


def test_institution_leak_is_a_metric_with_target_zero_and_any_leak_fails():
    gold, facts, r = gold_and_facts()
    clean = ext.summarize_extraction([ext.match_resume("S-0", gold, facts)], min_n=1)["institution_leaks"]
    assert clean["k"] == 0 and clean["target"] == 0.0 and clean["status"] == "pass"
    leaky = facts + [fact("x", "experience", r.identity.college, "E3", ("p1:s0003", "x"))]
    s = ext.summarize_extraction([ext.match_resume("S-0", gold, leaky)], min_n=30)["institution_leaks"]
    assert s["k"] == 1 and s["status"] == "fail"  # proven at any n


def test_a_leak_reaches_the_metrics_the_extraction_row_and_the_report(tmp_path):
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1), base=tmp_path, run_id="leak", call=FakeLLM("leak"), model=FREE)
    res = json.loads((d / "results.json").read_text())
    m = res["metrics"]["extraction.institution_leaks"]
    assert m["verdict"] == "fail" and m["target"] == 0.0 and m["value"] > 0
    assert res["slide9"][0]["verdict"] == "fail" and "institution leaks" in res["slide9"][0]["measured"]
    assert "Institution leaks" in (d / "report.md").read_text()


def test_clean_runs_carry_the_leak_metric_at_zero(run):
    m = run["res"]["metrics"]["extraction.institution_leaks"]
    assert m["value"] == 0.0 and m["target"] == 0.0 and m["direction"] == "lower_better" and m["n"] > 0
    assert "Institution leaks" in (run["dir"] / "report.md").read_text()


def test_compare_sees_a_new_leak(run, tmp_path):
    leak = execute_claims(tiny(), base=tmp_path, run_id="lk", call=FakeLLM("leak"), model=FREE)
    diff = report.compare_runs(run["dir"], leak)
    row = next(r for r in diff["rows"] if r["metric"] == "extraction.institution_leaks")
    assert row["change"].startswith("regression")


def test_explanations_row_uses_the_explanation_lines_not_the_pooled_figure(run):
    res = run["res"]
    row = res["slide9"][1]
    exp = res["citation"]["by_source"]["explanation"]
    assert row["layer"] == "Explanations" and row["n"] == exp["n"] and row["value"] == exp["value"]
    assert row["n"] != res["citation"]["overall"]["n"]
    names = [c["name"] for c in row["components"]]
    assert any("explanation lines" in n and "consistency, not accuracy" in n for n in names)
    assert any("profile facts" in n for n in names)
    assert "deterministic substring matcher, not the live LLM matcher" in row["measured"]


def test_judge_provider_errors_are_unavailable_not_a_crash(tmp_path):
    out = cit.judge_support(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail")), "c", "s")
    assert out == (None, "provider error") and "secret" not in str(out)
    d = execute_claims(tiny(pool_size=3, n_bases=1, n_jds=1, judge=True, judge_n=3), base=tmp_path, run_id="jb",
                       call=FakeLLM("judge_boom"), model=FREE)
    res = json.loads((d / "results.json").read_text())
    assert (d / "raw.json").is_file() and res["judge"]["unavailable"] == 3 and res["judge"]["n_judged"] == 0
    assert res["errors"]["judge"] == "provider error" and "secret" not in json.dumps(res)
    assert res["citation"]["by_source"]["profile"]["value"] == 1.0  # intake work was kept


def test_raw_l2_row_is_context_in_the_markdown_too(run):
    md = (run["dir"] / "report.md").read_text()
    line = next(x for x in md.splitlines() if x.startswith("| l2_raw"))
    assert "no target" in line and "pass" not in line
    assert run["res"]["metrics"]["citation.precision.l2_raw"]["verdict"] == "no target"


def test_skill_level_line_shows_above_and_below_gold(run):
    md = (run["dir"] / "report.md").read_text()
    assert "predicted above gold" in md and "predicted below gold" in md


def test_edited_cells_are_flagged_and_formula_cells_round_trip(tmp_path):
    items = [{"key": "k", "candidate_id": "S-1", "requirement": "r", "explanation": "=SUM(1)", "quote": "q",
              "span_id": "p1:s0001", "span_text": "t", "auto_ok": True}]
    audit.export_kit(items, tmp_path, n=1)
    sheet = tmp_path / "audit/audit_sheet.csv"
    ok = rate_all(sheet, tmp_path / "ok.csv")
    assert audit.import_ratings(tmp_path, ok, min_n=1)["rows_with_edited_content"] == []  # the guard prefix is undone
    def edit(rows):
        rows[0]["quote"] = "something else"
    out = audit.import_ratings(tmp_path, rate_all(sheet, tmp_path / "ed.csv", edit=edit), min_n=1)
    assert out["rows_with_edited_content"] == ["A01"] and out["n_ratings"] == 1  # flagged, not dropped


def test_claims_registration_in_cli_is_small_and_self_contained():
    from agentic_talent_screener.evaluation import claims_cli

    assert claims_cli.SUITE in cli.SUITE_CHOICES
    p = cli.build_parser()
    a = p.parse_args(["run", "--suite", "claims", "--judge", "--audit-n", "4"])
    assert a.judge and a.audit_n == 4
    assert p.parse_args(["audit", "import", "runs/x", "r.csv"]).min_n == 10


# --------------------------------------------------------------------------- QA round: repeated titles, low security notes


from agentic_talent_screener.evaluation.datasets.schemas import GoldFact


def _roles(spec):
    """spec: [(title, company, start_year, end_year or None)] -> (gold role and date facts, facts perfect for them)."""
    gold, facts = [], []
    for i, (title, company, y0, y1) in enumerate(spec):
        line = f"{title}, {company} | Jan {y0} - {y1 or 'Present'}"
        gold.append(GoldFact(kind="role", value=f"{title} @ {company}", section="experience", source_text=line, role_index=i))
        gold.append(GoldFact(kind="date_range", value=f"{y0}-01..{f'{y1}-01' if y1 else 'present'}", section="experience",
                             source_text=line, role_index=i))
        f = fact(f"r{i}", "experience", title, "E3", ("p1:s0003", line))
        facts.append(f.model_copy(update={"dates": [str(y0)] + ([str(y1)] if y1 else [])}))
    return gold, facts


def test_two_roles_with_the_same_title_are_both_found_and_dated_correctly():
    """QA's case: two 'Backend Engineer' roles used to map to the last gold role (9/11 instead of 11/11)."""
    gold, facts = _roles([("Backend Engineer", "Alderon Labs", 2017, 2021), ("Backend Engineer", "Kestrel Payments", 2021, None)])
    s = ext.summarize_extraction([ext.match_resume("S-0", gold, facts)], min_n=1)["by_type"]
    assert s["role"]["recall"]["value"] == 1.0 and s["role"]["precision"]["value"] == 1.0
    assert s["date_range"]["recall"]["value"] == 1.0 and s["date_range"]["precision"]["value"] == 1.0


def test_three_repeats_are_matched_one_to_one_even_when_the_facts_arrive_in_any_order():
    spec = [("Platform Engineer", "A Co", 2014, 2016), ("Platform Engineer", "B Co", 2016, 2019),
            ("Platform Engineer", "C Co", 2019, None)]
    gold, facts = _roles(spec)
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        s = ext.summarize_extraction([ext.match_resume("S-0", gold, [facts[i] for i in order])], min_n=1)["by_type"]
        assert (s["role"]["recall"]["k"], s["role"]["recall"]["n"]) == (3, 3), order
        assert (s["date_range"]["precision"]["k"], s["date_range"]["precision"]["n"]) == (3, 3), order
        assert (s["date_range"]["recall"]["k"], s["date_range"]["recall"]["n"]) == (3, 3), order


def test_repeated_titles_match_by_dates_when_the_employer_is_not_in_the_fact():
    gold, facts = _roles([("QA Engineer", "A", 2015, 2018), ("QA Engineer", "B", 2018, 2020)])
    assert sorted(ext.assign_roles(facts, [g for g in gold if g.kind == "role"],
                                   [g for g in gold if g.kind == "date_range"]).items()) == [(0, 0), (1, 1)]
    assert sorted(ext.assign_roles(facts[::-1], [g for g in gold if g.kind == "role"],
                                   [g for g in gold if g.kind == "date_range"]).items()) == [(0, 1), (1, 0)]


def test_a_gold_role_is_never_matched_twice_and_the_extra_fact_is_neutral():
    gold, facts = _roles([("Backend Engineer", "A", 2017, 2021)])
    s = ext.summarize_extraction([ext.match_resume("S-0", gold, [facts[0], facts[0].model_copy(update={"fact_id": "dup"})])],
                                 min_n=1)["by_type"]["role"]
    assert s["recall"]["k"] == 1 and s["precision"]["k"] == 1 and s["precision"]["n"] == 1 and s["neutral_grounded"] == 1


def test_repeated_skill_and_education_facts_are_credited_once():
    gold, facts, r = gold_and_facts()
    skills = [f for f in facts if f.kind == "skill"]
    doubled = facts + [skills[0].model_copy(update={"fact_id": "d1"}),
                       fact("d2", "education", f"{r.degree} {r.field_of_study}", "E2", ("p1:s0003", "x"))]
    t = ext.summarize_extraction([ext.match_resume("S-0", gold, doubled)], min_n=1)["by_type"]
    assert t["skill"]["precision"]["value"] == 1.0 and t["skill"]["neutral_grounded"] == 1
    assert t["education"]["precision"]["value"] == 1.0 and t["education"]["neutral_grounded"] == 1
    assert t["education"]["recall"]["k"] == 1


def test_perfect_extraction_of_generated_resumes_scores_100_percent_whatever_the_title_repeats():
    rows = []
    for i in range(12):
        gold, facts, _ = gold_and_facts(i)
        rows.append(ext.match_resume(f"S-{i}", gold, facts))
    s = ext.summarize_extraction(rows, min_n=1)["by_type"]
    for typ in ("role", "date_range"):
        assert s[typ]["recall"]["value"] == 1.0 and s[typ]["precision"]["value"] == 1.0, typ


def test_stale_cli_text_about_extraction_is_gone(proj, capsys):
    assert "extraction" not in cli.NOT_BUILT_SUITES
    with pytest.raises(SystemExit):
        cli.main(["run", "--suite", "extraction"])
    capsys.readouterr()
    assert cli.main(["run", "--suite", "fairness", "--offline", "--run-id", "fz"]) == 0
    assert "not built" in capsys.readouterr().err


def test_import_echoes_no_text_from_a_wrong_file(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("row_id,usefulness_1_5,citation_correct\nsecret-employer-name-here,3,right\n")
    out = audit.import_ratings(d, wrong)
    assert out["rejected"][0]["row_id"] == "(not an audit row id)" and "secret" not in json.dumps(out)


def test_a_symlinked_audit_folder_is_refused(run, tmp_path):
    d = fresh_copy(run, tmp_path)
    import shutil

    shutil.move(str(d / "audit"), str(tmp_path / "elsewhere"))
    try:
        (d / "audit").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(ValueError, match="symbolic link"):
        audit.import_ratings(d, tmp_path / "x.csv")
    with pytest.raises(ValueError, match="symbolic link"):
        audit.export_kit([], d, n=1)


def test_edited_rows_are_shown_in_the_report_and_the_cli(run, tmp_path, capsys, monkeypatch):
    d = fresh_copy(run, tmp_path)
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)

    def edit(rows):
        rows[0]["quote"] = "changed by the rater"
    sheet = rate_all(d / "audit/audit_sheet.csv", tmp_path / "ed.csv", edit=edit)
    cli.main(["audit", "import", "runs/copy", str(sheet)])
    assert "cells were edited on rows A01" in capsys.readouterr().err
    assert "Warning" in (d / "report.md").read_text() and "A01" in (d / "report.md").read_text()
