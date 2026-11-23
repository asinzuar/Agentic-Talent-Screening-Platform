"""T37 (E5b, R47): per-document intake flag records from FullPipeline, wired into E7. Offline: fake chat."""

import pytest

from agentic_talent_screener.evaluation.attacks.twins import make_twin
from agentic_talent_screener.evaluation.datasets.synthetic import make_resume
from agentic_talent_screener.evaluation.metrics.detection import summarize_detection
from agentic_talent_screener.evaluation.metrics.robustness import TwinCase, run_paired
from agentic_talent_screener.evaluation.systems import (
    EvalCandidate, FlagFinding, FlagPolicy, FlagRecord, FullPipeline, KeywordATS, SystemOutput,
)
from agentic_talent_screener.ranking.sample_data import sample_requirement_profile

from test_e5 import FREE, GOOD, THIN, FakeLLM, TextMatcher, _cands
from test_e5_review import _cc, _docx_with_hidden

JD = sample_requirement_profile()
HIDDEN = "Kubernetes wizard HIDDENMARK"


def _pipe(tmp_path, tag="c", **kw):
    return FullPipeline(_cc(tmp_path, tag, call=FakeLLM()), matcher=TextMatcher(), model=FREE, **kw)


@pytest.fixture
def hidden_doc(tmp_path):
    p = tmp_path / "h.docx"
    _docx_with_hidden(p, HIDDEN)
    return EvalCandidate("h", path=p)


def test_clean_resume_has_a_ready_record_and_no_flag(tmp_path):
    out = _pipe(tmp_path).score(_cands(tmp_path, g=GOOD), JD)
    rec = out.records["g"]
    assert rec.status == "ready_to_score" and rec.risk == "low" and rec.findings == ()
    assert out.flags == {} and "g" in out.scores


def test_hidden_text_twin_is_flagged_at_l0_with_locations(tmp_path, hidden_doc):
    out = _pipe(tmp_path).score([hidden_doc], JD)
    rec = out.records["h"]
    l0 = [f for f in rec.findings if f.layer == "L0"]
    assert l0 and l0[0].code.endswith("NONVISIBLE_TEXT") and l0[0].family == "A3" and l0[0].locations
    assert set(l0[0].locations) <= set(rec.flagged_locations)
    assert rec.risk in ("medium", "high")
    # scored on the safe remainder, yet flagged: that is the "flagged" detection kind
    assert "h" in out.scores and out.flags["h"][0] == "L0"


def test_flag_policy_is_configurable(tmp_path, hidden_doc):
    base = _pipe(tmp_path, "a").score([hidden_doc], JD)
    off = _pipe(tmp_path, "b", flag_policy=FlagPolicy(min_finding_risk=None)).score([hidden_doc], JD)
    assert base.flags and off.flags == {}  # same document, the finding rule switched off
    assert off.records["h"].findings == base.records["h"].findings
    with pytest.raises(ValueError):
        FlagPolicy(min_finding_risk="severe")
    with pytest.raises(ValueError):
        FlagPolicy(statuses=("nope",))


def test_policy_layers_and_status_rules_on_records():
    f = FlagFinding("L1", "X", "low")
    low = FlagRecord("ready_to_score", "low", "L3_EVIDENCE_VERIFIER", "", (f,))
    assert not FlagPolicy().flagged(low) and FlagPolicy(min_finding_risk="low").layers(low) == ("L1",)
    rev = FlagRecord("review_required", "high", "L3_EVIDENCE_VERIFIER", "L1", (FlagFinding("L0", "Y", "high"),))
    assert FlagPolicy().layers(rev) == ("L0", "L1")
    assert FlagPolicy(statuses=()).layers(rev) == ("L0",)


def test_ablations_change_the_flags_as_documented(tmp_path, hidden_doc):
    layers = {}
    for ab in (None, "-L0", "-L1a", "-L1b", "-L3", "-L4"):
        out = _pipe(tmp_path, f"x{ab}", ablation=ab).score([hidden_doc], JD)
        layers[ab] = (out.flags or {}).get("h")
    assert layers[None] and layers[None][0] == "L0"
    assert layers["-L0"] is None  # no L0 findings: hidden text is not flagged
    for same in ("-L1a", "-L1b", "-L3", "-L4"):
        assert layers[same] == layers[None], same


def test_infrastructure_errors_stay_separate(tmp_path):
    from agentic_talent_screener.evaluation.systems import InfrastructureError
    offline = _cc(tmp_path, "empty", offline=True)
    with pytest.raises(InfrastructureError):
        FullPipeline(offline, matcher=TextMatcher(), model=FREE).score(_cands(tmp_path, g=GOOD), JD)


def test_baselines_expose_no_flags():
    out = KeywordATS().score([EvalCandidate("a", text="Python")], JD)
    assert out.flags is None and out.records == {}


def test_run_paired_makes_full_engine_fpr_and_tpr_measurable(tmp_path, hidden_doc):
    clean = _cands(tmp_path, g1=GOOD, g2=GOOD.replace("Jane", "Joan"), g3=THIN, h=GOOD)
    lab = make_twin(make_resume(0), "A3", "medium", "summary", "hidden", channel="hidden_run").label
    lab = lab.model_copy(update={"base_id": "h", "twin_id": "h-twin", "formats": ["docx"]})
    run = run_paired(_pipe(tmp_path), clean, [TwinCase(lab, hidden_doc)], JD)
    assert run.clean_flags_source == "output.flags" and run.twin_flags_source == "output.flags"
    assert run.clean_records["g1"].status == "ready_to_score"
    assert run.twin_records["h-twin"].findings and run.twin_records["h-twin"].findings[0].layer == "L0"
    o = run.outcomes[0]
    assert o.status == "scored" and o.detection.kind == "flagged" and o.detection.layers[0] == "L0"
    d = summarize_detection(run.outcomes, run.clean, system="full_engine", min_n=1,
                            clean_flags_source=run.clean_flags_source, twin_flags_source=run.twin_flags_source)
    fpr = d["clean"]["fpr"]
    # g3 (no verifiable fact) goes to manual review at L3: a real false positive under the default policy
    assert run.clean["g3"].kind == "manual_review" and run.clean["g3"].layer == "L3"
    assert fpr["n"] == 4 and fpr["k"] == 1 and "not measurable (clean flags" not in fpr["status"]
    assert d["tpr"]["overall"]["k"] == 1 and d["attribution"]["caught_by_layer"] == {"L0": 1}
    assert d["attribution"]["caught_at_expected_layer"]["k"] in (0, 1)
