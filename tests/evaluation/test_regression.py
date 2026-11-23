"""Offline CI regression gate (architecture 4.7): a tiny deterministic suite whose key numbers are pinned.

Keyword, TF-IDF, BM25, the naive screener and the full pipeline (fake chat, text matcher) run on a 10-resume pool
with 2 twin bases over EVERY attack family (A1-A10 and A5b, core grid, 64 twins). ``baseline/results.json`` holds the pinned metrics; the run must match it (a strict pin: any change fails and needs a reviewed refresh).
To refresh after an intended change: ``UPDATE_REGRESSION_BASELINE=1 uv run pytest tests/evaluation/test_regression.py``
and review the diff. No network.
"""

import json
import os
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation import report
from agentic_talent_screener.evaluation.runner import execute

from test_e14 import FREE, FakeLLM, TextMatcher, tiny

BASELINE = Path(__file__).parent / "baseline"
PINNED_PREFIXES = ("ranking.", "robustness.", "detection.")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    d = execute(tiny(families=None, n_bases=2), base=tmp_path_factory.mktemp("reg"), run_id="reg", call=FakeLLM(), model=FREE,
                matcher=TextMatcher())
    return d, json.loads((d / "results.json").read_text())


def _pinned(r):
    return {k: {x: m[x] for x in ("value", "ci95", "n", "verdict", "direction", "target", "unit")}
            for k, m in r["metrics"].items() if k.startswith(PINNED_PREFIXES)}


def test_matches_pinned_baseline_within_tolerance(results, tmp_path):
    d, r = results
    if os.environ.get("UPDATE_REGRESSION_BASELINE"):
        BASELINE.mkdir(exist_ok=True)
        (BASELINE / "results.json").write_text(json.dumps({"run_id": "baseline", "metrics": _pinned(r)}, indent=1, sort_keys=True))
    diff = report.compare_runs(BASELINE, d, tolerance=1e-9)
    bad = [x for x in diff["rows"] if x["change"] != "unchanged" and x["change"] != "added"]
    assert not bad, report.compare_markdown(diff)


def test_key_numbers_are_pinned_in_code(results):
    """Readable anchors so a failure points at the number, not just a diff."""
    m = results[1]["metrics"]
    assert m["robustness.asr_all.keyword_ats"]["n"] == 64
    fams = {k.split(".family.")[1] for k in m if k.startswith("robustness.asr.full_engine.family.")}
    assert fams == {"A1", "A2", "A3", "A4", "A5", "A5b", "A6", "A7", "A8", "A9", "A10"}
    assert m["robustness.asr.keyword_ats"]["value"] == pytest.approx(PIN["asr_keyword"], abs=1e-9)
    assert m["robustness.asr.bm25"]["value"] == pytest.approx(PIN["asr_bm25"], abs=1e-9)
    assert m["robustness.asr_nonrelevant.full_engine"]["value"] == pytest.approx(PIN["asr_full_nonrel"], abs=1e-9)
    assert m["robustness.asr.full_engine"]["value"] == pytest.approx(PIN["asr_full"], abs=1e-9)
    assert m["detection.fpr.full_engine"]["value"] == pytest.approx(PIN["fpr_full"], abs=1e-9)
    assert m["robustness.asr.full_engine.family.A3"]["value"] == pytest.approx(PIN["asr_full_A3"], abs=1e-9)
    assert m["robustness.asr.full_engine.family.A4"]["value"] == pytest.approx(PIN["asr_full_A4"], abs=1e-9)
    assert m["ranking.ndcg10.bm25"]["value"] == pytest.approx(PIN["ndcg10_bm25"], abs=1e-9)
    assert m["ranking.ndcg10.full_engine"]["value"] == pytest.approx(PIN["ndcg10_full"], abs=1e-9)


def test_a_worse_run_fails_the_gate(results, tmp_path):
    d, r = results
    worse = tmp_path / "w"
    worse.mkdir()
    bad = json.loads(json.dumps(r))
    bad["metrics"]["robustness.asr.full_engine"]["value"] += 0.1
    bad["metrics"]["detection.fpr.full_engine"]["value"] = (bad["metrics"]["detection.fpr.full_engine"]["value"] or 0) + 0.1
    (worse / "results.json").write_text(json.dumps(bad))
    assert report.compare_runs(BASELINE, worse, tolerance=1e-9)["n_regressions"] >= 2


PIN = {"asr_keyword": 0.09375, "asr_bm25": 0.0, "asr_full": 0.0, "asr_full_nonrel": 0.0625, "fpr_full": 0.0, "ndcg10_bm25": 1.0, "ndcg10_full": 1.0,
       "asr_full_A3": 0.0, "asr_full_A4": 0.0}


_DET_SCRIPT = """
import json
from agentic_talent_screener.evaluation.datasets.synthetic import make_resume
from agentic_talent_screener.evaluation.metrics.ranking import bm25_scores, tie_aware_ndcg
docs = {f"d{i}": make_resume(i).to_text() for i in range(40)}
query = " ".join(sorted({s for i in range(40) for s, _ in make_resume(i).skills}))
scores = bm25_scores(query, docs)
rel = {k: (i * 7) % 6 for i, k in enumerate(sorted(docs))}
print(json.dumps([[k, repr(v)] for k, v in scores.items()] + [repr(tie_aware_ndcg(scores, rel, 10))]))
"""


def test_bm25_and_tie_aware_ndcg_do_not_depend_on_pythonhashseed():
    import subprocess
    import sys

    outs = []
    for seed in ("1", "4242", "98765"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", _DET_SCRIPT], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2]
