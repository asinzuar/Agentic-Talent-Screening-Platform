import math

import pytest

from agentic_talent_screener.evaluation.metrics.ranking import (
    bm25_scores,
    compare_systems,
    kendall_tau,
    ndcg,
    per_jd_ndcg,
    rank_by_score,
    tie_aware_ndcg,
)
from agentic_talent_screener.evaluation.metrics.stats import (
    MAX_RESAMPLES,
    bootstrap_ci,
    lift_check,
    paired_bootstrap_lift_ci,
    paired_permutation_test,
)

REL = {"a": 3, "b": 2, "c": 0, "d": 1}


def test_ndcg_perfect_and_hand_computed():
    assert ndcg(["a", "b", "d", "c"], REL, 10) == pytest.approx(1.0)
    # ranking b, a: dcg = 3 + 7/log2(3); ideal = 7 + 3/log2(3) + 1/2
    got = 3 + 7 / math.log2(3)
    ideal = 7 + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg(["b", "a"], REL, 10) == pytest.approx(got / ideal)


def test_ndcg_k_truncates_and_k_larger_than_n():
    assert ndcg(["b", "a", "d", "c"], REL, 1) == pytest.approx(3 / 7)
    assert ndcg(["a", "b", "d", "c"], REL, 100) == pytest.approx(1.0)


def test_ndcg_all_zero_and_unknown_ids():
    assert ndcg(["a", "b"], {"a": 0, "b": 0}, 10) == 0.0
    assert ndcg(["zzz"], REL, 10) == 0.0
    with pytest.raises(ValueError):
        ndcg(["a"], REL, 0)


def test_ndcg_tied_relevance_is_order_invariant():
    rel = {"a": 2, "b": 2, "c": 0}
    assert ndcg(["a", "b", "c"], rel, 10) == ndcg(["b", "a", "c"], rel, 10) == pytest.approx(1.0)


def test_rank_by_score_ties_not_by_raw_id_and_seeded():
    sc = {f"x{i}": 1.0 for i in range(12)}
    sc["top"] = 2.0
    r0 = rank_by_score(sc, seed=0)
    assert r0[0] == "top" and r0 == rank_by_score(sc, seed=0)
    assert r0[1:] != sorted(r0[1:])  # not alphabetical
    assert rank_by_score(sc, seed=1) != r0


def test_ndcg_rejects_duplicates():
    with pytest.raises(ValueError):
        ndcg(["a", "a", "a"], {"a": 3, "b": 0}, 10)


def test_tie_aware_ndcg_hand_computed_and_id_independent():
    rel = {"a": 3, "b": 1, "c": 0, "d": 0}
    scores = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    mean_g = (7 + 1) / 4
    disc = sum(1 / math.log2(i + 2) for i in range(4))
    ideal = 7 + 1 / math.log2(3)
    assert tie_aware_ndcg(scores, rel, 10) == pytest.approx(mean_g * disc / ideal)
    ren = {"zz": 3, "yy": 1, "xx": 0, "ww": 0}
    assert tie_aware_ndcg({k: 0.0 for k in ren}, ren, 10) == pytest.approx(
        tie_aware_ndcg(scores, rel, 10)
    )
    s2 = {"a": 4, "b": 3, "c": 2, "d": 1}
    assert tie_aware_ndcg(s2, rel, 10) == pytest.approx(ndcg(["a", "b", "c", "d"], rel, 10))
    assert tie_aware_ndcg(s2, {"a": 0}, 10) == 0.0
    with pytest.raises(ValueError):
        tie_aware_ndcg(s2, rel, 0)


def test_lift_ci_repeatable_and_contains_point():
    a = [0.9, 0.8, 0.85, 0.95, 0.7]
    b = [0.6, 0.65, 0.7, 0.8, 0.5]
    r1 = paired_bootstrap_lift_ci(a, b, seed=4)
    assert r1 == paired_bootstrap_lift_ci(a, b, seed=4)
    point = (sum(a) / 5 - sum(b) / 5) / (sum(b) / 5) * 100
    assert r1["lift_ci95"][0] <= point <= r1["lift_ci95"][1]
    out = compare_systems({"j1": 0.9, "j2": 0.8}, {"j1": 0.6, "j2": 0.5})
    assert "lift_ci95" in out and out["baseline_zero"] is False
    assert compare_systems({"j1": 0.5}, {"j1": 0.0})["baseline_zero"] is True
    with pytest.raises(ValueError):
        paired_bootstrap_lift_ci([], [])


def test_resample_caps():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], n_resamples=MAX_RESAMPLES + 1)
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [0.0], n_resamples=MAX_RESAMPLES + 1)
    with pytest.raises(ValueError):
        paired_bootstrap_lift_ci([1.0], [1.0], n_resamples=0)


def test_lift_check_fp_tolerance():
    assert lift_check(0.69, 0.6)["meets_target"] is True


def test_reexported_kendall():
    assert kendall_tau(["a", "b", "d", "c"], REL) == pytest.approx(1.0)


def test_bm25_hand_computed():
    docs = {"d1": "python python sql", "d2": "java sql", "d3": "cooking"}
    s = bm25_scores("python", docs)
    idf = math.log(1 + 2.5 / 1.5)  # N=3, df=1
    norm = 1.2 * (0.25 + 0.75 * 3 / 2)  # avgdl = 2, dl = 3
    assert s["d1"] == pytest.approx(idf * 2 * 2.2 / (2 + norm))
    assert s["d2"] == 0.0 and s["d3"] == 0.0
    assert rank_by_score(s)[0] == "d1"


def test_bm25_edge_cases():
    assert bm25_scores("x", {}) == {}
    assert bm25_scores("", {"a": "foo"}) == {"a": 0.0}
    assert bm25_scores("foo", {"a": ""}) == {"a": 0.0}


def test_bootstrap_ci_constant_and_bounds():
    p, lo, hi = bootstrap_ci([0.5] * 5)
    assert p == lo == hi == 0.5
    p, lo, hi = bootstrap_ci([0.0, 1.0, 0.5, 0.7], seed=1)
    assert lo <= p <= hi
    assert bootstrap_ci([0.0, 1.0, 0.5], seed=3) == bootstrap_ci([0.0, 1.0, 0.5], seed=3)
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.5)


def test_permutation_exact_hand_computed():
    # d = [1,1,1]: only all-plus or all-minus reach |mean| = 1 -> 2/8
    r = paired_permutation_test([2, 2, 2], [1, 1, 1])
    assert r["p_value"] == pytest.approx(0.25)
    assert r["mean_diff"] == pytest.approx(1.0)
    assert paired_permutation_test([1, 2], [1, 2])["p_value"] == 1.0


def test_permutation_monte_carlo_and_validation():
    r = paired_permutation_test([1.0] * 20, [0.0] * 20, n_resamples=500, seed=0)
    assert r["p_value"] == pytest.approx(1 / 501)
    with pytest.raises(ValueError):
        paired_permutation_test([1], [1, 2])
    with pytest.raises(ValueError):
        paired_permutation_test([], [])


def test_lift_check():
    assert lift_check(0.8, 0.6)["lift_pct"] == pytest.approx(33.3333, abs=1e-3)
    assert lift_check(0.8, 0.6)["meets_target"] is True
    assert lift_check(0.62, 0.6)["meets_target"] is False
    assert lift_check(0.5, 0.0)["lift_pct"] == 100.0


def test_per_jd_and_compare():
    rels = {"j1": REL, "j2": REL}
    sys_r = {"j1": ["a", "b", "d", "c"], "j2": ["a", "b", "d", "c"]}
    base_r = {"j1": ["c", "d", "b", "a"], "j2": ["b", "a", "d", "c"], "j3": ["a"]}
    s = per_jd_ndcg(sys_r, rels)
    b = per_jd_ndcg(base_r, rels)
    assert set(b) == {"j1", "j2"} and s["j1"] == pytest.approx(1.0)
    out = compare_systems(s, b)
    assert out["n_jds"] == 2 and out["meets_target"] is True
    assert out["p_value"] == pytest.approx(0.5)  # exact, n=2, both diffs positive
    with pytest.raises(ValueError):
        compare_systems({"x": 1.0}, {"y": 1.0})
