"""Ranking metrics: nDCG@k, BM25 baseline, per-JD aggregation, lift vs baseline.

Graded relevance in, ranked IDs out. Reuses the existing harness functions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from agentic_talent_screener.ranking.eval_harness import (  # noqa: F401  (re-exported)
    cosine_scores,
    kendall_tau,
    keyword_ats_scores,
    ndcg_at_k,
    spearman_rho,
)

from .stats import bootstrap_ci, lift_check, paired_permutation_test

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def ndcg(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """nDCG@k (default 10), graded relevance, exponential gain. k must be >= 1."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return ndcg_at_k(ranked_ids, relevance, k)


def rank_by_score(scores: Mapping[str, float]) -> list[str]:
    """Order IDs by score descending; ties broken by ID so results are deterministic."""
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _tokenize(text: str) -> list[str]:
    return [t for t in (x.strip(".-") for x in _TOKEN.findall(text.lower())) if t]


def bm25_scores(
    query: str, docs: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75
) -> dict[str, float]:
    """Okapi BM25 of ``query`` against each doc. IDF = ln(1 + (N - df + 0.5) / (df + 0.5))."""
    if not docs:
        return {}
    toks = {i: _tokenize(t) for i, t in docs.items()}
    n = len(toks)
    avgdl = sum(len(t) for t in toks.values()) / n
    df: Counter[str] = Counter()
    for t in toks.values():
        df.update(set(t))
    q_terms = set(_tokenize(query))
    out: dict[str, float] = {}
    for i, t in toks.items():
        tf = Counter(t)
        dl = len(t)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            norm = k1 * (1 - b + b * (dl / avgdl if avgdl else 0.0))
            s += idf * f * (k1 + 1) / (f + norm)
        out[i] = s
    return out


def per_jd_ndcg(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Mapping[str, float]],
    k: int = 10,
) -> dict[str, float]:
    """nDCG@k for every JD id present in both mappings (sorted by JD id)."""
    return {jd: ndcg(rankings[jd], relevance[jd], k) for jd in sorted(rankings) if jd in relevance}


def compare_systems(
    system: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    target_pct: float = 15.0,
    seed: int = 0,
    n_resamples: int = 10000,
) -> dict[str, object]:
    """Paired comparison of per-JD scores: means, bootstrap CI, permutation p, lift vs target."""
    jds = sorted(set(system) & set(baseline))
    if not jds:
        raise ValueError("no JDs in common")
    a = [system[j] for j in jds]
    b = [baseline[j] for j in jds]
    mean_a, lo, hi = bootstrap_ci(a, seed=seed)
    mean_b = sum(b) / len(b)
    test = paired_permutation_test(a, b, n_resamples=n_resamples, seed=seed)
    return {
        "n_jds": len(jds),
        "system_mean": mean_a,
        "system_ci95": (lo, hi),
        "baseline_mean": mean_b,
        "p_value": test["p_value"],
        **lift_check(mean_a, mean_b, target_pct),
    }
