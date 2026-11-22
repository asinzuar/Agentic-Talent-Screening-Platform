"""Ranking metrics: nDCG@k, BM25 baseline, per-JD aggregation, lift vs baseline.

Graded relevance in, ranked IDs out. Reuses the existing harness functions.
"""

from __future__ import annotations

import hashlib
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

from .stats import (
    bootstrap_ci,
    lift_check,
    paired_bootstrap_lift_ci,
    paired_permutation_test,
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def ndcg(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """nDCG@k (default 10), graded relevance, exponential gain. k must be >= 1.

    Duplicate IDs in ``ranked_ids`` raise ValueError (they would inflate the score).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_ids contains duplicate candidate IDs")
    return ndcg_at_k(ranked_ids, relevance, k)


def rank_by_score(scores: Mapping[str, float], seed: int = 0) -> list[str]:
    """Order IDs by score descending. Ties are broken by sha256(f"{seed}:{id}"), not the raw
    ID, so naming cannot bias results. Prefer ``tie_aware_ndcg`` for scoring tied systems."""

    def h(i: str) -> str:
        return hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()

    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], h(kv[0])))]


def tie_aware_ndcg(
    scores: Mapping[str, float], relevance: Mapping[str, float], k: int = 10
) -> float:
    """Expected nDCG@k over all random orderings within each group of tied scores
    (McSherry and Najork, 2008). Each tied item at positions p..p+m-1 contributes the
    group's mean gain at every position. Independent of ID strings. Items with no
    judgement count as relevance 0. Ideal DCG uses all judged items."""
    if k < 1:
        raise ValueError("k must be >= 1")
    groups: dict[float, list[float]] = {}
    for cid, sc in scores.items():
        groups.setdefault(sc, []).append(2.0 ** relevance.get(cid, 0.0) - 1.0)
    dcg, pos = 0.0, 0
    for sc in sorted(groups, reverse=True):
        gains = groups[sc]
        mean_gain = sum(gains) / len(gains)
        for _ in gains:
            if pos >= k:
                break
            dcg += mean_gain / math.log2(pos + 2)
            pos += 1
        if pos >= k:
            break
    ideal_gains = sorted((2.0**r - 1.0 for r in relevance.values()), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / ideal if ideal > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    return [t for t in (x.strip(".-") for x in _TOKEN.findall(text.lower())) if t]


def bm25_scores(
    query: str, docs: Mapping[str, str], *, k1: float = 1.2, b: float = 0.75
) -> dict[str, float]:
    """Okapi BM25 (k1=1.2, b=0.75) of ``query``; query terms count once (k3=0). Per doc. IDF = ln(1 + (N - df + 0.5) / (df + 0.5))."""
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
    cis = paired_bootstrap_lift_ci(a, b, seed=seed)
    return {
        "n_jds": len(jds),
        "system_mean": mean_a,
        "system_ci95": (lo, hi),
        "baseline_mean": mean_b,
        "p_value": test["p_value"],
        "baseline_zero": mean_b <= 0,
        **cis,
        **lift_check(mean_a, mean_b, target_pct),
    }
