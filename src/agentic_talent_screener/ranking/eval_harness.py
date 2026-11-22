"""Evaluation harness: four systems side by side (SPEC §8 "Matching & ranking", "Robustness").

Pure Python + NumPy, no extra ML dependencies.

Systems
  keyword_ats          share of requirements whose option text appears as a whole word anywhere
                       in the candidate text (hidden text included); no proof levels, no flags
  cosine_similarity    TF-IDF cosine of JD requirement text vs candidate text (optional
                       ``embed_fn`` hook for real embeddings later)
  ablated_no_security  our scorer with no L3 grounding and no L4 flag exclusion
  full_engine          the full scorer (L3 + L4) under the active ScoringConfig

The sample ground truth is hand-assigned for the *synthetic* pool, so numbers here prove the
mechanics, not real-world accuracy (SPEC §8 asks for ~20 human-graded pairs).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, ScoringConfig
from .matching import MatchingAgent
from .sample_data import sample_matcher, sample_pool, sample_requirement_profile
from .schemas import (
    RequirementProfile,
    Span,
    VerifiedProfile,
)
from .scorer import score_pool

SYSTEMS = ("keyword_ats", "cosine_similarity", "ablated_no_security", "full_engine")
CHEATERS = ("C-0417", "C-0555")
CHEATER_LABELS = {"C-0417": "Hidden Text", "C-0555": "Buzzword Stuffing"}
LIFT_TARGET_PCT = 15.0  # SPEC §8: "clearly better (~15%)"

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def sample_ground_truth() -> dict[str, float]:
    """Hand-graded relevance (0-5) for the synthetic sample pool."""
    return {
        "C-0932": 5.0,
        "C-0101": 3.5,
        "C-0204": 3.0,
        "C-0770": 2.0,
        "C-0417": 1.0,
        "C-0555": 0.5,
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 5) -> float:
    """NDCG@k with exponential gain (2^rel - 1). Unknown IDs count as relevance 0."""

    def dcg(rels: Sequence[float]) -> float:
        return sum((2.0**r - 1.0) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    got = dcg([relevance.get(i, 0.0) for i in ranked_ids])
    ideal = dcg(sorted(relevance.values(), reverse=True))
    return got / ideal if ideal > 0 else 0.0


def _avg_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks of ``values`` (ascending), ties share the average rank."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr))
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> float:
    """Spearman rank correlation between the system order and the ground-truth order."""
    ids = [i for i in ranked_ids if i in relevance]
    if len(ids) < 2:
        return 0.0
    sys_rank = np.arange(1, len(ids) + 1, dtype=float)
    truth_rank = _avg_ranks([-relevance[i] for i in ids])  # best grade = rank 1
    if sys_rank.std() == 0 or truth_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(sys_rank, truth_rank)[0, 1])


def kendall_tau(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> float:
    """Kendall tau-b between the system order and the ground truth (ties in truth handled)."""
    ids = [i for i in ranked_ids if i in relevance]
    n = len(ids)
    if n < 2:
        return 0.0
    conc = disc = ties_truth = 0
    for a in range(n):
        for b in range(a + 1, n):
            d = relevance[ids[a]] - relevance[ids[b]]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
            else:
                ties_truth += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - ties_truth) * n0)  # system order has no ties
    return (conc - disc) / denom if denom > 0 else 0.0


def top1_hit(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> bool:
    """True if the system's #1 is one of the ground-truth best candidates."""
    if not ranked_ids or not relevance:
        return False
    return relevance.get(ranked_ids[0], 0.0) == max(relevance.values())


def lift_pct(new: float, base: float) -> float:
    """Relative improvement of ``new`` over ``base`` in percent."""
    if base <= 0:
        return 100.0 if new > 0 else 0.0
    return (new - base) / base * 100.0


# --------------------------------------------------------------------------------------
# Baseline systems
# --------------------------------------------------------------------------------------


def keyword_ats_scores(
    candidates: Sequence[VerifiedProfile], profile: RequirementProfile
) -> dict[str, float]:
    """Share of active requirements with any option keyword in the text. No proof levels."""
    reqs = profile.active_requirements
    out: dict[str, float] = {}
    for c in candidates:
        text = _candidate_text(c).lower()
        hits = sum(
            any(
                re.search(rf"(?<![a-z0-9]){re.escape(o.text.lower())}(?![a-z0-9])", text)
                for o in r.criteria
            )
            for r in reqs
        )
        out[c.candidate_id] = hits / len(reqs) if reqs else 0.0
    return out


_TOKEN = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> list[str]:
    return [t for t in (x.strip(".") for x in _TOKEN.findall(text.lower())) if t]


def _jd_text(profile: RequirementProfile) -> str:
    return " ".join(
        f"{r.statement} {r.quote or r.rationale or ''}" for r in profile.active_requirements
    )


def _candidate_text(c: VerifiedProfile) -> str:
    """Everything a naive reader sees: every statement and cited quote, hidden text included."""
    parts: list[str] = []
    for f in c.facts:
        parts.append(f.statement)
        parts.extend(s.quote for s in f.spans)
    return " ".join(parts)


def cosine_scores(
    candidates: Sequence[VerifiedProfile],
    profile: RequirementProfile,
    embed_fn: EmbedFn | None = None,
) -> dict[str, float]:
    """TF-IDF cosine similarity of JD requirement text vs candidate text.

    If ``embed_fn`` is given it must map a list of texts to an (n, d) array; cosine similarity
    of those embeddings is used instead (future hook).
    """
    jd = _jd_text(profile)
    docs = [_candidate_text(c) for c in candidates]
    if embed_fn is not None:
        vecs = np.asarray(embed_fn([jd, *docs]), dtype=float)
    else:
        toks = [_tokens(t) for t in (jd, *docs)]
        vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
        n_docs = len(toks)
        df = np.zeros(len(vocab))
        for t in toks:
            for w in set(t):
                df[vocab[w]] += 1
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        vecs = np.zeros((n_docs, len(vocab)))
        for row, t in enumerate(toks):
            for w in t:
                vecs[row, vocab[w]] += 1.0
        vecs = (1.0 + np.log(np.where(vecs > 0, vecs, 1.0))) * (vecs > 0) * idf
    norms = np.linalg.norm(vecs, axis=1)
    out: dict[str, float] = {}
    for i, c in enumerate(candidates, start=1):
        denom = norms[0] * norms[i]
        out[c.candidate_id] = float(vecs[0] @ vecs[i] / denom) if denom > 0 else 0.0
    return out


def _strip_protection(c: VerifiedProfile) -> VerifiedProfile:
    """Emulate 'no L3': every cited span counts as visible and every fact looks grounded."""

    def open_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
        if not spans:
            return (Span(location="p1:s0001", quote="(ungrounded)"),)
        return tuple(
            s.model_copy(update={"visible": True, "quote": s.quote.strip() or "(ungrounded)"})
            for s in spans
        )

    return c.model_copy(
        update={
            "facts": tuple(
                f.model_copy(update={"spans": open_spans(f.spans)})
                if f.evidence_level or f.kind == "education"
                else f
                for f in c.facts
            )
        }
    )


# --------------------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------------------


def _order(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def _metrics(ranked: Sequence[str], truth: Mapping[str, float], k: int) -> dict[str, Any]:
    return {
        f"ndcg_at_{k}": round(ndcg_at_k(ranked, truth, k), 6),
        "spearman_rho": round(spearman_rho(ranked, truth), 6),
        "kendall_tau": round(kendall_tau(ranked, truth), 6),
        "top1_hit": top1_hit(ranked, truth),
    }


def run_benchmark(
    candidates: Sequence[VerifiedProfile | Mapping[str, Any]] | None = None,
    profile: RequirementProfile | None = None,
    matcher: MatchingAgent | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
    ground_truth: Mapping[str, float] | None = None,
    cheaters: Sequence[str] = CHEATERS,
    k: int = 5,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Run all four systems on one pool and return a JSON-serialisable result."""
    profile = profile or sample_requirement_profile()
    matcher = matcher or sample_matcher()
    truth = dict(ground_truth if ground_truth is not None else sample_ground_truth())
    raw_candidates = list(candidates) if candidates is not None else sample_pool()
    pool = [
        c if isinstance(c, VerifiedProfile) else VerifiedProfile.model_validate(c)
        for c in raw_candidates
    ]

    full = score_pool(pool, profile, matcher, config)
    ablated = score_pool(
        [_strip_protection(c) for c in pool],
        profile,
        matcher,
        config.model_copy(update={"exclude_flag_risks": ()}),
    )
    scores: dict[str, dict[str, float]] = {
        "keyword_ats": keyword_ats_scores(pool, profile),
        "cosine_similarity": cosine_scores(pool, profile, embed_fn),
        "ablated_no_security": {r.candidate_id: r.fit_score for r in ablated.results},
        "full_engine": {r.candidate_id: r.fit_score for r in full.results},
    }
    rankings = {
        "keyword_ats": _order(scores["keyword_ats"]),
        "cosine_similarity": _order(scores["cosine_similarity"]),
        # score_pool already orders ok results first, errors last
        "ablated_no_security": [r.candidate_id for r in ablated.results],
        "full_engine": [r.candidate_id for r in full.results],
    }

    systems: dict[str, Any] = {}
    for name in SYSTEMS:
        ranked = rankings[name]
        systems[name] = {
            "ranking": ranked,
            "scores": {cid: round(float(s), 6) for cid, s in scores[name].items()},
            "ranks": {cid: i + 1 for i, cid in enumerate(ranked)},
            "metrics": _metrics(ranked, truth, k),
        }

    ndcg_key = f"ndcg_at_{k}"
    full_ndcg = systems["full_engine"]["metrics"][ndcg_key]
    lift: dict[str, Any] = {}
    for base in ("keyword_ats", "cosine_similarity", "ablated_no_security"):
        pct = lift_pct(full_ndcg, systems[base]["metrics"][ndcg_key])
        lift[f"vs_{base}"] = {"lift_pct": round(pct, 3), "target_met": pct >= LIFT_TARGET_PCT}

    cheater_ranks = {
        name: {cid: systems[name]["ranks"].get(cid) for cid in cheaters} for name in SYSTEMS
    }
    blocked = {
        name: sum(1 for cid in cheaters if (r := cheater_ranks[name][cid]) is not None and r > 3)
        for name in SYSTEMS
    }

    return {
        "k": k,
        "profile": {"profile_id": profile.profile_id, "version": profile.version},
        "config": {
            "config_version": config.config_version,
            "matcher_version": matcher.version,
            "exclude_flag_risks": list(config.exclude_flag_risks),
        },
        "ground_truth": truth,
        "systems": systems,
        "lift": lift,
        "adversarial": {
            "candidates": list(cheaters),
            "labels": {cid: CHEATER_LABELS.get(cid, cid) for cid in cheaters},
            "ranks": cheater_ranks,
            "cheaters_blocked": blocked,
            "total": len(cheaters),
        },
        "note": "Synthetic sample ground truth: demonstrates mechanics, not real-world accuracy.",
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def format_markdown(result: Mapping[str, Any]) -> str:
    k = result["k"]
    lines = [
        f"| System | NDCG@{k} | Spearman ρ | Kendall τ | Top-1 | Cheaters in Top 3 |",
        "|---|---|---|---|---|---|",
    ]
    total = result["adversarial"]["total"]
    for name in SYSTEMS:
        s = result["systems"][name]
        m = s["metrics"]
        blocked = result["adversarial"]["cheaters_blocked"][name]
        lines.append(
            f"| {name} | {m[f'ndcg_at_{k}']:.3f} | {m['spearman_rho']:+.3f} | "
            f"{m['kendall_tau']:+.3f} | {'yes' if m['top1_hit'] else 'no'} | {total - blocked}/{total} |"
        )
    lines += ["", "| Full engine lift over | % lift | ≥15% target |", "|---|---|---|"]
    for key, v in result["lift"].items():
        lines.append(f"| {key[3:]} | {v['lift_pct']:+.1f}% | {'met' if v['target_met'] else 'not met'} |")
    lines += ["", "Adversarial ranks (1 = top):", "", "| System | " + " | ".join(result["adversarial"]["candidates"]) + " |"]
    lines.append("|---|" + "---|" * len(result["adversarial"]["candidates"]))
    for name in SYSTEMS:
        r = result["adversarial"]["ranks"][name]
        lines.append(f"| {name} | " + " | ".join(str(r[c]) for c in result["adversarial"]["candidates"]) + " |")
    lines += ["", f"_{result['note']}_"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_markdown(run_benchmark()))
