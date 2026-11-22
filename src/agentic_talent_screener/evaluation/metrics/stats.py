"""Statistics helpers: bootstrap CI, paired permutation test, lift check. Seeded, offline."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from agentic_talent_screener.ranking.eval_harness import lift_pct


MAX_RESAMPLES = 100_000  # hard cap for bootstrap and permutation resamples (DoS guard)


def _check_resamples(n: int) -> None:
    if not 1 <= n <= MAX_RESAMPLES:
        raise ValueError(f"n_resamples must be in [1, {MAX_RESAMPLES}]")


def bootstrap_ci(
    values: Sequence[float],
    *,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point estimate, lower, upper)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    _check_resamples(n_resamples)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    point = float(stat(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boots = np.array([stat(arr[row]) for row in idx], dtype=float)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test on mean(a - b).

    Exact enumeration when 2^n <= n_resamples, otherwise Monte Carlo with the
    +1 correction (p = (extreme + 1) / (n_resamples + 1)).
    """
    _check_resamples(n_resamples)
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if len(a) == 0:
        raise ValueError("inputs must be non-empty")
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    observed = float(d.mean())
    tol = 1e-12
    if n < 30 and 2**n <= n_resamples:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = (signs * d).mean(axis=1)
        p = float(np.mean(np.abs(means) >= abs(observed) - tol))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
        means = (signs * d).mean(axis=1)
        extreme = int(np.sum(np.abs(means) >= abs(observed) - tol))
        p = (extreme + 1) / (n_resamples + 1)
    return {"mean_diff": observed, "p_value": p, "n": float(n)}


def lift_check(new: float, base: float, target_pct: float = 15.0) -> dict[str, float | bool]:
    """Relative lift of ``new`` over ``base`` in percent and whether it meets the target."""
    lift = lift_pct(new, base)
    return {"lift_pct": lift, "target_pct": target_pct, "meets_target": lift >= target_pct - 1e-9}


def paired_bootstrap_lift_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Paired bootstrap over units (JDs): CI of lift-% of mean(a) over mean(b), and of mean(a - b)."""
    _check_resamples(n_resamples)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be non-empty and of equal length")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n_resamples, x.size))
    ma, mb = x[idx].mean(axis=1), y[idx].mean(axis=1)
    lifts = np.array([lift_pct(float(p), float(q)) for p, q in zip(ma, mb)])
    alpha = (1 - confidence) / 2
    ql = np.quantile(lifts, [alpha, 1 - alpha])
    qd = np.quantile(ma - mb, [alpha, 1 - alpha])
    return {"lift_ci95": (float(ql[0]), float(ql[1])), "diff_ci95": (float(qd[0]), float(qd[1]))}
