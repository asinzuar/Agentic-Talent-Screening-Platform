"""CLI wiring for the claims suite (E11), kept apart so ``cli.py`` needs only a handful of added lines.

``cli.py`` calls ``register`` while building the parser, ``run`` for ``--suite claims`` and ``audit`` for the
``audit import`` command. The claims suite is free-only: it builds its own ``CachedChat`` with ``BudgetGuard(0.0)``
(see ``claims_suite.run_claims``); there is no paid path to enable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets.synthetic import safe_output_dir

SUITE = "claims"


def _int(low: int, high: int | None = None):
    def parse(value: str) -> int:
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
        if v < low or (high is not None and v > high):
            raise argparse.ArgumentTypeError(f"must be an integer in [{low}, {high if high is not None else 'inf'}]")
        return v
    return parse


def register(sub: argparse._SubParsersAction, run_parser: argparse.ArgumentParser) -> None:
    """Add the claims-only options to ``run`` and the ``audit`` command."""
    run_parser.add_argument("--judge", action="store_true",
                            help="claims suite: also ask a free-only LLM judge (CachedChat, opt-in, counts against "
                                 "--max-llm-calls) whether cited spans support their claims")
    run_parser.add_argument("--judge-n", type=_int(0, 500), default=30, help="claims suite: citations to judge")
    run_parser.add_argument("--audit-n", type=_int(1, 100), default=10,
                            help="claims suite: explanation lines to export for human audit")
    au = sub.add_parser("audit", help="claims suite human audit: import a rated sheet and refresh the report")
    au.add_argument("action", choices=["import"])
    au.add_argument("run_dir")
    au.add_argument("ratings_csv")
    au.add_argument("--min-n", type=_int(1), default=10)


PAID_REFUSAL = ("refused [free_only]: the claims suite is free-only and has no paid path; "
                "run it without --allow-paid and --budget-usd")


def run(args: argparse.Namespace, root: Path, runs_dir: Path) -> int:
    from .claims_suite import ClaimsSettings, execute_claims

    if getattr(args, "allow_paid", False) or (getattr(args, "budget_usd", None) or 0) > 0:
        print(PAID_REFUSAL, file=sys.stderr)  # same wording style as the core suite's console refusals
        return 2

    cs = ClaimsSettings(
        seed=args.seed, pool_size=args.pool_size, n_bases=args.n_bases, n_jds=args.n_jds, grid=args.grid,
        min_n=args.min_verdict_n, offline=args.offline, max_llm_calls=args.max_llm_calls,
        cache_dir=args.cache_dir, judge=args.judge, judge_n=args.judge_n, audit_n=args.audit_n)
    run_dir = execute_claims(cs, base=root, runs_dir=runs_dir, run_id=args.run_id)
    print(f"run directory: {run_dir}")
    print(f"report: {run_dir / 'report.md'}")
    print(f"audit sheet: {run_dir / 'audit' / 'audit_sheet.csv'} (humans rate it, then `audit import`)")
    return 0


def audit(args: argparse.Namespace, root: Path) -> int:
    from . import audit as audit_kit
    from .report import build_report

    run_dir = safe_output_dir(args.run_dir, root)
    res = audit_kit.import_ratings(run_dir, Path(args.ratings_csv), min_n=args.min_n)
    build_report(run_dir)
    print(f"imported {res['n_ratings']} ratings ({res['n_rows_rated']} rows, {res['n_pending']} pending, "
          f"{res['n_rejected']} rejected); citation right: {res['citation_right']['status']}")
    if res.get("rows_with_edited_content"):
        print("warning: cells were edited on rows " + ", ".join(res["rows_with_edited_content"]), file=sys.stderr)
    for bad in res["rejected"][:10]:
        print(f"rejected line {bad['line']} ({bad['row_id']}): {bad['reason']}", file=sys.stderr)
    return 0 if not res["n_rejected"] else 1
