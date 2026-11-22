"""CLI shell. Subcommands are stubs until later tasks; ``run`` enforces the budget guard."""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path

from .budget import PAID_NEEDS_APPROVAL, BudgetGuard
from .config import SUITES, EvalConfig
from .datasets.synthetic import safe_output_dir
from .manifest import build_manifest, write_manifest


def _budget(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from None
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    if v != 0:
        raise argparse.ArgumentTypeError(PAID_NEEDS_APPROVAL)
    return v


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic_talent_screener.evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("action", choices=["fetch"])
    d.add_argument("--dataset", choices=["deepset", "hiring_audit", "onet", "esco"], required=True)
    t = sub.add_parser("twins")
    t.add_argument("--bases", default="data/eval/bases")
    t.add_argument("--grid", default="core")
    t.add_argument("--out", default="data/eval/twins")
    r = sub.add_parser("run")
    r.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    r.add_argument("--systems", default="full")
    r.add_argument("--budget-usd", type=_budget, default=0.0)
    r.add_argument("--offline", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--runs-dir", default="runs")
    rep = sub.add_parser("report")
    rep.add_argument("run_dir")
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    return p


def _run(args: argparse.Namespace) -> int:
    args.runs_dir = str(safe_output_dir(args.runs_dir, _project_root()))
    suites = SUITES if args.suite == "all" else (args.suite,)
    cfg = EvalConfig(
        suites=suites,
        systems=tuple(s for s in args.systems.split(",") if s),
        seed=args.seed,
        budget_usd=args.budget_usd,
        offline=args.offline,
        runs_dir=args.runs_dir,
    )
    guard = BudgetGuard(cfg.budget_usd)
    run_dir = Path(cfg.runs_dir) / uuid.uuid4().hex[:12]
    extra = {
        "budget_usd": guard.budget_usd,
        "suites": list(cfg.suites),
        "systems": list(cfg.systems),
        "offline": cfg.offline,
        "suites_executed": [],
        "status": "skeleton",
    }
    write_manifest(run_dir, build_manifest(seed=cfg.seed, models=[], extra=extra))
    print("note: skeleton run, no suites were executed", file=sys.stderr)
    print(f"run directory: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "twins":
            args.out = str(safe_output_dir(args.out, _project_root()))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{args.command} is not implemented yet", file=sys.stderr)
    return 2
