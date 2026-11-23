"""Evaluation CLI: data, twins, run, report, compare.

``run`` enforces the free-only guard (``--budget-usd`` is 0) and the live-call cap (``--max-llm-calls``).
Every path goes through ``safe_output_dir`` (only ``data/eval`` and ``runs`` are writable).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .budget import PAID_NEEDS_APPROVAL
from .datasets import synthetic
from .datasets.synthetic import safe_output_dir
from .runner import DEFAULT_MAX_LLM_CALLS, CoreSettings, ALL_SYSTEMS, SUITE_PARTS, execute, resolve_systems

NOT_BUILT_SUITES = ("fairness", "extraction")  # E10 and E11: no runner yet; a run records that and stops
SUITE_CHOICES = ("core", "ranking", "robustness", "detection", *NOT_BUILT_SUITES, "all")


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


def _int_min(low: int, high: int | None = None):
    def parse(value: str) -> int:
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
        if v < low or (high is not None and v > high):
            raise argparse.ArgumentTypeError(f"must be an integer in [{low}, {high if high is not None else 'inf'}]")
        return v
    return parse


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic_talent_screener.evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data", help="generate the synthetic pool, or fetch public datasets (not built yet)")
    d.add_argument("action", choices=["generate", "fetch"])
    d.add_argument("--dataset", choices=["deepset", "hiring_audit", "onet", "esco"])
    d.add_argument("--n", type=_int_min(synthetic.MIN_POOL_SIZE, synthetic.MAX_POOL_SIZE),
                   default=synthetic.DEFAULT_POOL_SIZE)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="data/eval/synthetic")
    d.add_argument("--formats", default=",".join(synthetic.FORMATS))
    t = sub.add_parser("twins", help="generate adversarial twins over the attack grid")
    t.add_argument("--n-bases", type=_int_min(1, 200), default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--grid", choices=["core", "full"], default="core")
    t.add_argument("--families", default="", help="comma-separated family ids, default all")
    t.add_argument("--out", default="data/eval/twins")
    t.add_argument("--formats", default="pdf,docx,txt")
    r = sub.add_parser("run", help="run a suite and write runs/<run_id>/")
    r.add_argument("--suite", choices=SUITE_CHOICES, default="core")
    r.add_argument("--systems", default="default",
                   help="comma-separated names or groups (default, all, baselines, ablations); "
                        f"names: {', '.join(ALL_SYSTEMS)}")
    r.add_argument("--budget-usd", type=_budget, default=0.0)
    r.add_argument("--offline", action="store_true", help="never call the network; use cached LLM responses only")
    r.add_argument("--max-llm-calls", type=_int_min(0), default=DEFAULT_MAX_LLM_CALLS,
                   help="abort with partial results after this many real HTTP requests to the provider (retries count, cache hits do not; eval runs cap llm.py retries at 1; "
                        "the free tier allows about 50 a day)")
    r.add_argument("--min-verdict-n", type=_int_min(1), default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--pool-size", type=_int_min(1, 500), default=150)
    r.add_argument("--n-bases", type=_int_min(1, 500), default=20)
    r.add_argument("--n-jds", type=_int_min(1, 60), default=10)
    r.add_argument("--grid", choices=["core", "full"], default="core")
    r.add_argument("--families", default="", help="comma-separated attack family ids, default all")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--cache-dir", default="data/eval/cache")
    r.add_argument("--run-id", default=None)
    rep = sub.add_parser("report", help="(re)write results.json, report.md and per_item.jsonl for a run")
    rep.add_argument("run_dir")
    c = sub.add_parser("compare", help="diff two runs' results.json; exit 1 when B regressed")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--tolerance", type=float, default=0.0)
    c.add_argument("--all", action="store_true", help="also list unchanged metrics")
    return p


def _data(args: argparse.Namespace) -> int:
    if args.action == "fetch":
        print("data fetch is not implemented yet (public dataset loaders are E8)", file=sys.stderr)
        return 2
    out = safe_output_dir(args.out, _project_root())
    formats = tuple(f for f in args.formats.split(",") if f)
    path = synthetic.write_pool(synthetic.generate_pool(args.n, args.seed), out, formats, _project_root())
    print(f"wrote {args.n} synthetic resumes to {path}")
    return 0


def _twins(args: argparse.Namespace) -> int:
    from .attacks.twins import generate_twins, write_twins

    out = safe_output_dir(args.out, _project_root())
    families = [f for f in args.families.split(",") if f] or None
    bases = [synthetic.make_resume(i, args.seed) for i in range(args.n_bases)]
    twins = generate_twins(bases, args.grid, args.seed, families)
    path = write_twins(twins, out, tuple(f for f in args.formats.split(",") if f), _project_root())
    print(f"wrote {len(twins)} twins to {path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    root = _project_root()
    runs_dir = safe_output_dir(args.runs_dir, root)
    if args.suite in NOT_BUILT_SUITES or args.suite == "all":
        return _skeleton(args, runs_dir)
    settings = CoreSettings(
        suite=args.suite, systems=resolve_systems(args.systems), seed=args.seed, pool_size=args.pool_size,
        n_bases=args.n_bases, n_jds=args.n_jds, grid=args.grid,
        families=tuple(f for f in args.families.split(",") if f) or None, min_n=args.min_verdict_n,
        offline=args.offline, max_llm_calls=args.max_llm_calls, cache_dir=args.cache_dir)
    run_dir = execute(settings, base=root, runs_dir=runs_dir, run_id=args.run_id)
    print(f"run directory: {run_dir}")
    print(f"report: {run_dir / 'report.md'}")
    return 0


def _skeleton(args: argparse.Namespace, runs_dir: Path) -> int:
    """Fairness (E10) and extraction (E11) have no runner yet. ``all`` runs the core suite and says so."""
    import uuid

    from .manifest import build_manifest, write_manifest

    if args.suite == "all":
        core = argparse.Namespace(**{**vars(args), "suite": "core"})
        rc = _run(core)
        print("note: fairness (E10) and extraction (E11) are not built; the report marks them not measurable",
              file=sys.stderr)
        return rc
    run_dir = runs_dir / (args.run_id or uuid.uuid4().hex[:12])
    extra = {"budget_usd": 0.0, "suites": [args.suite], "offline": args.offline, "suites_executed": [],
             "status": "skeleton", "reason": f"the {args.suite} suite is not built yet"}
    write_manifest(run_dir, build_manifest(seed=args.seed, models=[], extra=extra))
    print(f"note: the {args.suite} suite is not built yet; no suites were executed", file=sys.stderr)
    print(f"run directory: {run_dir}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from .report import build_report

    run_dir = safe_output_dir(args.run_dir, _project_root())
    results = build_report(run_dir)
    print(f"wrote {run_dir / 'results.json'}, {run_dir / 'report.md'}, {run_dir / 'per_item.jsonl'}")
    sm = results["summary"]
    print(f"slide-9: {sm['pass']} pass, {sm['fail']} fail, {sm['inconclusive']} inconclusive, "
          f"{sm['not measurable']} not measurable")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .report import compare_markdown, compare_runs

    root = _project_root()
    diff = compare_runs(safe_output_dir(args.run_a, root), safe_output_dir(args.run_b, root), args.tolerance)
    print(compare_markdown(diff, show_all=args.all))
    return 1 if diff["n_regressions"] else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "data":
            return _data(args)
        if args.command == "twins":
            return _twins(args)
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            return _report(args)
        if args.command == "compare":
            return _compare(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2  # pragma: no cover


__all__ = ["SUITE_PARTS", "build_parser", "main"]
