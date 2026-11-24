"""Evaluation CLI: data, twins, run, report, compare.

``run`` enforces the free-only guard (``--budget-usd`` is 0) and the live-call cap (``--max-llm-calls``).
Every path goes through ``safe_output_dir`` (only ``data/eval`` and ``runs`` are writable).
"""

from __future__ import annotations

import argparse
from typing import Any
import math
import sys
from pathlib import Path

from . import claims_cli
from .budget import MAX_RUN_BUDGET_USD, MODEL_POLICIES, PHASE_CAP_USD, valid_budget
from .datasets import synthetic
from .datasets.synthetic import safe_output_dir
from .runner import DEFAULT_MAX_LLM_CALLS, CoreSettings, ALL_SYSTEMS, SUITE_PARTS, execute, resolve_systems

NOT_BUILT_SUITES = ("fairness",)  # E10: no runner yet; a run records that and stops (E11 is `claims`)
SUITE_CHOICES = ("core", "ranking", "robustness", "detection", *NOT_BUILT_SUITES, "all")
SUITE_CHOICES = (*SUITE_CHOICES, claims_cli.SUITE)  # E11: registered here, built in claims_cli / claims_suite


def _budget(value: str) -> float:
    try:
        return valid_budget(float(value))  # NaN, inf, negatives and anything above the ceiling are refused
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _fraction(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from None
    if not 0.0 <= v <= 1.0:  # also rejects NaN
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
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
    r.add_argument("--allow-paid", action="store_true",
                   help="permit the allow-listed paid model (qwen/qwen3.8-flash) when free models are rate limited; "
                        "spend is measured from OpenRouter /key, capped per run and for phase M1")
    r.add_argument("--budget-usd", type=_budget, default=None,
                   help=f"per-run paid cap in USD, only with --allow-paid (default {MAX_RUN_BUDGET_USD}, max {MAX_RUN_BUDGET_USD})")
    r.add_argument("--model-policy", choices=list(MODEL_POLICIES), default="free-first",
                   help="free-first (default): two free models, then the paid model only after both are rate limited "
                        "and only with --allow-paid. paid-direct: use the allow-listed paid model straight away "
                        "(needs --allow-paid; same per-run, phase M1, baseline and base-URL caps)")
    r.add_argument("--rate-limit-switch-after", type=_int_min(1, 20), default=3,
                   help="consecutive 429s on one model before routing to the next")
    r.add_argument("--offline", action="store_true", help="never call the network; use cached LLM responses only")
    r.add_argument("--max-llm-calls", type=_int_min(0), default=DEFAULT_MAX_LLM_CALLS,
                   help="abort with partial results after this many real HTTP requests to the provider (retries count, cache hits do not; eval runs cap llm.py retries at 1; "
                        "the free tier allows about 50 a day)")
    r.add_argument("--min-verdict-n", type=_int_min(1), default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--pool-size", type=_int_min(1, 500), default=150,
                   help="synthetic resumes in the pool (clean ranking population). Only part of it is relevant to "
                        "the robustness JD, so size it at about 4x the clean resumes you need relevant")
    r.add_argument("--n-bases", type=_int_min(1, 500), default=20,
                   help="resumes that get attack twins = the clean resumes of the robustness and detection splits. "
                        "For verdicts use at least --min-verdict-n (30) bases; twins = bases x attack grid")
    r.add_argument("--relevant-fraction", type=_fraction, default=0.5,
                   help="share of the --n-bases drawn from resumes relevant to the robustness JD (default 0.5, "
                        "the rest are non-relevant). Capped by how many relevant resumes the pool holds; "
                        "the run warns when the pool is too small")
    r.add_argument("--twins-per-base", type=_int_min(1, 500), default=None,
                   help="attack twins per base: N grid cells per base (rotating through the cells so each is used about "
                        "equally often). Default: every cell of the grid for every base. One per base with 30 bases "
                        "gives 30 twins in 30 clusters, a tighter interval than 8 bases x 4 cells")
    r.add_argument("--n-jds", type=_int_min(1, 60), default=10,
                   help="synthetic job descriptions for ranking; ranking n = the number of JDs")
    r.add_argument("--grid", choices=["core", "full"], default="core")
    r.add_argument("--families", default="", help="comma-separated attack family ids, default all")
    r.add_argument("--dry-run", action="store_true",
                   help="print the planned sizes (resumes, relevant bases, twins, JDs) and a rough paid-cost estimate, "
                        "then exit without any model call, file or spend")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--cache-dir", default="data/eval/cache", help="deprecated and ignored; LLM responses are cached in eval_store.sqlite3 at the repo root")
    r.add_argument("--run-id", default=None)
    claims_cli.register(sub, r)
    po = sub.add_parser("phase-open", help="open the M1 spend phase once in the committed store (needed before any paid run)")
    po.add_argument("--phase", default="M1", choices=["M1"])
    rep = sub.add_parser("report", help="(re)write results.json, report.md and per_item.jsonl for a run")
    rep.add_argument("run_dir")
    c = sub.add_parser("compare", help="diff two runs (run dirs, or run ids in the eval store); exit 1 when B regressed")
    c.add_argument("run_a", help="run id (runs/<id> or the eval store) or a path")
    c.add_argument("run_b", help="run id (runs/<id> or the eval store) or a path")
    c.add_argument("--tolerance", type=float, default=0.0)
    c.add_argument("--all", action="store_true", help="also list unchanged metrics")
    st = sub.add_parser("store", help="eval store maintenance (see evaluation/STORE.md)")
    st.add_argument("action", choices=["merge", "stats"])
    st.add_argument("other", nargs="?", help="merge: the other eval_store.sqlite3 to union into the root store")
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
    if args.suite == claims_cli.SUITE:
        return claims_cli.run(args, root, runs_dir)
    if args.suite in NOT_BUILT_SUITES or args.suite == "all":
        return _skeleton(args, runs_dir)
    if args.budget_usd is not None and args.budget_usd > 0 and not args.allow_paid:
        raise ValueError("--budget-usd above 0 needs --allow-paid")
    budget = (MAX_RUN_BUDGET_USD if args.budget_usd is None else args.budget_usd) if args.allow_paid else 0.0
    if args.allow_paid and budget <= 0:
        raise ValueError("--allow-paid needs --budget-usd above 0")
    settings = CoreSettings(
        allow_paid=args.allow_paid, budget_usd=budget, switch_after_429=args.rate_limit_switch_after,
        model_policy=args.model_policy, relevant_fraction=args.relevant_fraction,
        twins_per_base=args.twins_per_base,
        runs_dir=args.runs_dir, suite=args.suite, systems=resolve_systems(args.systems), seed=args.seed, pool_size=args.pool_size,
        n_bases=args.n_bases, n_jds=args.n_jds, grid=args.grid,
        families=tuple(f for f in args.families.split(",") if f) or None, min_n=args.min_verdict_n,
        offline=args.offline, max_llm_calls=args.max_llm_calls, cache_dir=args.cache_dir)
    if args.dry_run:
        print(estimate_text(settings, root))
        return 0
    run_dir = execute(settings, base=root, runs_dir=runs_dir, run_id=args.run_id)
    print(f"run directory: {run_dir}")
    print(f"report: {run_dir / 'report.md'}")
    _print_refusals(run_dir)
    _print_coverage(run_dir, settings)
    return 0


# Dry-run assumptions (printed with the estimate). The worst case uses the cap's own formula (budget.worst_case_usd) with
# the real per-stage max_tokens; the input bound is the document bytes plus a per-stage prompt-and-schema overhead,
# calibrated on a reviewed mocked run (T67: about $0.015 reserved per document end to end). The expected figure uses
# chars/4 input tokens and typical output sizes. Estimates, never caps.
STAGE_MAX_TOKENS = {"l1": 1536, "l2": 4096, "match": 20000, "naive": 64}  # as in l1_injection, l2_resume_agent, matching
STAGE_OVERHEAD_BYTES = {"l1": 4000, "l2": 7000, "match": 3500, "naive": 1500}
STAGE_EXPECTED_OUT_TOKENS = {"l1": 150, "l2": 1200, "match": 800, "naive": 40}
L2_CALLS_PER_DOC = 2  # the live store holds 100 L1 and 200 L2 answers for a 100-document pool (t41v3-run1)
MATCH_SHARE = 0.3  # matching calls per document: attacked or stopped documents never reach the matcher


def cached_answers(store_root: Path | None, model: str) -> dict[str, int]:
    """Cached answers per stage for ``model`` in the committed store, read-only (nothing is opened for writing)."""
    import sqlite3

    from .store import store_path

    if store_root is None or not store_path(store_root).is_file():
        return {}
    try:
        con = sqlite3.connect(f"file:{store_path(store_root).as_posix()}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT schema_name, COUNT(*) FROM llm_cache WHERE model=? GROUP BY 1", (model,)).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {}
    names = {"l1b_security_findings": "l1", "l2_resume_extraction": "l2", "semantic_matches": "match"}
    return {names[n]: int(k) for n, k in rows if n in names}


SETTLED_OVER_EXPECTED = 1.2  # settled spend ran 18% above /key in the live run (t41v3-run1: $0.212 vs $0.179)


def _resumed_lines(lines: list[str], sysp: dict[str, Any], probe: dict[str, Any], cached: dict[str, int], model: str,
                   cap: float, hold: float, expected: Any, worst: Any) -> float:
    """Resumed-run forecast from the exact probe (per stage and per document, not ratios). Appends the lines and
    returns the number of calls that remain. A document that missed in intake is stubbed in the probe and stops early,
    so its matching call is added as an upper bound."""
    n_jds = probe["n_jds"]

    def cost(d: dict[str, int]) -> float:
        return sum(d.get(s, 0) * expected(s) for s in ("l1", "l2", "match", "naive"))

    def with_matching(d: dict[str, int], docs_missing_intake: int) -> dict[str, int]:
        out = dict(d)
        out["match"] = out.get("match", 0) + docs_missing_intake * n_jds
        return out

    clean = sysp["clean"]
    miss_clean = max(clean.get("l1", 0), (clean.get("l2", 0) + 1) // 2)
    def second_l2(d: dict[str, int]) -> dict[str, int]:
        """A document missing from the cache at L1 and L2 makes two L2 calls in a real run (the stub makes one)."""
        out = dict(d)
        out["l2"] = d.get("l2", 0) + min(d.get("l1", 0), d.get("l2", 0))
        return out

    parts = [with_matching(second_l2(clean), miss_clean)] + [
        with_matching(second_l2(t), 1 if (t.get("l1", 0) or t.get("l2", 0)) else 0) for t in sysp["twins"]]
    totals = {k: sum(p.get(k, 0) for p in parts) for k in ("l1", "l2", "match")}
    calls = float(sum(totals.values()))
    exp_total = sum(cost(p) for p in parts)
    settled = exp_total * SETTLED_OVER_EXPECTED
    next_worst = worst("match")
    room = cap - hold - next_worst
    lines += [
        f"cached answers in the store for {model}: " + ", ".join(f"{k} {cached.get(k, 0)}" for k in ("l1", "l2", "match"))
        + " (free on a resumed run)",
        f"resumed run, probed document by document against the store: {calls:.0f} calls remain (L1 {totals['l1']}, "
        f"L2 {totals['l2']}, matching {totals['match']}); the clean pool needs {sum(clean.get(k, 0) for k in ('l1', 'l2', 'match'))}"
        f" ({len(sysp['twins']) - sum(1 for t in sysp['twins'] if not t)} of {len(sysp['twins'])} twins still need calls)",
        f"resumed run cost: expected ${exp_total:.2f}, settled about ${settled:.2f} (x{SETTLED_OVER_EXPECTED}); the cap leaves "
        f"${room:.2f} for settled spend (${cap:.2f} - headroom ${hold:.2f} - one worst-case matching call ${next_worst:.2f})",
    ]
    if settled <= room:
        lines.append(f"resumed run under the ${cap:.2f} cap: it can finish")
    else:
        spent, done_twins = cost(parts[0]) * SETTLED_OVER_EXPECTED, 0
        if spent > room:
            where = "inside the clean pool (no complete pool, so no FPR and no ranking)"
        else:
            for t in parts[1:]:
                spent += cost(t) * SETTLED_OVER_EXPECTED
                if spent > room:
                    break
                done_twins += 1
            where = (f"after the clean pool and about {done_twins} of {len(parts) - 1} twins "
                     f"(first documents in run order)")
        lines.append(f"warning: resumed run under the ${cap:.2f} cap will NOT finish: it stops {where}, "
                     f"and is scored as partial (cap reached after N of M documents)")
    return calls


def estimate_text(settings: CoreSettings, store_root: Path | None = None) -> str:
    """Planned sizes and cost on the cap's own basis. Pure arithmetic over the settings: no model call, no files."""
    from .attacks.twins import generate_twins
    from .budget import HOLD_PER_CALL_MAX_USD, HOLDBACK_CALLS, PAID_MODEL, PAID_PRICES_USD_PER_TOKEN, worst_case_usd
    from .jds import relevance_labels, synthetic_jds
    from .metrics.detection import FPR_MIN_N
    from .probe import probe_remaining
    from .runner import _is_llm

    pool = [synthetic.make_resume(i, settings.seed) for i in range(settings.pool_size)]
    jds = synthetic_jds(settings.n_jds, settings.seed)
    rel0 = relevance_labels(jds[:1], {r.resume_id: r.gold_facts() for r in pool})[jds[0].profile_id]
    rel_i = [r for r in pool if rel0[r.resume_id] > 0]
    non_i = [r for r in pool if rel0[r.resume_id] <= 0]
    want_rel = min(len(rel_i), round(settings.n_bases * settings.relevant_fraction))
    want_non = min(len(non_i), settings.n_bases - want_rel)
    want_rel = min(len(rel_i), settings.n_bases - want_non)
    n_bases = want_rel + want_non
    n_twins = len(generate_twins(rel_i[:want_rel] + non_i[:want_non], settings.grid, settings.seed, settings.families,
                                 per_base=settings.twins_per_base))
    docs = len(pool) + n_twins
    doc_bytes = sum(len(r.to_text().encode("utf-8")) for r in pool) // max(len(pool), 1)
    p_in, p_out = PAID_PRICES_USD_PER_TOKEN[PAID_MODEL]

    def worst(stage: str) -> float:
        msgs = [{"role": "user", "content": "x" * (doc_bytes + STAGE_OVERHEAD_BYTES[stage] - 16)}]
        return worst_case_usd(PAID_MODEL, msgs, {}, STAGE_MAX_TOKENS[stage])

    def expected(stage: str) -> float:
        return ((doc_bytes + STAGE_OVERHEAD_BYTES[stage]) / 4 * p_in) + STAGE_EXPECTED_OUT_TOKENS[stage] * p_out

    calls = w_total = e_total = 0.0
    lines_sys: list[str] = []
    engine_docs = 0
    engine_w = engine_e = engine_c = 0.0
    stage_calls = {"l1": docs, "l2": docs * L2_CALLS_PER_DOC, "match": docs * MATCH_SHARE * len(jds)}
    for name in settings.systems:
        if name == "naive_llm":
            n = len(pool) * len(jds) + n_twins
            c, w, e = n, n * worst("naive"), n * expected("naive")
        elif _is_llm(name):
            c = docs * (1 + L2_CALLS_PER_DOC + MATCH_SHARE * len(jds))  # intake is cached across JDs
            w = docs * (worst("l1") + L2_CALLS_PER_DOC * worst("l2") + MATCH_SHARE * len(jds) * worst("match"))
            e = docs * (expected("l1") + L2_CALLS_PER_DOC * expected("l2") + MATCH_SHARE * len(jds) * expected("match"))
            if not engine_docs:
                engine_docs, engine_w, engine_e, engine_c = docs, w, e, c
        else:
            continue
        calls, w_total, e_total = calls + c, w_total + w, e_total + e
        lines_sys.append(f"  {name}: about {c:.0f} calls, worst case ${w:.2f}, expected ${e:.2f}")
    cap = settings.budget_usd if settings.allow_paid else 0.0
    resumed = False
    lines = [
        f"plan: pool {len(pool)} resumes ({len(rel_i)} relevant to the robustness JD); {n_bases} twin bases "
        f"({want_rel} relevant, {want_non} not); {n_twins} twins"
        + (f" ({settings.twins_per_base} per base)" if settings.twins_per_base else "")
        + f"; {len(jds)} JDs; systems {', '.join(settings.systems)}",
        f"statistics: ASR n = {n_twins} twins over {n_bases} base clusters; FPR n = {len(pool)} clean resumes "
        f"(a pass needs {FPR_MIN_N}); min-verdict-n {settings.min_n}",
        "model calls if nothing is cached:",
        *lines_sys,
        f"reserved worst case (what the cap reserves before a call settles): ${w_total:.2f} for the plan",
        f"expected billed (estimate): ${e_total:.2f} for the plan "
        f"(assumes chars/4 input tokens and typical output sizes at {PAID_MODEL} prices)",
    ]
    hold = (HOLDBACK_CALLS * min(max(engine_w - engine_e, 0.0) / engine_c, HOLD_PER_CALL_MAX_USD)
            if engine_c else 0.0)
    if cap > 0 and e_total > 0 and engine_docs:
        lines.append(f"headroom held back (the last {HOLDBACK_CALLS} paid calls stay at worst case, so lag in /key "
                     f"or a low usage block is covered): about ${hold:.2f}")
        e_plain = e_total
        e_total += hold  # the cap is checked against settled spend plus this headroom
        d_billed = engine_docs * min(cap / e_total, 1.0)
        d_worst = engine_docs * min(cap / w_total, 1.0) if w_total else engine_docs
        lines.append(
            f"cap ${cap:.2f} per run: each call settles to its billed cost, so the cap stops full_engine after about "
            f"{d_billed:.0f} of {engine_docs} documents ({'it can finish' if cap >= e_total else 'it cannot finish'}); "
            f"if the provider reports no usage the worst case stays reserved and the cap stops it after about "
            f"{d_worst:.0f} documents")
        if cap < e_total:
            lines.append(f"warning: full_engine cannot finish under ${cap:.2f}: it needs about ${e_total:.2f} "
                         f"({engine_docs} documents); shrink --pool-size, --n-bases or --twins-per-base")
        elif cap < w_total:
            lines.append("note: it finishes only while the provider returns usage for each call; a call without it "
                         "keeps its worst case reserved and the run may stop early")
        probe_model = PAID_MODEL if settings.model_policy == "paid-direct" else settings.free_models[0]
        cached = cached_answers(store_root, probe_model)
        if cached and engine_docs:
            probe = probe_remaining(settings, store_root, probe_model) if store_root is not None else None
            if probe is not None and "full_engine" in probe["systems"]:
                calls, resumed = _resumed_lines(lines, probe["systems"]["full_engine"], probe, cached, probe_model, cap,
                                                hold, expected, worst), True
            else:
                rem = {s: max(stage_calls[s] - cached.get(s, 0), 0.0) for s in stage_calls}
                rem_exp = sum(rem[s] * expected(s) for s in rem)
                calls, resumed = sum(rem.values()), True
                lines += [
                    f"cached answers in the store for {probe_model}: " + ", ".join(
                        f"{k} {cached.get(k, 0)}" for k in ("l1", "l2", "match")) + " (free on a resumed run)",
                    f"resumed run (ratio estimate, the exact probe was not available): about {calls:.0f} calls remain; "
                    f"expected ${rem_exp:.2f} + headroom ${hold:.2f}",
                ]
    lines.append(f"hard caps: ${cap:.2f} per run and ${PHASE_CAP_USD:.2f} for phase M1, measured from OpenRouter /key; "
                 f"--max-llm-calls {settings.max_llm_calls} stops after that many sent requests"
                 + (f" (the {'resumed run' if resumed else 'plan'} needs about {calls:.0f}, so it may stop early)"
                    if calls > settings.max_llm_calls else ""))
    if n_bases < settings.n_bases or want_rel < round(settings.n_bases * settings.relevant_fraction):
        lines.append("warning: the pool cannot supply the requested bases and relevant share; raise --pool-size.")
    if n_twins < settings.min_n:
        lines.append(f"warning: {n_twins} twins is below --min-verdict-n {settings.min_n}; the ASR verdict reads not measurable.")
    if len(pool) < FPR_MIN_N:
        lines.append(f"warning: {len(pool)} clean resumes is below {FPR_MIN_N}; the FPR verdict can only read fail or not measurable.")
    return chr(10).join(lines)


def _print_coverage(run_dir: Path, settings: CoreSettings) -> None:
    """Say so when the pool could not supply the requested bases or the relevant share."""
    import json

    try:
        cov = json.loads((run_dir / "raw.json").read_text(encoding="utf-8"))["settings"].get("relevance_coverage")
    except (OSError, ValueError, KeyError):
        return
    if not cov:
        return
    want_rel = round(settings.n_bases * settings.relevant_fraction)
    if cov["n_bases"] < settings.n_bases or cov["bases_relevant"] < want_rel:
        print(f"warning: asked for {settings.n_bases} bases ({want_rel} relevant) but got {cov['n_bases']} "
              f"({cov['bases_relevant']} relevant): the pool of {cov['pool_size']} holds {cov['pool_relevant']} "
              f"relevant resumes for the robustness JD. Raise --pool-size.", file=sys.stderr)


def _print_refusals(run_dir: Path) -> None:
    """Tell the user why a run stopped short: one line per refusal code, fixed key-free text."""
    import json

    try:
        raw = json.loads((run_dir / "raw.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for code, message in sorted((raw.get("refusals") or {}).items()):
        print(f"refused [{code}]: {message}", file=sys.stderr)


def _phase_open(args: argparse.Namespace) -> int:
    """Record /key usage as the phase baseline for this key (insert-once, by fingerprint) and print it for the PO to
    commit as PHASE_BASELINES_USD. Fails closed (writes nothing) when /key is unreadable."""
    from . import budget
    from .store import EvalStore

    try:
        usage = budget.fetch_key_usage()
        fp = budget.key_fingerprint()
    except budget.BudgetExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    code = budget.PHASE_BASELINES_USD.get(args.phase)
    if code is not None and usage < code - 1e-12:
        print(f"error: /key usage {usage:.6f} is below the committed {args.phase} baseline {code:.6f} "
              "(replaced key?); the baseline needs a reviewed change", file=sys.stderr)
        return 2
    with EvalStore.for_root(_project_root()) as store:
        stored = budget.StoreLedger(store).open_phase(args.phase, fp, usage, budget.PHASE_CAP_USD)
    print(f"phase {args.phase}: /key usage now {usage:.6f} USD; store baseline for this key {stored:.6f} USD")
    if code is None:
        print(f"set PHASE_BASELINES_USD[{args.phase!r}] = {stored:.6f} in evaluation/budget.py (a reviewed commit), "
              "log it in decisions.log, and commit the store; paid runs are refused until then")
    else:
        print(f"committed baseline {code:.6f}; spent on this key since the phase opened: {max(usage - code, 0):.6f} USD "
              f"of {budget.PHASE_CAP_USD:.2f}")
    return 0


def _skeleton(args: argparse.Namespace, runs_dir: Path) -> int:
    """Fairness (E10) has no runner yet. ``all`` runs the core suite and says so (E11 runs as ``--suite claims``)."""
    import uuid

    from .manifest import build_manifest, write_manifest

    if args.suite == "all":
        core = argparse.Namespace(**{**vars(args), "suite": "core"})
        rc = _run(core)
        print("note: fairness (E10) is not built and the claims suite (E11) is run with --suite claims; "
              "the report marks both not measurable",
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


def _run_ref(ref: str, root: Path) -> Path | str:
    """A run directory under ``runs/``, or a bare run id (looked up in the eval store when no directory exists)."""
    import re

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", ref):
        return ref  # a bare run id: compare_runs looks in the eval store first, then runs/<id> under the repo root
    return safe_output_dir(ref, root)


def _store(args: argparse.Namespace) -> int:
    from .store import EvalStore

    with EvalStore.for_root(_project_root()) as store:
        if args.action == "stats":
            print(f"{store.path.name}: {store.size_bytes() / 1048576:.2f} MB, schema v{store.schema_version}, "
                  f"{len(store.list_runs())} runs")
            return 0
        if not args.other:
            raise ValueError("store merge needs the other .sqlite3 file")
        added = store.merge_from(args.other)
    print("merged: " + ", ".join(f"{k} +{v}" for k, v in added.items()))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .report import compare_markdown, compare_runs

    root = _project_root()
    diff = compare_runs(_run_ref(args.run_a, root), _run_ref(args.run_b, root), args.tolerance, store_root=root)
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
        if args.command == "phase-open":
            return _phase_open(args)
        if args.command == "report":
            return _report(args)
        if args.command == "store":
            return _store(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "audit":
            return claims_cli.audit(args, _project_root())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2  # pragma: no cover


__all__ = ["SUITE_PARTS", "build_parser", "main"]
