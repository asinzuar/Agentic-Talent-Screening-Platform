"""Profile one or more resume-intake runs without changing the intake implementation.

The report deliberately excludes resume text, claims, security findings, and model
responses.  It is suitable for synthetic or consented test resumes only.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any
from unittest.mock import patch


SUPPORTED_RESUME_SUFFIXES = frozenset({".pdf", ".docx", ".txt"})


def _expand_resume_inputs(inputs: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Expand direct resume paths and the immediate supported files in directories."""
    resumes: list[Path] = []
    empty_or_missing: list[Path] = []
    seen: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(
                (
                    entry
                    for entry in input_path.iterdir()
                    if entry.is_file() and entry.suffix.lower() in SUPPORTED_RESUME_SUFFIXES
                ),
                key=lambda entry: entry.name.lower(),
            )
            if not candidates:
                empty_or_missing.append(input_path)
                continue
        else:
            empty_or_missing.append(input_path)
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                resumes.append(candidate)
                seen.add(resolved)
    return resumes, empty_or_missing


def _slowest_functions(stats: pstats.Stats, limit: int) -> list[dict[str, Any]]:
    """Return the slowest functions by cumulative time in a JSON-safe form."""
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, own_seconds, cumulative_seconds, _ = values
        rows.append(
            {
                "function": f"{Path(filename).name}:{line} ({function})",
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "own_seconds": round(own_seconds, 6),
                "cumulative_seconds": round(cumulative_seconds, 6),
            }
        )
    return sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:limit]


def _summary(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate successful runs without mixing in failed/incomplete invocations."""
    completed = list(runs)
    elapsed = [run["total_elapsed_seconds"] for run in completed]
    statuses = Counter(run["result"]["status"] for run in completed)
    model_calls: dict[str, dict[str, float | int]] = {}
    for stage in ("l1_semantic", "l2_extraction"):
        call_durations = [
            duration
            for run in completed
            for duration in run["model_calls"][stage]["seconds"]
        ]
        model_calls[stage] = {
            "total_calls": len(call_durations),
            "total_seconds": round(sum(call_durations), 6),
            "mean_seconds_per_call": round(mean(call_durations), 6) if call_durations else 0.0,
            "mean_seconds_per_resume": round(
                sum(call_durations) / len(completed), 6
            ) if completed else 0.0,
        }
    return {
        "successful_runs": len(completed),
        "total_elapsed_seconds": round(sum(elapsed), 6),
        "latency_seconds": {
            "mean": round(mean(elapsed), 6) if elapsed else 0.0,
            "median": round(median(elapsed), 6) if elapsed else 0.0,
            "minimum": round(min(elapsed), 6) if elapsed else 0.0,
            "maximum": round(max(elapsed), 6) if elapsed else 0.0,
        },
        "result_status_counts": dict(sorted(statuses.items())),
        "model_calls": model_calls,
    }


def _profile_path(profile_dir: Path, resume: Path, index: int) -> Path:
    """Keep a separate standard cProfile record for every measured resume."""
    return profile_dir / f"{index:03d}-{resume.stem}.prof"


def profile_resume(
    resume: Path,
    model: str | None,
    l1b_model: str | None,
    stop_after_l2: bool,
    profile_path: Path,
    top: int,
) -> dict[str, Any]:
    """Run the existing public intake entry point and return timing metadata."""
    # Import only for a real profiling run so ``--help`` works even before the
    # project has been installed.  ``uv run`` installs it before normal use.
    from agentic_talent_screener.resume_intake import l1_injection, pipeline

    model_call_seconds: dict[str, list[float]] = {"l1_semantic": [], "l2_extraction": []}
    original_l1_chat_json: Callable[..., Any] = l1_injection.chat_json
    original_l2_call_model: Callable[..., Any] = pipeline.call_model

    def timed_l1_chat_json(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        try:
            return original_l1_chat_json(*args, **kwargs)
        finally:
            model_call_seconds["l1_semantic"].append(time.perf_counter() - started_at)

    def timed_l2_call_model(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        try:
            return original_l2_call_model(*args, **kwargs)
        finally:
            model_call_seconds["l2_extraction"].append(time.perf_counter() - started_at)

    profiler = cProfile.Profile()
    started_at = time.perf_counter()
    with (
        patch.object(l1_injection, "chat_json", timed_l1_chat_json),
        patch.object(pipeline, "call_model", timed_l2_call_model),
    ):
        result = profiler.runcall(
            pipeline.normalize_resume,
            resume,
            model,
            l1b_model=l1b_model,
            show_progress=False,
            stop_after_l2=stop_after_l2,
        )
    elapsed_seconds = time.perf_counter() - started_at

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler)
    return {
        "resume_filename": resume.name,
        "total_elapsed_seconds": round(elapsed_seconds, 6),
        "model_calls": {
            stage: {
                "count": len(seconds),
                "seconds": [round(value, 6) for value in seconds],
                "total_seconds": round(sum(seconds), 6),
            }
            for stage, seconds in model_call_seconds.items()
        },
        "result": {
            "status": result.status,
            "stopped_at": result.stopped_at,
            "requires_human_review": result.requires_human_review,
        },
        "cprofile_file": str(profile_path),
        "slowest_functions_by_cumulative_time": _slowest_functions(stats, top),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile existing resume intake for one or more synthetic/consented resumes, sequentially."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Resume files or folders containing immediate PDF, DOCX, or TXT resume files",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model ID (default: ATS_LLM_MODEL)")
    parser.add_argument("--l1b-model", default=None, help="L1 semantic-judge model (default: --model)")
    parser.add_argument(
        "--stop-after-l2",
        action="store_true",
        help="Stop at schema-validated L2 output, matching the pipeline development mode.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of slow functions to report (default: 20)")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("artifacts/profiles/resume-intake"),
        help="Directory for one standard-library cProfile file per resume.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional JSON timing summary path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    resumes, invalid_inputs = _expand_resume_inputs(args.inputs)
    if invalid_inputs:
        parser.error(
            "Input path(s) do not exist or contain no supported resume files: "
            + ", ".join(str(path) for path in invalid_inputs)
        )

    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, resume in enumerate(resumes, start=1):
        try:
            runs.append(
                profile_resume(
                    resume,
                    args.model,
                    args.l1b_model,
                    args.stop_after_l2,
                    _profile_path(args.profile_dir, resume, index),
                    args.top,
                )
            )
        except Exception as error:
            errors.append(
                {"resume_filename": resume.name, "error": f"{type(error).__name__}: {error}"}
            )

    report = {
        "requested_runs": len(resumes),
        "summary": _summary(runs),
        "runs": runs,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2)
    if args.report_out is None:
        print(rendered)
    else:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote timing report to {args.report_out}")
    if not runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
