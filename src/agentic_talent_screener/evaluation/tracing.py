"""E12 tracing at the eval boundary (no C1 / no llm.py change).

``TracedChat`` wraps E1's ``CachedChat`` and logs one metadata record per LLM call;
``TracedSystem`` wraps an E5 system and logs one record per resume. Records go to
``runs/<run>/traces.jsonl`` (path validated with ``safe_output_dir``).

Privacy: records carry only ids, counts, timings and error CLASS names. Never prompt or
response text, error messages, PII or keys. Resume ids are hashed to a short ref.

What this boundary can and cannot see (C1 in ``llm.py`` would close the gaps; see
``evaluation/EFFICIENCY.md``):
* ``attempts`` counts invocations of the underlying call made for this one call (1 on a
  miss). Retries inside ``llm.py`` are invisible here.
* Token counts are ESTIMATES from character counts, not provider usage.
* ``call_site`` is the schema name (the only call-site signal visible at the boundary).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .budget import BudgetExceeded
from .cache import CacheMiss, CachedChat
from .datasets.synthetic import safe_output_dir

CHARS_PER_TOKEN = 4  # rough heuristic; labelled as an estimate everywhere
TRACE_FILE = "traces.jsonl"

#: failures that invalidate a run rather than describe model behaviour
INFRASTRUCTURE_KINDS = frozenset({"budget_guard", "cache_miss", "infrastructure"})
#: failures where the model returned nothing to validate (not schema failures)
NO_OUTPUT_KINDS = frozenset({"transport", "timeout", "provider_error", "other"})


def error_kind(exc: BaseException) -> str:
    """Class-level failure kind only; the message is never recorded."""
    from .systems import InfrastructureError  # lazy: systems imports the pipeline

    if isinstance(exc, InfrastructureError):
        return "infrastructure"
    if isinstance(exc, BudgetExceeded):
        return "budget_guard"
    if isinstance(exc, CacheMiss):
        return "cache_miss"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "transport"
    if isinstance(exc, ValueError):  # llm.py: invalid JSON / non-object; pydantic errors subclass it
        return "invalid_output"
    if isinstance(exc, RuntimeError):
        return "provider_error"
    return "other"


def ref(value: str, salt: str = "") -> str:
    """Short salted hash of a resume id; the salt is per run and never written to the trace."""
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:10]


def estimate_tokens(chars: int) -> int:
    return -(-max(chars, 0) // CHARS_PER_TOKEN)


def check_schema(result: Any, schema: dict[str, Any]) -> bool:
    """Boundary check: top-level object, required keys present, no extras when forbidden,
    and basic JSON types of declared properties. A cheap proxy, not a full validator."""
    if not isinstance(result, dict):
        return False
    if any(k not in result for k in schema.get("required", [])):
        return False
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False and any(k not in props for k in result):
        return False
    types = {"string": str, "boolean": bool, "array": list, "object": dict, "integer": int, "number": (int, float)}
    for k, spec in props.items():
        t = spec.get("type") if isinstance(spec, dict) else None
        if k in result and isinstance(t, str) and t in types:
            v = result[k]
            if isinstance(v, bool) and t in ("integer", "number"):
                return False
            if not isinstance(v, types[t]):
                return False
    return True


class TraceSink:
    """Append-only JSONL writer plus an in-memory copy for the report."""

    def __init__(self, run_dir: str | Path, base: str | Path | None = None) -> None:
        self.run_dir = safe_output_dir(run_dir, base)
        runs_root = (Path(base).resolve() if base is not None else Path.cwd().resolve()) / "runs"
        if runs_root.resolve() not in self.run_dir.parents:
            raise ValueError("trace output must stay under runs/<run>")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / TRACE_FILE
        if self.path.is_symlink():
            self.path.unlink()
        self.path.write_text("", encoding="utf-8")  # start empty so the file and ``records`` agree
        self.salt = secrets.token_hex(8)
        self.records: list[dict[str, Any]] = []
        self.system: str | None = None
        self.resume_ref: str | None = None
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        record = {**record, "system": self.system}
        with self._lock:
            self.records.append(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_traces(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / TRACE_FILE
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


class TracedChat(CachedChat):
    """CachedChat that records metadata for every call. Drop-in for ``build_systems``."""

    def __init__(self, chat: CachedChat, sink: TraceSink) -> None:
        self._sink = sink
        self._underlying = chat._call
        self._invocations = 0
        super().__init__(
            chat.cache, chat.guard, offline=chat.offline, call=self._counting_call,
            default_model=chat._default_model,
        )

    def _counting_call(self, *a: Any, **kw: Any) -> dict[str, Any]:
        self._invocations += 1
        call = self._underlying
        if call is None:
            from agentic_talent_screener.llm import chat_json as call
        return call(*a, **kw)

    def __call__(self, messages: list[dict[str, str]], *, schema: dict[str, Any], schema_name: str,
                 model: str | None = None, **kw: Any) -> dict[str, Any]:
        hits_before, inv_before = self.cache.hits, self._invocations
        started = time.perf_counter()
        result: dict[str, Any] | None = None
        err: str | None = None
        try:
            result = super().__call__(messages, schema=schema, schema_name=schema_name, model=model, **kw)
            return result
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised unchanged
            err = error_kind(exc)
            raise
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            hit = self.cache.hits > hits_before
            prompt_chars = sum(len(str(m.get("content", ""))) for m in messages) + len(json.dumps(schema))
            out_chars = len(json.dumps(result, ensure_ascii=False)) if result is not None else 0
            infra = err in INFRASTRUCTURE_KINDS
            if infra:
                prompt_chars = 0  # never sent: keep out of modelled cost
            # schema_valid is only defined when the model returned output to validate
            if infra or err in NO_OUTPUT_KINDS:
                valid = None
            else:
                valid = err is None and check_schema(result, schema)
            try:
                resolved = model or (self._default_model() if self._default_model else "default")
            except Exception:  # noqa: BLE001 - model id is metadata only
                resolved = "default"
            self._sink.emit({
                "type": "call",
                "call_site": schema_name,
                "model": resolved,
                "resume_ref": self._sink.resume_ref,
                "latency_ms": latency_ms,
                "attempts": self._invocations - inv_before,
                "cache": "hit" if hit else "miss",
                "schema_valid": valid,
                "error_kind": err,
                "est_prompt_tokens": estimate_tokens(prompt_chars),
                "est_completion_tokens": estimate_tokens(out_chars),
                "tokens_basis": "estimated_from_chars",
            })


class TracedSystem:
    """Wraps an E5 system and records per-resume wall time and outcome.

    ``per_candidate=True`` scores one candidate per inner call (exact per-resume timing and
    call attribution); only valid for systems whose score for a resume does not depend on the
    rest of the pool (the naive LLM screener). Otherwise the pool is scored at once and each
    resume gets the pool mean (``timing_basis: pool_mean``)."""

    def __init__(self, inner: Any, sink: TraceSink, *, per_candidate: bool = False) -> None:
        self.inner, self.sink, self.per_candidate = inner, sink, per_candidate
        self.name = inner.name

    def score(self, candidates: Sequence[Any], jd: Any) -> Any:
        from .systems import SystemOutput

        self.sink.system = self.name
        if not candidates:
            return self.inner.score(candidates, jd)
        if self.per_candidate:
            merged = SystemOutput()
            for c in candidates:
                self.sink.resume_ref = ref(c.candidate_id, self.sink.salt)
                t0 = time.perf_counter()
                start_idx = len(self.sink.records)
                try:
                    out = self.inner.score([c], jd)
                except BaseException as exc:  # noqa: BLE001
                    self._resume(c, t0, "error", error_kind(exc), start_idx, "exact")
                    raise
                finally:
                    self.sink.resume_ref = None
                merged.scores.update(out.scores)
                merged.unranked.update(out.unranked)
                status = "ranked" if c.candidate_id in out.scores else "unranked"
                self._resume(c, t0, status, None, start_idx, "exact")
            return merged
        t0 = time.perf_counter()
        start_idx = len(self.sink.records)
        try:
            out = self.inner.score(candidates, jd)
        except BaseException as exc:  # noqa: BLE001
            self._resume(None, t0, "error", error_kind(exc), start_idx, "pool_mean", n=len(candidates))
            raise
        for c in candidates:
            status = "ranked" if c.candidate_id in out.scores else "unranked"
            self._resume(c, t0, status, None, start_idx, "pool_mean", n=len(candidates))
        return out

    def _resume(self, c: Any, t0: float, status: str, err: str | None, start_idx: int, basis: str,
                n: int = 1) -> None:
        wall_ms = (time.perf_counter() - t0) * 1000
        calls = [r for r in self.sink.records[start_idx:] if r["type"] == "call"]
        stage: dict[str, float] = {}
        for r in calls:
            stage[r["call_site"]] = round(stage.get(r["call_site"], 0.0) + r["latency_ms"] / n, 3)
        self.sink.emit({
            "type": "resume",
            "resume_ref": ref(c.candidate_id, self.sink.salt) if c is not None else None,
            "wall_ms": round(wall_ms / n, 3),
            "timing_basis": basis,
            "status": status,
            "error_kind": err,
            "calls": round(len(calls) / n, 3),
            "pool_size": n,
            "live_calls": sum(1 for r in calls if r["cache"] == "miss" and r["error_kind"] not in INFRASTRUCTURE_KINDS),
            "cache_hits": sum(1 for r in calls if r["cache"] == "hit"),
            "stage_ms": stage,
        })


def trace_systems(systems: dict[str, Any], sink: TraceSink) -> dict[str, TracedSystem]:
    """Wrap a ``build_systems`` registry. Only the naive screener is safe to time per candidate."""
    return {k: TracedSystem(s, sink, per_candidate=(k == "naive_llm")) for k, s in systems.items()}


__all__ = ["TraceSink", "TracedChat", "TracedSystem", "check_schema", "error_kind", "load_traces", "trace_systems"]
