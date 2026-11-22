# E12: efficiency, cost and agent-reliability from local traces

`tracing.py` records metadata at the eval boundary (no `llm.py` change, C1 not signed off):
`TracedChat(cached_chat, sink)` logs one `call` record per LLM call, `TracedSystem` one `resume`
record per resume, into `runs/<run>/traces.jsonl` (path checked by `safe_output_dir`).
`metrics/efficiency.py::summarize_all(records, PriceTable(...))` returns one summary and verdict set per system (cost, latency and reliability are never pooled across systems).

Usage:

    sink = TraceSink("runs/<run>")
    chat = TracedChat(cached_chat, sink)
    systems = trace_systems(build_systems(chat), sink)
    ...score...; summaries = summarize_all(sink.records)  # one per system; never pooled

Never logged: prompt or response text, error messages, PII, keys. Resume ids are hashed.

## Labels you must keep
* Cost is $0 measured (free models only). The INR/resume figure is **modelled, not measured**:
  estimated tokens (chars/4) x a configurable price table (default: paid Gemma 4 31B,
  $0.10/M in, $0.34/M out, architecture.md section 5) x an assumed 85 INR/USD.
* Latency verdict is `not measurable` for pooled systems (only a pool mean exists), and when fewer than half the resumes were timed per-resume and live; the report shows how many qualified. A timed-out resume fails it.
* Schema-valid is judged on live (cache-miss) calls only; replayed outputs are biased (failures are not cached) and reported separately. Transport, timeout and provider errors are failure kinds, not schema failures.
* Retry rate is `not measurable until C1`.
* `schema_valid` is a boundary check (type, required keys, no extra keys), a proxy for the
  validation inside `llm.py`. Resume ids are hashed with a per-run salt; traces are restricted to `runs/`.
* Tool-call correctness is N/A (no agent has tools).

## What C1 (llm.py `LLMCallEvent` observer) would add
* Real `prompt_tokens` / `completion_tokens` (provider `usage`) replacing the estimate.
* True `attempts` and HTTP status from the retry loop in `llm.py` (transport retries, 429s).
* Per-attempt latency, the real `call_site`, and schema validation as `llm.py` sees it.
* Provider-reported cost when available.

Until then the numbers above stay estimates.

## Verdict rules (minimum sample size)
Every verdict carries `n` and `ci95`. `min_n` is `EvalConfig.min_verdict_n` (default 30), passed to `summarize(..., min_n=)`.
* A **fail** is allowed below `min_n` only when already proven: an observed resume over 20 s, a schema-valid Wilson upper bound below 99%, or a modelled-cost CI lower bound at or above INR 1.
* Below `min_n` otherwise: `not measurable (n<min)`.
* **Schema-valid** (target 99%): pass needs `n >= min_n` and the Wilson 95% **lower bound** >= 99% (about 380 all-valid live calls). With enough data but a CI that spans the target the status is `inconclusive`.
* **Modelled cost** (target INR 1): `ci95` is a bootstrap over calls times calls per resume (modelled). Pass needs `n >= min_n` and the CI upper bound below the target; `inconclusive` if the CI spans it.
* **Latency** (target max under 20 s): pass needs `n >= min_n` per-resume-timed, non-replayed resumes and max under 20 s. `ci95` is the Wilson interval of the share of resumes under 20 s (information; the rule itself is on the max).
