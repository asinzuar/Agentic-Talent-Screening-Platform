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
