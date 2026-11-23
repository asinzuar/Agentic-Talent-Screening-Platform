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

## Running the evaluation tests
* Before merging any change under `evaluation/`, run `uv run pytest` (the whole suite, about 2 minutes, fully offline, no network or API key).
* `tests/evaluation/test_regression.py` is the offline regression gate. It runs a tiny deterministic suite (keyword, TF-IDF, BM25, naive screener and the full pipeline on a fake chat) over all 11 attack families and compares the key numbers with the pinned `tests/evaluation/baseline/results.json`. Any change fails the gate. After an intended change, refresh the baseline with `UPDATE_REGRESSION_BASELINE=1 uv run pytest tests/evaluation/test_regression.py` and review the diff.
* This is a manual pre-merge step. No CI workflow exists yet, by decision, so nothing runs the gate on each pull request.

## Sizing a live run (pool, relevant share, JDs, twins)
Options of `run` (see `run --help`):

| Option | Meaning |
| --- | --- |
| `--pool-size N` | Synthetic resumes (default 150). Only about 20 to 30% are relevant to the robustness JD (20 resumes held 4 relevant, 80 hold about 22, 150 about 34), so size the pool at roughly 4x the relevant resumes you need. |
| `--n-bases N` | Resumes that get attack twins. These are the clean resumes of the robustness and detection splits, so this is the `n` behind those verdicts: use at least `--min-verdict-n` (30). |
| `--relevant-fraction F` | Share of the bases taken from resumes relevant to the robustness JD (default 0.5). Capped by what the pool holds; the run prints a warning when the pool is too small. |
| `--n-jds N` | Synthetic JDs for ranking. Ranking `n` is the number of JDs, so a ranking verdict needs 30 or more. |
| `--families A1,...` / `--grid core\|full` | Attack families and grid. Twins = bases x cells: the core grid is 32 per base, so all families on 30 bases is 960 twins and thousands of model calls. One family (`A1`) is 4 per base. |
| `--min-verdict-n N` | Minimum n for any verdict (default 30). |

A small run (`--pool-size 20 --n-bases 4`) therefore reads `not measurable (n<min)` on purpose: 4 bases, 8 twins, 4 relevant resumes.

`run --dry-run` prints the plan and a rough paid-cost estimate (assumptions are printed with it) and exits without any model call. It is an estimate, never a cap; the caps are `--budget-usd` (per run), the phase M1 cap and `--max-llm-calls`.

### Routing and outages
* `--model-policy free-first` (default): two free models, then the allow-listed paid model once both have each hit `--rate-limit-switch-after` consecutive 429s (state is per run, shared by all systems and calls) and only with `--allow-paid`.
* `--model-policy paid-direct`: skip the free models and use the allow-listed paid model from the first call. It needs `--allow-paid` and a budget; every cap still applies (per run, phase M1 and baseline checks, base URL pin). The manifest and report record the policy.
* A system whose model calls failed with no output (rate limit, provider error, timeout) and fewer than `--min-verdict-n` calls succeeded reads `not measurable (provider outage: N/M calls failed)` on every metric and slide-9 row. It never reads pass or fail.
* `compare A B` takes bare run ids: the eval store is searched first, then `runs/<id>` under the repo root, whatever the working directory.

### Example: first numbers, paid-direct, about 30 clean resumes
```
# 1. plan and estimate (no calls, no spend)
uv run python -m agentic_talent_screener.evaluation run --systems bm25,full_engine --families A1 \
  --pool-size 80 --n-bases 30 --relevant-fraction 0.5 --n-jds 1 --max-llm-calls 650 \
  --allow-paid --budget-usd 0.50 --model-policy paid-direct --run-id first-numbers --dry-run
# 2. the same command without --dry-run (opens spend; needs `phase-open` done once)
# 3. the claims suite is free-only (no paid path), run it separately
uv run python -m agentic_talent_screener.evaluation run --suite claims --pool-size 80 --n-bases 30 --n-jds 1 --run-id first-numbers-claims
```
The estimate for step 1 is about 800 uncached model calls, of which at most 650 are sent (about $0.45 at the allow-listed price); this is a rough figure from stated assumptions. If the call cap is reached the run is partial and says so. Raising the cap or the budget above $0.50 needs approval.
