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
| `--pool-size N` | Synthetic resumes (default 150). This is the clean set: the false-positive rate (FPR) n. A pass needs 100 or more (`FPR_MIN_N`). Only about 20 to 30% are relevant to the robustness JD (20 resumes held 4 relevant, 80 about 22, 150 about 34), so the pool must also be big enough for the relevant bases. |
| `--n-bases N` | Resumes that get attack twins. For the attack success rate (ASR), n is the number of twins; the bases are the clusters of the clustered interval. |
| `--relevant-fraction F` | Share of the bases taken from resumes relevant to the robustness JD (default 0.5). Capped by what the pool holds; the run prints a warning when the pool is too small. |
| `--twins-per-base N` | N grid cells per base instead of all of them (rotating, so every cell is used about equally). 30 bases with 1 twin each is 30 twins in 30 clusters. |
| `--n-jds N` | Synthetic JDs for ranking. Ranking n is the number of JDs, so a ranking verdict needs 30 or more. |
| `--families A1,...` / `--grid core\|full` | Attack families and grid. Twins = bases x cells: the core grid is 32 per base, one family such as `A1` is 4 per base. Use one family: two halve the twins per family without raising any verdict's n. |
| `--min-verdict-n N` | Minimum n for any verdict (default 30). |

A small run (`--pool-size 20 --n-bases 4`) therefore reads `not measurable (n<min)` on purpose. A 30-twin ASR with no hits still has a Wilson upper bound of about 11%, so a pass against 5% needs about 75 or more twins.

### Routing, spend and outages
* `--model-policy free-first` (default): two free models, then the allow-listed paid model once both have each hit `--rate-limit-switch-after` consecutive 429s (state is per run, shared by all systems and calls) and only with `--allow-paid`. Only a real 429 counts: an HTTP 429, or an HTTP-200 body whose `error.code` is 429. A 502 or 400 never opens paid spend.
* `--model-policy paid-direct`: skip the free models and use the allow-listed paid model from the first call. It needs `--allow-paid` and a budget; every cap still applies (per run, phase M1 and baseline checks, base URL pin). The manifest and report record the policy.
* **Spend accounting.** Before each paid call the run reserves its worst case (the input bytes plus `max_tokens`, times two attempts). When the call ends, the reservation is settled to the billed cost from the provider's `usage` block (`usage.cost`, or token counts times the allow-listed price). If the usage is missing or malformed the call keeps its full worst case. The run's counted spend is the larger of settled costs plus unsettled reservations and the `/key` delta since the run started, so a usage block that lies low is still bounded by the provider's own counter. The phase M1 rule (the larger of ledger and key views) is unchanged. `raw.json` records `reserved_worst_usd`, `settled_usd` and the measured delta; the ledger row holds the counted reservation as its estimate and the `/key` delta as its measurement.
* **Headroom.** The last 40 settled paid calls stay at their worst case on top of the counted spend, so a lagging `/key` counter or an under-reporting usage block is covered (tested with lag probes at 0, 10 and 100 reads, usage lying low). **Trust boundary:** the $0.50 per-run cap is strict unless OpenRouter both under-reports usage and its `/key` counter lags by more than about 125 calls; the worst overrun seen in testing was about $0.62. The $1.50 phase M1 cap (which the next run's first `/key` read enforces) and the provider-side key limit remain the backstops. You can set the key's credit limit in OpenRouter (about $1.60) as a hard stop on the provider's side. Any attempt that raised or got no response (a timeout) has no usage block, so its call keeps its full worst case. A usage block reporting 0 is floored at $0.000001 per attempt, or at the token-count price when tokens are present.
* A system whose model calls mostly failed with no output (rate limit, provider error, timeout; at least as many failed as were answered) and fewer than `--min-verdict-n` calls were answered reads `not measurable (provider outage: N/M calls failed, K answered)` on every metric. One transient error, or many answered-but-wrong outputs, never reads as an outage.
* `compare A B` takes bare run ids: the eval store is searched first, then `runs/<id>` under the repo root, whatever the working directory.

### `run --dry-run`
Prints the plan and an estimate on the cap's own basis, each labelled: the reserved worst case, the expected billed cost, and where the cap stops the run, with a warning when a system cannot finish. It makes no model call. It is an estimate; the caps are `--budget-usd`, the phase M1 cap and `--max-llm-calls`. Its assumptions are in `cli.py` (per-stage `max_tokens` 1536, 4096 and 20000, a prompt overhead per stage, matching on about 30% of documents).

### Example: first numbers, paid-direct (pool 100, 20 bases x A1 = 80 twins)
```
# 1. plan and estimate (no calls, no spend)
uv run python -m agentic_talent_screener.evaluation run --systems full_engine --families A1 \
  --pool-size 100 --n-bases 20 --relevant-fraction 0.5 --n-jds 1 --max-llm-calls 500 \
  --allow-paid --budget-usd 0.50 --model-policy paid-direct --run-id first-numbers --dry-run
# 2. the same command without --dry-run (needs `phase-open` done once)
# 3. the claims suite is free-only (no paid path); run it separately
uv run python -m agentic_talent_screener.evaluation run --suite claims --pool-size 100 --n-bases 20 --n-jds 1 --run-id first-numbers-claims
```
Step 1 prints (this is the output of the command above):
```
plan: pool 100 resumes (26 relevant to the robustness JD); 20 twin bases (10 relevant, 10 not); 80 twins; 1 JDs; systems full_engine
statistics: ASR n = 80 twins over 20 base clusters; FPR n = 100 clean resumes (a pass needs 100); min-verdict-n 30
model calls if nothing is cached:
  full_engine: about 414 calls, worst case $2.71, expected $0.23
reserved worst case (what the cap reserves before a call settles): $2.71 for the plan
expected billed (estimate): $0.23 for the plan (assumes chars/4 input tokens and typical output sizes at qwen/qwen3.8-flash prices)
headroom held back (the last 40 paid calls stay at worst case, so lag in /key or a low usage block is covered): about $0.24
cap $0.50 per run: each call settles to its billed cost, so the cap stops full_engine after about 180 of 180 documents (it can finish); if the provider reports no usage the worst case stays reserved and the cap stops it after about 33 documents
note: it finishes only while the provider returns usage for each call; a call without it keeps its worst case reserved and the run may stop early
hard caps: $0.50 per run and $1.50 for phase M1, measured from OpenRouter /key; --max-llm-calls 500 stops after that many sent requests
```
naive_llm is left out of the paid run: ranking and lift stay not measurable at one JD whatever is spent. If the provider returns no usage block, each call keeps its worst case and the cap stops the run after about 33 documents with no full_engine result. Raising the cap or the budget above $0.50 needs approval.
