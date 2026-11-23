# The eval store (`eval_store.sqlite3`)

One SQLite file at the repo root, committed to git, so a teammate can clone and re-run `report` / `compare`
with zero API calls. Code: `evaluation/store.py`. Size budget: 20 MB (a warning at close, not an error).

## It is public and synthetic-only

Anyone who can read the repo can read this file. Nothing secret and nothing about a real person goes in:

- only synthetic prompts and their responses, hashes, vectors, run summaries and spend totals;
- the store itself refuses a cache write when the caller did not pass `synthetic=True` (the default is refuse), and
  refuses any prompt or response that contains a key shape (`sk-or-`, `sk-ant-`, `ghp_...`, `github_pat_...`, `AKIA...`).
  The check runs inside `EvalStore.put_llm`, so every route (including E15a's routed calls) goes through it;
- big per-item raw files (`raw.json`, `traces.jsonl`, `per_item.jsonl`) stay in git-ignored `runs/`.

## What is in it

| Table | Holds |
|---|---|
| `embeddings` | `model_id`, `text_sha256`, `dim`, float32 vector blob |
| `llm_cache` | key (hash of model, schema, messages, options), model, schema name, response JSON |
| `runs` | per-run `manifest` and `results` summary JSON, git sha (the runner writes one row per run) |
| `spend_ledger` | append-only spend rows with a unique `entry_id`; E15a writes them |

`phase_total(phase)` returns USD as a float (sum of `max(usd_measured, usd_estimated)` per row);
`phase_summary(phase)` gives the breakdown. `EvalStore.batch()` groups many writes into one transaction.
The file uses WAL while open (fast writes); `close()` checkpoints, returns to rollback-journal mode and VACUUMs, so
only the `.sqlite3` file is left. `-wal`, `-shm` and `-journal` are git-ignored; the DB is not. `.gitattributes`
marks it `binary`.

## Using it

- `compare` takes a run directory, a path, or a bare run id that is looked up in `runs/<id>` and then in the store.
- `--cache-dir` is deprecated and ignored (the path is only validated). Responses go to the root store.
  Old JSON entries under `data/eval/cache/` are no longer read.
- `python -m agentic_talent_screener.evaluation store stats` prints size and run count.
- Commit the store at milestones, not after every run: a binary file adds close to its full size to history each time.

## Merging two copies (do not use git's binary merge)

Git cannot merge a SQLite file. "Take ours" or "take theirs" silently drops the other side's rows, and a dropped
`spend_ledger` row undercounts spend. Use the union command instead:

```
git show :3:eval_store.sqlite3 > theirs.sqlite3     # their version, during the conflicted merge
git checkout --ours eval_store.sqlite3              # keep ours
python -m agentic_talent_screener.evaluation store merge theirs.sqlite3
git add eval_store.sqlite3
```

`store merge` unions every table with `INSERT OR IGNORE` (cache, embeddings and runs by key; the ledger by
`entry_id`), so it is lossless and idempotent. It refuses a store whose schema version differs from the code's.
E15a also keeps a `/key` baseline at the start of a phase, so spend stays bounded by the provider's own counter
even if rows are lost.

## Phase baselines (E15a, schema v2)

`phase_baselines (phase, key_fp, baseline_usd, cap_usd, opened_at)` holds OpenRouter's `/key` usage when a spend phase
opened, per key **fingerprint** (a salted sha256 prefix; never the key). Rows are insert-once; the only allowed UPDATE
lowers `baseline_usd`, which `store merge` uses to keep the lower baseline when two stores opened the same phase.
The figures are public, like the other spend totals. The authoritative baseline is the reviewed constant
`PHASE_BASELINES_USD` in `budget.py` (set from the output of `phase-open`); the store row can only lower it, so a
deleted, reverted or forked store cannot reset the $1.50 M1 total.

## The embedding baseline (E15c)

`embedding` is the deck's "plain embedding match": cosine(JD query, raw resume text), no LLM, no API.

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, about 90 MB, 384-d) through `transformers`
  `AutoTokenizer` / `AutoModel`, attention-masked mean pooling, L2-normalised. CPU only, batches of 16, text truncated
  to the model's 256 tokens (a long resume is read from its start only).
- **Revision:** pinned to commit `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` (`embedding.MODEL_REVISION`). The store key
  is `model_id = <name>@<revision>` plus `sha256(text)`, so a new revision never reuses old vectors.
- **Cache:** the Hugging Face cache is `$HF_HOME`, default `<repo>/.tmp/hf` (git-ignored). Never put it on C:. The
  first live use downloads the model once, from huggingface.co only; nothing else is fetched.
- **Vectors** go in the `embeddings` table (384 float32 = 1536 bytes each), written only from synthetic runs.
  A re-run and the offline gate read them back and never recompute or download.
- **Offline:** `--offline` with a missing vector gives status `not_run`, reason `embedding not cached`. It is never
  scored as zero, never a crash, and the slide 9 lift is then not confirmed (it needs keyword and embedding both).
- **Warm the cache:** run once without `--offline`, then commit the store.
- Needs `torch` at runtime (already in the lock through docling).
