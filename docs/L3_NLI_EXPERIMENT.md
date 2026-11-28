# L3 Qwen 3B experiment

Train on a Colab NVIDIA GPU; run the exported model in this local project through
Ollama. This is a college experiment, not an established accuracy improvement.
The code/notebook alone is not a trained model. No GPU training or live model
evaluation has been completed by adding these files.

## 1. Train and export

Commit/push your implementation branch and open `notebooks/l3_qwen_qlora.ipynb`
in Colab. Set `REPO_REF`, select a GPU, mount Drive, and execute the cells in order.
The notebook uses `uv` and the optional `l3-train` group without installing the
whole application. It never needs your `.env` or OpenRouter key.
If `uv` is absent, the setup cell bootstraps just `uv` from PyPI using the
notebook's Python interpreter. This avoids the standalone installer's URL, which
can return HTTP 403 in Colab. All project dependencies still go through `uv`.

The setup cell's `run_module` helper streams subprocess stdout/stderr into the
notebook and saves each invocation under `/content/ats-l3-logs`. A bare
`CalledProcessError` from an older helper only means the child failed: the
underlying traceback is required before changing packages or training settings.
If a notebook is already open in Colab, replace/re-execute its helper definition
to get this logging; editing the local notebook does not update that live session.
Keep checkpoints if a retry reports a non-empty run directory. Use `--resume`
for a real saved checkpoint, or a new run name for a fresh attempt; do not erase
the original run to hide a startup error.
An unsuccessful model-loading attempt may leave only `run_config.json`. Retrying
that directory is permitted when the configuration is identical. Any checkpoint,
adapter, or other artifact still prevents an accidental fresh training run there.

Training uses `Qwen/Qwen2.5-3B-Instruct` (Qwen research licence) and
[WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) (CC BY 4.0).
Cite Liu et al., *WANLI: Worker and AI Collaboration for Natural Language
Inference Dataset Creation*, EMNLP Findings 2022. Inspect both sources' terms
before use beyond this academic experiment.

Defaults: seed 42, 3,000 train / 500 validation / 1,000 test examples, labels
approximately balanced; group/premise overlap removed. Token-overlong examples
are excluded before sampling. The dataset manifest pins source/model revisions,
IDs and hashes. Never edit the frozen splits after looking at test results.

QLoRA defaults: NF4 double quantisation, FP16 compute, rank 16, alpha 32,
dropout .05, all linear adapter targets, learning rate 2e-4, two epochs,
microbatch 1, accumulation 16, maximum sequence 1024, gradient checkpointing.
Warmup uses an integer 3% of scheduled optimiser steps (12 for the default
3,000-example single-GPU run). This avoids the removed `warmup_ratio` argument
in newer Transformers. Argument/data validation happens before loading weights.
Only assistant JSON response tokens contribute to loss. The best validation-loss
checkpoint is exported; training does not read the test file. Checkpoints and
loss history live on Drive. Use `--resume <checkpoint>` after interruptions.

Export reloads original FP16 weights, merges the adapter, then uses llama.cpp
to create Q4_K_M GGUF. The conversion commit and checksums are recorded. The
bundle includes the adapter and tokenizer; keep it if Colab expires. GPU quota,
training duration and improvements are not guaranteed. Preserve intermediate
merged Hugging Face weights on Drive if you later want SageMaker deployment.

Download `l3-model-bundle.zip`, `test.jsonl` and the data `manifest.json` from Drive.
Extract the bundle under `artifacts/l3_nli/bundle`, and place the data files in
`artifacts/l3_nli/data`. Training artifacts/weights are ignored by Git.

## 2. Import into local Ollama

Run from the repository root in PowerShell (Ollama must already be installed/running):

```powershell
Push-Location artifacts/l3_nli/bundle
ollama create ats-l3-qwen2.5-3b -f Modelfile.adapter
Pop-Location
ollama list
```

No original model is deleted. L3 defaults to `http://127.0.0.1:11434`;
`ATS_L3_OLLAMA_URL` can select another trusted/private server, including an EC2
host. Model storage (e.g. S3) is separate from the compute serving it.

```powershell
uv run python -m agentic_talent_screener.resume_intake.pipeline "PATH_TO_SYNTHETIC_RESUME.pdf" --l3-verifier ollama --l3-model ats-l3-qwen2.5-3b
```

L1/L2 retain their OpenRouter configuration. The default L3 verifier in the
running app is the OpenRouter client with model reasoning disabled
(`--l3-verifier openrouter`); the local Qwen model (`ats-l3-qwen2.5-3b` through
Ollama) is the opt-in backend selected by `--l3-verifier ollama`, and
`--l3-verifier deterministic` is an explicit literal-only opt-out.
`--stop-after-l2` never calls the NLI model. Before the verifier is called,
deterministic trust-boundary gates require each
cited span to exist, be visible, and not be L0/L1-excluded; enforce
PII/institution policy; and construct only bounded, PII-minimised safe evidence.
Qwen receives only the claim plus that evidence. Only `entailed` crosses the
boundary; `contradicted` and `insufficient_evidence` reject the L2 claim. Date
extraction, evidence levels, rank credits, and citations stay deterministic.
This checks support inside the resume, not real-world credentials.
The progress output names the L3 model, and JSON `events` includes
`L3_NLI_MODEL:ats-l3-qwen2.5-3b`. The corresponding `event_details` entry reports
completion and retained-fact count, or `L3_NLI_LITERAL_FALLBACK`. Each accepted
fact records audit-only `verification_method`: `nli_literal` when literal
support also exists, or `nli_semantic` when Qwen support is semantic-only.
Unsupported claims have `CLAIM_NOT_ENTAILED` and the model's verdict. Only
retained facts appear in `profile.facts` for later processing. If the local model
is unavailable, replies invalidly, or receives an oversized packet, earlier
Qwen-verified facts remain intact and remaining claims may only cross using
literal support against the same safe evidence. These facts are marked
`literal_fallback`; the profile is `verified_with_flag` and requires human review
before ranking. There is no L2 repair retry. Long packets are rejected before
inference by a conservative UTF-8 bound; they are never silently truncated.

## 3. Measure the deployed artifact

For a live smoke check without invoking `uv`, activate the already-installed
project virtual environment and run:

```powershell
$env:ATS_RUN_L3_LIVE = "1"
python -m pytest tests/resume_intake/test_nli_live.py -v -s
Remove-Item Env:ATS_RUN_L3_LIVE
```

This calls the real local model on one synthetic claim/evidence pair and prints
the model digest and verdict. Once opted in, an unavailable model fails rather
than being silently skipped. Normal test runs skip it; existing offline NLI tests
mock the verifier and do not establish that your exported model is running.
`ATS_L3_LIVE_MODEL` optionally selects another imported model tag. No OpenRouter
key is required. A single passing smoke test does not establish NLI accuracy.

To exercise the pipeline path through the real local L3 model as well:

```powershell
$env:ATS_RUN_L3_LIVE = "1"
python -m pytest tests/resume_intake/test_nli_live.py tests/resume_intake/test_pipeline_nli.py -v -s
Remove-Item Env:ATS_RUN_L3_LIVE
```

The integration test controls the hosted L1/L2 responses, while running actual
L0 parsing, PII redaction, L2 schema validation and L3 verification. It checks
that a supported paraphrase is retained and a conflicting numerical claim is
dropped. It makes no OpenRouter calls and does not measure full hosted extraction
quality. Offline integration tests also cover model selection and the flagged
literal fallback for an unavailable verifier, without an L2 retry.

Start with three pairs per class for a smoke test; then use the fixed 300-pair
subset or all 1,000 pairs (`--limit 0`). A small CPU model can still take time:

```powershell
uv run python -m agentic_talent_screener.evaluation.nli evaluate --pairs artifacts/l3_nli/data/test.jsonl --out runs/l3-smoke --limit 9
uv run python -m agentic_talent_screener.evaluation.nli evaluate --pairs artifacts/l3_nli/data/test.jsonl --out runs/l3-adapter --limit 300
uv run python -m agentic_talent_screener.evaluation.nli synthetic --out artifacts/l3_nli/resume-pairs.jsonl
uv run python -m agentic_talent_screener.evaluation.nli evaluate --pairs artifacts/l3_nli/resume-pairs.jsonl --out runs/l3-synthetic --limit 0
```

Each new report directory contains JSON/Markdown metrics and ID-only predictions.
Report accuracy, macro-F1, per-class precision/recall, confusion matrix,
entailment precision, unsupported false acceptance, failures and latency together.
Failed predictions count as errors, not discarded observations. Three consecutive
processing failures stop a run and mark it partial. Wilson intervals describe
individual rates, not uncertainty due to correlated synthetic templates. The
synthetic file contains 100 cases from ten templates and is labelled by construction,
not independently human graded. It is evaluation-only and is not used for training.

Model digest, prompt version, dataset hash and selected-ID hash distinguish runs.
Quantisation can change predictions: use deployed GGUF measurements in the demo.

## 4. Project evaluation and fixed-L2 replay

The existing claims suite can use the local model. Its L1/L2 calls are free-only by default;
when free models cannot complete intake, an explicitly approved paid route uses the core
runner's allow-list, per-run budget, M1 phase ledger and OpenRouter usage accounting. Local
NLI remains a separate, local backend from the paid L1/L2 route.

```powershell
uv run python -m agentic_talent_screener.evaluation run --suite claims --l3-verifier ollama --l3-model ats-l3-qwen2.5-3b --pool-size 10 --n-bases 2 --n-jds 1 --max-llm-calls 40 --run-id claims-l3-adapter
uv run python -m agentic_talent_screener.evaluation.nli replay --input runs/claims-l3-adapter/l3_inputs.jsonl --out runs/l3-fixed-inputs.json
```

For an approved, directly routed paid L1/L2 comparison, open the M1 phase first and make the spend
choice explicit (the default remains free-only):

```powershell
uv run python -m agentic_talent_screener.evaluation phase-open --phase M1
uv run python -m agentic_talent_screener.evaluation run --suite claims --allow-paid --budget-usd 0.50 --model-policy paid-direct --l3-verifier ollama --l3-model ats-l3-qwen2.5-3b --pool-size 10 --n-bases 2 --n-jds 1 --max-llm-calls 40 --run-id claims-l3-adapter-paid
```

The calls cap can produce partial claims results; never present these as complete.
The suite exports synthetic-only first-valid L2 inputs for replay. Replay compares
literal L3 with NLI on exactly the same claims/citations, without repeating L1/L2.
An `--offline` claims run refuses Ollama rather than silently making local requests.
The word-overlap citation metric is retained as a **lexical diagnostic**, not semantic
ground truth. Claims results report verification-method counts and, when applicable,
export a seeded PII-minimised `audit/semantic_l3_facts.jsonl` population for human
inspection. The independent labelled-pair report assesses NLI accuracy; the
evaluated model must not grade its own answers as ground truth.

## 5. Optional baseline

Set `INCLUDE_BASELINE=True` before the first export. Import
`Modelfile.baseline` as `ats-l3-qwen2.5-3b-base`, then repeat the same evaluate
command with that tag and a different output directory. If already exported,
use a separate export run directory rather than overwriting a bundle.
Both models must come from the same base revision, Q4_K_M conversion settings,
prompt and test IDs. Do not compare an arbitrary existing Ollama quantisation
against a differently exported adapter model as a controlled LoRA improvement.

Prioritise the adapter run. Without a completed comparable baseline, present
absolute performance only. Preserve a separate test set if you tune again.
