"""Standalone NLI data preparation, local evaluation and fixed-L2 replay.

No provider calls or optional training imports occur until explicitly requested.
Run with ``python -m agentic_talent_screener.evaluation.nli --help``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from agentic_talent_screener.resume_intake.nli import (
    DEFAULT_NLI_MODEL, LABELS, PROMPT_VERSION, NliUnavailable, OllamaVerifier,
)

LABEL_MAP = {"entailment": "entailed", "contradiction": "contradicted", "neutral": "insufficient_evidence"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_keys(row: dict) -> set[str]:
    premise = " ".join(row["evidence"].casefold().split())
    return {f"source:{row['group_id']}", f"premise:{hashlib.sha256(premise.encode()).hexdigest()}"}


def select_rows(rows: list[dict], n: int, seed: int, blocked: set[str]) -> list[dict]:
    if n < 1:
        raise ValueError("Split size must be positive.")
    rows = sorted(rows, key=lambda r: str(r["id"]))
    random.Random(seed).shuffle(rows)
    targets = {label: n // 3 + (i < n % 3) for i, label in enumerate(LABELS)}
    seen, counts, result = set(), Counter(), []
    for row in rows:
        pair = (row["evidence"].strip(), row["claim"].strip())
        if group_keys(row) & blocked or pair in seen or counts[row["label"]] >= targets[row["label"]]:
            continue
        result.append(row)
        seen.add(pair)
        counts[row["label"]] += 1
        if len(result) == n:
            break
    if len(result) != n:
        raise ValueError(f"Not enough eligible rows for balanced split: requested {n}, found {len(result)}.")
    return result


def make_splits(train: list[dict], test: list[dict], *, seed=42, train_n=3000, val_n=500, test_n=1000) -> dict:
    selected_test = select_rows(test, test_n, seed, set())
    blocked = set().union(*(group_keys(row) for row in selected_test))
    selected_val = select_rows(train, val_n, seed + 1, blocked)
    blocked.update(set().union(*(group_keys(row) for row in selected_val)))
    selected_train = select_rows(train, train_n, seed + 2, blocked)
    return {"train": selected_train, "validation": selected_val, "test": selected_test}


def prepare_wanli(out: Path, *, revision: str | None = None) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from agentic_talent_screener.resume_intake.nli import messages

    if out.exists() and any(out.iterdir()):
        raise ValueError("Dataset output must be a new or empty directory; frozen splits will not be overwritten.")
    revision = HfApi().dataset_info("alisawuffles/WANLI", revision=revision).sha
    model_revision = HfApi().model_info("Qwen/Qwen2.5-3B-Instruct").sha
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision=model_revision)
    dataset = load_dataset("alisawuffles/WANLI", revision=revision)
    eligible = {}
    for name in ("train", "test"):
        rows = []
        for row in dataset[name]:
            label = LABEL_MAP[row["gold"]]
            item = {"id": f"wanli-{row['id']}", "group_id": str(row["pairID"]),
                    "claim": row["hypothesis"], "evidence": row["premise"], "label": label,
                    "source": "WANLI", "source_split": name}
            chat = messages(item["claim"], item["evidence"]) + [
                {"role": "assistant", "content": json.dumps({"verdict": label})}]
            if len(tokenizer.apply_chat_template(chat, tokenize=True, return_dict=False)) <= 1024:
                rows.append(item)
        eligible[name] = rows
    splits = make_splits(eligible["train"], eligible["test"])
    hashes = {}
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[path.name] = sha256(path)
    write_json(out / "manifest.json", {"dataset": "alisawuffles/WANLI", "revision": revision,
               "license": "CC-BY-4.0", "model": "Qwen/Qwen2.5-3B-Instruct", "model_revision": model_revision,
               "seed": 42, "prompt_version": PROMPT_VERSION, "max_sequence_length": 1024,
               "counts": {k: len(v) for k, v in splits.items()}, "sha256": hashes})


def synthetic_pairs() -> list[dict]:
    """100 held-out synthetic pairs, not credentials/authenticity assessments."""
    result = []
    for i in range(10):
        n = i + 3
        cases = [
            ("paraphrase", "Developed Python pipelines.", "Built data pipelines using Python.", "entailed"),
            ("quantities", f"Managed a team of exactly {n} engineers.", f"Managed {n} engineers.", "entailed"),
            ("combined", "Used Python. Built a Flask API.", "Built a Flask API and used Python.", "entailed"),
            ("team_inflation", f"The Atlas team had exactly {n} members.",
             f"The Atlas team had exactly {n+5} members.", "contradicted"),
            ("percentage", f"Reduced Atlas latency by exactly {n}%.",
             f"Reduced Atlas latency by exactly {n+10}%.", "contradicted"),
            ("dates", "The Atlas project started in 2022 and ended in 2023.",
             "The Atlas project started in 2018.", "contradicted"),
            ("leadership", "Contributed code to the Atlas project.", "Led the Atlas project.", "insufficient_evidence"),
            ("proficiency", "Completed a Python course.", "Is an expert Python developer.", "insufficient_evidence"),
            ("wrong_citation", "Used SQL for reporting.", "Built a Kubernetes cluster.", "insufficient_evidence"),
            ("combined_overreach", "Built a Flask API. Used PostgreSQL.",
             "Led a team building a Flask API with PostgreSQL.", "insufficient_evidence"),
        ]
        for category, evidence, claim, label in cases:
            # Distinct project names keep examples identifiable; templates remain
            # deliberately disclosed, not 100 independent human judgements.
            result.append({"id": f"synthetic-{i:02d}-{category}", "group_id": f"synthetic-resume-{i}",
                           "evidence": evidence.replace("Atlas", f"Project{i}"),
                           "claim": claim.replace("Atlas", f"Project{i}"), "label": label,
                           "category": category, "source": "synthetic-by-construction"})
    return result


def rate(k: int, n: int) -> dict:
    if not n:
        return {"value": None, "n": 0, "ci95": None}
    p, z = k / n, 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"value": p, "n": n, "ci95": [max(0, center-half), min(1, center+half)]}


def metrics(rows: list[dict]) -> dict:
    matrix = {label: {p: 0 for p in (*LABELS, "error")} for label in LABELS}
    for row in rows:
        matrix[row["gold"]][row.get("prediction") or "error"] += 1
    per_class, f1s = {}, []
    for label in LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[g][label] for g in LABELS)
        gold = sum(matrix[label].values())
        precision, recall = tp/predicted if predicted else 0, tp/gold if gold else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        f1s.append(f1)
        per_class[label] = {"precision": rate(tp, predicted), "recall": rate(tp, gold), "f1": f1}
    valid = sum(row.get("prediction") in LABELS for row in rows)
    unsupported = [r for r in rows if r["gold"] != "entailed"]
    correct = sum(r.get("prediction") == r["gold"] for r in rows)
    return {"accuracy": rate(correct, len(rows)), "macro_f1": statistics.mean(f1s) if rows else None,
            "per_class": per_class, "confusion_matrix": matrix,
            "entailment_precision": per_class["entailed"]["precision"],
            "unsupported_false_acceptance": rate(sum(r.get("prediction") == "entailed" for r in unsupported),
                                                 len(unsupported)) if valid else rate(0, 0),
            "failure_rate": rate(len(rows)-valid, len(rows)),
            "invalid_response_rate": rate(sum(r.get("error_code") == "L3_NLI_INVALID_RESPONSE" for r in rows), len(rows)),
            "failures_by_code": dict(Counter(r["error_code"] for r in rows if r.get("error_code"))),
            "note": "Failures count as incorrect in accuracy/recall. False acceptance alone is not evidence of quality.",
            "mean_latency_seconds": statistics.mean(r["seconds"] for r in rows) if rows else None}


def evaluate(pairs: Path, out: Path, model: str, limit: int = 300) -> dict:
    from agentic_talent_screener.llm import ollama_model_digest

    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new report directory; previous results will not be overwritten.")
    data = read_jsonl(pairs)
    if any(r.get("label") not in LABELS or not all(isinstance(r.get(k), str) and r[k].strip()
                                                 for k in ("claim", "evidence", "id")) for r in data):
        raise ValueError("Invalid labelled pair dataset.")
    if len({r["id"] for r in data}) != len(data):
        raise ValueError("Pair IDs must be unique.")
    # Choose a fixed balanced subset, then iterate in a stable order. Synthetic
    # data can be intentionally unbalanced, so limit=0 evaluates the whole file.
    if limit:
        if limit < 3 or limit > len(data):
            raise ValueError("Limit must be zero (all) or between 3 and the dataset size.")
        ordered = sorted(data, key=lambda r: hashlib.sha256(f"42|{r['id']}".encode()).hexdigest())
        buckets = {label: [r for r in ordered if r["label"] == label] for label in LABELS}
        selected = []
        while len(selected) < limit:
            for label in LABELS:
                if buckets[label] and len(selected) < limit:
                    selected.append(buckets[label].pop())
        data = selected
    digest = ollama_model_digest(model)
    verifier, results = OllamaVerifier(model), []
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in data:
            start = time.perf_counter()
            row = {"id": item["id"], "gold": item["label"], "category": item.get("category"), "prediction": None}
            try:
                row["prediction"] = verifier.verify(item["claim"], item["evidence"])
                failures = 0
            except NliUnavailable as error:
                row["error"] = str(error)
                row["error_code"] = str(error).split(":", 1)[0]
                failures += 1
            row["seconds"] = time.perf_counter() - start
            results.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if failures == 3:
                break
    summary = {"model": model, "model_digest": digest, "prompt_version": PROMPT_VERSION,
               "dataset_sha256": sha256(pairs), "selected_ids_sha256": hashlib.sha256(
                   json.dumps([r["id"] for r in data]).encode()).hexdigest(),
               "n_planned": len(data), "n_processed": len(results),
               "status": "complete" if len(results) == len(data) and not any(r.get("error") for r in results) else "partial",
               "metrics": metrics(results),
               "label_source": "Dataset labels, not the evaluated model's own judgement. Synthetic templates are correlated."}
    if ollama_model_digest(model) != digest:
        summary["status"] = "invalid: model tag changed during evaluation"
    write_json(out / "results.json", summary)
    m = summary["metrics"]
    (out / "report.md").write_text(
        f"# L3 NLI evaluation\n\nModel: `{model}`\n\nStatus: {summary['status']}; "
        f"{len(results)}/{len(data)} pairs.\n\n"
        + "\n".join(f"- {k}: {json.dumps(m[k])}" for k in (
            "accuracy", "macro_f1", "entailment_precision", "unsupported_false_acceptance", "invalid_response_rate", "failure_rate"))
        + "\n\nNo baseline improvement or deployment threshold is claimed. "
        "Latency includes cold start; repeated pairs can hit the in-memory cache.\n", encoding="utf-8")
    return summary


def replay(source: Path, out: Path, model: str) -> None:
    from agentic_talent_screener.resume_intake.l2_resume_agent import AgentOutput
    from agentic_talent_screener.resume_intake.l3_evidence_verifier import verify_agent_output
    from agentic_talent_screener.llm import ollama_model_digest

    if out.exists():
        raise ValueError("Replay output already exists.")
    digest, verifier, results = ollama_model_digest(model), OllamaVerifier(model), []
    for row in read_jsonl(source):
        if row.get("synthetic") is not True:
            raise ValueError("Replay accepts only explicitly synthetic claims-suite captures.")
        output = AgentOutput.model_validate(row["agent_output"])
        args = (output, row["forensic_report"], set(row["excluded_span_ids"]), row["blind_candidate_id"])
        literal = verify_agent_output(*args)
        item = {"candidate_id": row["candidate_id"], "literal_fact_ids": [f.fact_id for f in literal.facts]}
        try:
            profile = verify_agent_output(*args, support_verifier=verifier)
            item.update(nli_fact_ids=[f.fact_id for f in profile.facts],
                        events=[e.model_dump() for e in profile.verification_events], status="complete")
        except NliUnavailable as error:
            item.update(status="manual_review", error=str(error))
        results.append(item)
    write_json(out, {"model": model, "model_digest": digest, "source_sha256": sha256(source),
                    "items": results, "note": "Same captured L2 inputs; no accuracy claim without independent labels."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--revision")
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--out", type=Path, required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--pairs", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--model", default=DEFAULT_NLI_MODEL)
    ev.add_argument("--limit", type=int, default=300)
    rp = commands.add_parser("replay")
    rp.add_argument("--input", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--model", default=DEFAULT_NLI_MODEL)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_wanli(args.out, revision=args.revision)
        elif args.command == "synthetic":
            if args.out.exists():
                raise ValueError("Synthetic output already exists.")
            write_jsonl(args.out, synthetic_pairs())
        elif args.command == "evaluate":
            print(json.dumps(evaluate(args.pairs, args.out, args.model, args.limit), indent=2))
        else:
            replay(args.input, args.out, args.model)
    except (ValueError, ConnectionError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
