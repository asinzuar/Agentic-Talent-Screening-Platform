from __future__ import annotations

import json
import pytest

from agentic_talent_screener.evaluation.nli import (
    evaluate, group_keys, make_splits, metrics, synthetic_pairs, write_jsonl,
)
from agentic_talent_screener.evaluation.nli_training import encode_example
from agentic_talent_screener.resume_intake.nli import LABELS


def examples(prefix, n):
    return [{"id": f"{prefix}-{i}", "group_id": f"{prefix}-{i//3}", "claim": f"Claim {i}",
             "evidence": f"Evidence {prefix} {i}", "label": LABELS[i % 3]} for i in range(n)]


def test_splits_are_balanced_disjoint_reproducible():
    train, test = examples("train", 120), examples("test", 30)
    train += [{**test[0], "id": "copied", "group_id": "copied-source"}]
    splits = make_splits(train, test, train_n=9, val_n=6, test_n=30)
    assert splits == make_splits(train, test, train_n=9, val_n=6, test_n=30)
    groups = {k: set().union(*(group_keys(row) for row in rows)) for k, rows in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert all(sum(r["label"] == label for r in splits["train"]) == 3 for label in LABELS)


def test_metrics_count_failed_predictions_and_false_acceptance():
    result = metrics([
        {"gold": "entailed", "prediction": "entailed", "seconds": 1},
        {"gold": "contradicted", "prediction": "entailed", "seconds": 1},
        {"gold": "insufficient_evidence", "prediction": None, "seconds": 1},
    ])
    assert result["accuracy"]["value"] == pytest.approx(1/3)
    assert result["entailment_precision"]["value"] == 0.5
    assert result["unsupported_false_acceptance"]["value"] == 0.5
    assert result["confusion_matrix"]["insufficient_evidence"]["error"] == 1
    assert metrics([])["accuracy"]["value"] is None


def test_masking_trains_only_assistant_response_and_refuses_truncation():
    class Tokenizer:
        def apply_chat_template(self, chat, *, tokenize, add_generation_prompt=False, return_dict=False):
            assert return_dict is False
            return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
    row = {"claim": "claim", "evidence": "evidence", "label": "entailed"}
    encoded = encode_example(row, Tokenizer())
    assert encoded["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match="oversized"):
        encode_example(row, Tokenizer(), max_length=4)


def test_synthetic_cases_cover_project_failures():
    pairs = synthetic_pairs()
    assert len(pairs) == len({r["id"] for r in pairs}) == 100
    assert {r["label"] for r in pairs} == set(LABELS)
    assert {r["category"] for r in pairs} >= {"leadership", "proficiency", "dates", "wrong_citation", "percentage"}


def test_colab_source_entrypoint_does_not_need_the_installed_application():
    import subprocess
    import sys
    from pathlib import Path
    source = str(Path(__file__).resolve().parents[2] / "src")
    code = (f"import sys,runpy; sys.path.insert(0,{source!r}); "
            "sys.argv=['training','--help']; "
            "runpy.run_module('agentic_talent_screener.evaluation.nli_training',run_name='__main__')")
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "export" in result.stdout


def test_evaluation_saves_reproducible_results_and_stops_on_provider_failure(tmp_path, monkeypatch):
    from agentic_talent_screener import llm
    from agentic_talent_screener.resume_intake.nli import NliUnavailable
    from agentic_talent_screener.evaluation import nli
    monkeypatch.setattr(llm, "ollama_model_digest", lambda _: "sha-test")
    class Fake:
        def __init__(self, *_): pass
        def verify(self, *_): raise NliUnavailable("test processing failure")
    monkeypatch.setattr(nli, "OllamaVerifier", Fake)
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, synthetic_pairs())
    result = evaluate(path, tmp_path / "results", "model", limit=9)
    assert result["status"] == "partial" and result["n_processed"] == 3
    assert result["metrics"]["failure_rate"]["value"] == 1
    assert result["metrics"]["unsupported_false_acceptance"]["value"] is None
    assert (tmp_path / "results" / "report.md").exists()
    with pytest.raises(ValueError, match="overwritten"):
        evaluate(path, tmp_path / "results", "model", limit=9)
