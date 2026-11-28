from __future__ import annotations

import json
import pytest

from agentic_talent_screener.evaluation.nli import (
    evaluate, group_keys, make_splits, metrics, synthetic_pairs, write_jsonl,
)
from agentic_talent_screener.evaluation.nli_training import (
    encode_example, prepare_run_directory, training_arguments_kwargs,
)
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


@pytest.mark.parametrize("examples,steps", [(3000, 12), (17, 1), (1, 1)])
def test_warmup_uses_integer_steps_for_the_training_schedule(tmp_path, examples, steps):
    kwargs = training_arguments_kwargs(tmp_path, examples)
    assert kwargs["warmup_steps"] == steps
    assert isinstance(kwargs["warmup_steps"], int)
    assert "warmup_ratio" not in kwargs


def test_training_arguments_match_the_installed_transformers_api(tmp_path):
    import inspect
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    kwargs = training_arguments_kwargs(tmp_path / "run", 3000)
    inspect.signature(transformers.TrainingArguments).bind(**kwargs)
    # Exercise constructor validation without a GPU or model download.
    args = transformers.TrainingArguments(**{**kwargs, "use_cpu": True, "fp16": False})
    assert args.warmup_steps == 12
    assert args.load_best_model_at_end


def test_metadata_only_startup_retry_is_safe_and_configuration_must_match(tmp_path):
    run = tmp_path / "run"
    prepare_run_directory(run, {"seed": 42})
    original = (run / "run_config.json").read_bytes()
    prepare_run_directory(run, {"seed": 42})
    assert (run / "run_config.json").read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        prepare_run_directory(run, {"seed": 7})
    assert (run / "run_config.json").read_bytes() == original


@pytest.mark.parametrize("artifact", ["checkpoint-188", "adapter", "training_manifest.json", "unrelated.txt"])
def test_training_artifacts_cannot_be_overwritten_by_a_fresh_run(tmp_path, artifact):
    run = tmp_path / "run"
    prepare_run_directory(run, {"seed": 42})
    (run / artifact).write_text("existing work")
    with pytest.raises(ValueError, match="contains artifacts"):
        prepare_run_directory(run, {"seed": 42})
    assert (run / artifact).read_text() == "existing work"


def test_resume_requires_an_actual_matching_checkpoint(tmp_path):
    run = tmp_path / "run"
    prepare_run_directory(run, {"seed": 42})
    checkpoint = run / "checkpoint-188"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="saved Trainer"):
        prepare_run_directory(run, {"seed": 42}, str(checkpoint))
    for name in ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_text("test checkpoint")
    prepare_run_directory(run, {"seed": 42}, str(checkpoint))
    with pytest.raises(ValueError, match="differs"):
        prepare_run_directory(run, {"seed": 7}, str(checkpoint))


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


@pytest.mark.parametrize("exit_code", [0, 3])
def test_notebook_runner_exposes_child_output_and_saves_traceback(tmp_path, monkeypatch, capsys, exit_code):
    import ast
    import subprocess
    import sys
    import time
    from pathlib import Path
    notebook = json.loads((Path(__file__).resolve().parents[2] / "notebooks/l3_qwen_qlora.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_module")
    namespace = dict(UV="unused", TRAIN_REQ="unused", REPO=tmp_path, training_env={},
                     Path=lambda _path: tmp_path, subprocess=subprocess, time=time)
    original_popen = subprocess.Popen
    code = "print('actual child diagnostic', flush=True)"
    if exit_code:
        code += "; import sys; print('GPU startup error', file=sys.stderr); sys.exit(3)"
    monkeypatch.setattr(subprocess, "Popen", lambda _cmd, **kwargs:
                        original_popen([sys.executable, "-c", code], **kwargs))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<notebook-helper>", "exec"), namespace)
    if exit_code:
        with pytest.raises(RuntimeError, match="exit 3"):
            namespace["run_module"]("example.training", "train")
    else:
        namespace["run_module"]("example.training", "train")
    logs = list(tmp_path.glob("*.log"))
    assert len(logs) == 1 and "actual child diagnostic" in logs[0].read_text()
    captured = capsys.readouterr().out
    assert "actual child diagnostic" in captured
    if exit_code:
        assert "GPU startup error" in captured and "GPU startup error" in logs[0].read_text()


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
