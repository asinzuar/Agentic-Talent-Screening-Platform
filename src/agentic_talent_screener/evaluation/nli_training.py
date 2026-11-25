"""Colab GPU training and GGUF export; heavy dependencies are lazy and optional."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

from .nli import read_jsonl, sha256, write_json
from agentic_talent_screener.resume_intake.nli import PROMPT_VERSION, messages


def encode_example(row: dict, tokenizer, max_length: int = 1024) -> dict:
    prompt = messages(row["claim"], row["evidence"])
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content":
                                                   json.dumps({"verdict": row["label"]})}], tokenize=True, return_dict=False)
    if full[:len(prefix)] != prefix or len(full) > max_length or len(full) <= len(prefix):
        raise ValueError("Invalid chat template boundary or oversized training example; do not truncate.")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


class AssistantCollator:
    """Right-pad training inputs and preserve masked prompt/padding labels."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]):
        import torch
        longest = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [pad] * (longest-len(row[key])) for row in rows])
                for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def train(data: Path, out: Path, resume: str | None = None) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise ValueError("Training requires a Colab NVIDIA GPU. Select Runtime > Change runtime type > GPU.")
    if out.exists() and any(out.iterdir()) and resume is None:
        raise ValueError("Training output is not empty; use a new run directory or --resume checkpoint.")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Dataset prompt version does not match the training code.")
    # Training reads only train/validation; the held-out test labels are never loaded.
    for name in ("train.jsonl", "validation.jsonl"):
        if sha256(data / name) != manifest["sha256"][name]:
            raise ValueError("Frozen dataset checksum mismatch.")
    configuration = {"dataset": manifest, "prompt_version": PROMPT_VERSION, "epochs": 2,
                     "rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 2e-4,
                     "batch_size": 1, "gradient_accumulation": 16, "seed": 42}
    if resume:
        previous = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError("Resume configuration differs from the original run.")
    else:
        write_json(out / "run_config.json", configuration)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["model_revision"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=manifest["model_revision"],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        device_map={"": 0}, dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=2, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=16, learning_rate=2e-4,
        warmup_ratio=0.03, fp16=True, gradient_checkpointing=True, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=10,
        report_to="none", seed=42, data_seed=42, optim="adamw_torch", remove_unused_columns=False,
    )
    encoded = {name: Dataset.from_list([encode_example(row, tokenizer) for row in read_jsonl(data / f"{name}.jsonl")])
               for name in ("train", "validation")}
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
                      data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=resume)
    adapter = out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    write_json(out / "training_manifest.json", {
        "dataset": manifest, "prompt_version": PROMPT_VERSION, "training_arguments": args.to_dict(),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
        "packages": {p: importlib.metadata.version(p) for p in (
            "torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "huggingface-hub")},
        "gpu": torch.cuda.get_device_name(0),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "note": "Test labels were not used. No baseline or accuracy improvement is claimed.",
    })


def export(run: Path, llama_cpp: Path, *, include_baseline: bool = False) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    manifest = json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
    if sha256(run / "adapter" / "adapter_model.safetensors") != manifest["adapter_sha256"]:
        raise ValueError("Adapter checksum mismatch.")
    dataset = manifest["dataset"]
    bundle = run / "bundle"
    if bundle.exists():
        raise ValueError("Bundle already exists; exports will not be overwritten.")
    converter = llama_cpp / "convert_hf_to_gguf.py"
    quantizer = llama_cpp / "build" / "bin" / "llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise ValueError("Build llama.cpp's llama-quantize target before exporting.")
    # Reload original FP16 weights rather than merging into the 4-bit training copy.
    base = AutoModelForCausalLM.from_pretrained(dataset["model"], revision=dataset["model_revision"],
                                              dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(run / "adapter")
    bundle.mkdir(parents=True)
    tokenizer.save_pretrained(bundle / "tokenizer")
    llama_revision = subprocess.check_output(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True).strip()

    def save_weights(model, suffix: str) -> None:
        directory = run / f"merged-{suffix}"
        model.save_pretrained(directory, safe_serialization=True)
        tokenizer.save_pretrained(directory)

    if include_baseline:
        save_weights(base, "baseline")
    merged = PeftModel.from_pretrained(base, run / "adapter").merge_and_unload()
    save_weights(merged, "adapter")
    # The converter runs in another process. Release this 6 GB FP16 model first
    # rather than keeping two full copies resident on a standard Colab runtime.
    del merged, base
    import gc
    gc.collect()

    def convert(suffix: str, tag: str) -> None:
        directory = run / f"merged-{suffix}"
        intermediate = run / f"{suffix}-f16.gguf"
        import sys
        subprocess.run([sys.executable, str(converter), str(directory), "--outfile", str(intermediate),
                        "--outtype", "f16"], check=True)
        output = bundle / f"{tag}.gguf"
        subprocess.run([str(quantizer), str(intermediate), str(output), "Q4_K_M"], check=True)
        (bundle / f"Modelfile.{suffix}").write_text(
            f"FROM ./{output.name}\nPARAMETER temperature 0\nPARAMETER num_ctx 8192\n", encoding="utf-8")

    if include_baseline:
        convert("baseline", "ats-l3-qwen2.5-3b-base")
    convert("adapter", "ats-l3-qwen2.5-3b")
    import shutil
    shutil.copytree(run / "adapter", bundle / "adapter")
    write_json(bundle / "manifest.json", {
        "training": manifest, "llama_cpp_revision": llama_revision, "quantization": "Q4_K_M",
        "prompt_version": PROMPT_VERSION,
        "files": {str(p.relative_to(bundle)): sha256(p) for p in bundle.rglob("*") if p.is_file()},
        "note": "Re-evaluate the deployed GGUF: merging/quantization can change predictions.",
    })
    # Include adapter and manifest in the download, not just the merged inference model.
    shutil.make_archive(str(run / "l3-model-bundle"), "zip", bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    tr.add_argument("--data", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--resume")
    ex = commands.add_parser("export")
    ex.add_argument("--run", type=Path, required=True)
    ex.add_argument("--llama-cpp", type=Path, required=True)
    ex.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.out, args.resume)
    else:
        export(args.run, args.llama_cpp, include_baseline=args.include_baseline)


if __name__ == "__main__":
    main()
