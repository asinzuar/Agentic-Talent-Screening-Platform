"""Explicit local-model smoke test; normal offline test runs skip this module.

PowerShell, with the existing project virtual environment activated:
    $env:ATS_RUN_L3_LIVE = "1"
    python -m pytest tests/resume_intake/test_nli_live.py -v -s
    Remove-Item Env:ATS_RUN_L3_LIVE
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ATS_RUN_L3_LIVE") != "1",
    reason="Set ATS_RUN_L3_LIVE=1 to call the real local Ollama model.",
)


def test_real_ollama_verifier_supports_python_claim() -> None:
    from agentic_talent_screener.llm import ollama_model_digest
    from agentic_talent_screener.resume_intake.nli import DEFAULT_NLI_MODEL, NliUnavailable, OllamaVerifier

    model = os.environ.get("ATS_L3_LIVE_MODEL", DEFAULT_NLI_MODEL)
    try:
        digest = ollama_model_digest(model)
    except (ConnectionError, ValueError):
        pytest.fail(
            f"Cannot find local model {model!r}. Start Ollama, import the GGUF model, "
            "and check `ollama list` (and ATS_L3_OLLAMA_URL if customised).",
            pytrace=False,
        )
    try:
        verdict = OllamaVerifier(model).verify("Used Python", "Built a data pipeline using Python.")
    except NliUnavailable as error:
        pytest.fail(f"Local model smoke test could not obtain a verdict: {error}", pytrace=False)
    print(f"Model: {model}\nDigest: {digest}\nVerdict: {verdict}")
    assert verdict == "entailed", (
        f"The real model returned {verdict!r} for a clearly supported Python claim. "
        "This is a model-verdict failure, not a uv installation failure."
    )
