"""Measure claim parity for clean resumes and their adversarial twins.

This deliberately invokes the production intake flow: OpenRouter for L1/L2 and
the configured local Ollama model for L3.  It does not use the evaluation
cache, static matcher, or semantic batching.  The report holds aggregate
counts and PII-free verified statements only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_talent_screener.resume_intake.nli import OllamaVerifier
from agentic_talent_screener.resume_intake.pipeline import normalize_resume


def _facts(result: object) -> list[tuple[str, str]]:
    profile = getattr(result, "profile", None)
    if profile is None:
        return []
    return sorted((fact.kind, fact.statement) for fact in profile.facts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True,
                        help="JSON list with candidate, clean, and twin PDF paths.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    # One verifier lifecycle permits safe in-process reuse of repeated semantic
    # checks. Batching stays explicitly disabled.
    verifier = OllamaVerifier(enable_batching=False)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        clean = normalize_resume(pair["clean"], l3_verifier="ollama", support_verifier=verifier,
                                 show_progress=False)
        twin = normalize_resume(pair["twin"], l3_verifier="ollama", support_verifier=verifier,
                                show_progress=False)
        clean_facts, twin_facts = _facts(clean), _facts(twin)
        rows.append({
            "candidate": pair["candidate"],
            "clean_status": clean.status,
            "twin_status": twin.status,
            "clean_l2_claims": clean.l2_proposed_claim_count,
            "twin_l2_claims": twin.l2_proposed_claim_count,
            "l2_claim_count_equal": clean.l2_proposed_claim_count == twin.l2_proposed_claim_count,
            "clean_verified_facts": len(clean_facts),
            "twin_verified_facts": len(twin_facts),
            "verified_statements_equal": clean_facts == twin_facts,
            "clean_only_count": len(set(clean_facts) - set(twin_facts)),
            "twin_only_count": len(set(twin_facts) - set(clean_facts)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
