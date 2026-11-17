# AGENTS.md

Context for AI coding agents working in this repository.

## Project

**Agentic Talent Screener**, Team 30's capstone for *AI-ML in Practice* (IIT Bombay). It is an agentic LLM system that ranks a whole applicant pool against a job description on verifiable resume evidence, explains each decision with cited spans, and is built and measured to resist adversarial resumes (prompt injection, hidden text, keyword stuffing, etc.). A human makes every hiring decision; the system only supports it.

Three core pillars:
1. **P1:** Evidence-grounded matching and pool-wide ranking
2. **P2:** Defence-in-depth guardrails (layers L0–L5)
3. **P3:** Adversarial evaluation harness (attacker agent + metrics)

## Current state: EARLY SPECCING

- The proposal has been approved. **No implementation exists yet**, and `src/agentic_talent_screener/__init__.py` holds only a placeholder `main()`.
- We are refining `SPEC.md` before writing any code. It is a draft with known gaps (see its "Open Questions" section).
- **Do not start implementing features unless explicitly asked.** Prefer helping to clarify, challenge and refine the spec.
- When the spec changes, keep it internally consistent. Mark assumptions as _(inferred)_ and add unresolved items to Open Questions rather than silently deciding them.

## Key files

| File | Purpose |
|---|---|
| `SPEC.md` | Working specification (draft) and the single source of truth for project intent. This is the main artefact right now. |
| `README.md` | Short project intent. |
| `pyproject.toml` / `uv.lock` | Python project (Python ≥ 3.12, managed with `uv`). Only `docling` is a dependency so far. |

## Non-negotiable design principles

- **Trust boundary:** raw resume text never reaches a model that has tools, the scoring rubric, or other candidates' data. Only schema-validated, span-verified facts cross it.
- **Evidence over mentions:** every claim used for scoring must cite a span in the visible resume text.
- **Human decides:** no autonomous hire/reject decisions; security flags and offending spans are surfaced to a human. Humans also own the scoring criteria: no candidate is scored against a requirement profile a human hasn't approved.
- **Each defence layer is independently measurable,** so it can be ablated.
- **Consent-respecting data:** development and evaluation data must be public, licensed or synthetic, with no real people's data collected without consent. At runtime, resumes are PII submitted by candidates: use them only to screen that application, and don't gather candidate data from third-party sources.

## Tooling

- Use `uv` for environments and dependencies (`uv add`, `uv run`). Don't use bare `pip`.
- Planned stack: LangGraph (orchestration) and Langfuse (tracing). Models and UI are still TBD.
