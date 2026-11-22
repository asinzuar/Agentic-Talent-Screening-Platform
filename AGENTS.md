# AGENTS.md

Context for AI coding agents working in this repository.

## Project

**Agentic Talent Screener**, Team 30's capstone for *AI-ML in Practice* (IIT Bombay). It is an agentic LLM system that ranks a whole applicant pool against a job description on verifiable resume evidence, explains each decision with cited spans, and is built and measured to resist adversarial resumes (prompt injection, hidden text, keyword stuffing, etc.). A human makes every hiring decision; the system only supports it.

Three core pillars:
1. **P1:** Evidence-grounded matching and pool-wide ranking
2. **P2:** Defence-in-depth guardrails (layers L0–L5)
3. **P3:** Adversarial evaluation harness (attacker agent + metrics)

## Current state: IMPLEMENTATION IN PROGRESS

- The proposal has been approved, and implementation is in progress. `SPEC.md` is still a draft with known gaps (see its "Open Questions" section).
- **Implemented so far:**
  - the ranking module (`ranking/`): requirement-based scoring, the Matching Agent interface with a stub, audit ledger and evaluation harness;
  - a FastAPI app (`api/`) serving the ranking endpoints and an HR workbench on synthetic sample data;
  - resume intake L0–L3 (`resume_intake/`): document forensics, injection detection, quarantined extraction, evidence verification, and an adapter that hands verified skill facts to ranking as free text;
  - the standalone Job Analysis Agent (`job_analysis.py`): drafts an unapproved requirements profile from JD text, with source quotes and reviewer flags.
- **Not yet built:** the requirements review workflow, a model-backed Matching Agent, the Explanation Agent, the attacker agent, and LangGraph / Langfuse integration.
- **Do not start implementing features unless explicitly asked.** Prefer helping to clarify, challenge and refine the spec.
- Keep code and spec in step: when one changes, update the other or record the mismatch in Open Questions.
- When the spec changes, keep it internally consistent. Mark assumptions as _(inferred)_ and add unresolved items to Open Questions rather than silently deciding them.

## Key files

| File | Purpose |
|---|---|
| `SPEC.md` | Working specification (draft) and the single source of truth for project intent. |
| `RESUME_INTAKE_SPEC.md` | Companion spec for the resume intake lane (L0–L3). `SPEC.md` wins on conflicts. |
| `docs/RANKING_SPEC.md`, `docs/HR_SCORING_GUIDE.md` | Ranking module spec, and a plain-language guide to scoring for HR. |
| `docs/examples/` | Worked examples supporting the spec (e.g. draft requirement profiles for sample JDs). Illustrative, not decisions. |
| `src/agentic_talent_screener/` | Code: `api/` (FastAPI app and workbench), `ranking/`, `resume_intake/`, `job_analysis.py`. Tests are in `tests/`. |
| `README.md` | Short project intent and dev commands. |
| `pyproject.toml` / `uv.lock` | Python project (Python ≥ 3.12, managed with `uv`). |

## Non-negotiable design principles

- **Trust boundary:** raw resume text never reaches a model that has tools, the scoring rubric, or other candidates' data. Only schema-validated, span-verified facts cross it. A fact's quote (bounded, PII-screened, filtered) goes to HR review and the Explanation Agent only; the Matching Agent sees statements, not quotes (see `SPEC.md` §13.2).
- **Evidence over mentions:** every claim used for scoring must cite a span in the visible resume text.
- **Human decides:** no autonomous hire/reject decisions; security flags and offending spans are surfaced to a human. Humans also own the scoring criteria: no candidate is scored against a requirement profile a human hasn't approved.
- **Each defence layer is independently measurable,** so it can be ablated.
- **Consent-respecting data:** development and evaluation data must be public, licensed or synthetic, with no real people's data collected without consent. At runtime, resumes are PII submitted by candidates: use them only to screen that application, and don't gather candidate data from third-party sources.

## Tooling

- Use `uv` for environments and dependencies (`uv add`, `uv run`). Don't use bare `pip`.
- Run tests with `uv run pytest` and the API with `uv run fastapi dev`.
- In use: FastAPI and Pydantic. Planned: LangGraph (orchestration) and Langfuse (tracing). The final UI is still TBD.
- All model calls go through OpenRouter via `src/agentic_talent_screener/llm.py`. The default model is Gemma 4 (`google/gemma-4-31b-it:free`), overridable with `ATS_LLM_MODEL`; the key is `OPENROUTER_API_KEY`. Both are read from the environment or the git-ignored `.env`.
- **Secrets:** never read, print, display or log the contents of `.env`, and especially never the value of `OPENROUTER_API_KEY` (e.g. no `cat .env`, no echoing the variable, no printing settings objects that hold it). If you need to know whether the key is set, check only for its presence without showing the value. Never commit `.env` or paste a key into code, tests, docs or command output.
