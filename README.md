# Agentic Talent Screener

Capstone project for **AI-ML in Practice** (IIT Bombay), Team 30.

Agentic Talent Screener helps recruiters compare applicants against approved job requirements using evidence cited from resumes. It separates resume security, semantic matching, and deterministic scoring. Humans own the screening criteria and make all hiring decisions.

> **Status:** the core modules are implemented and testable on their own. The FastAPI demo still uses a synthetic pool and the static matcher; it does not run the live job-description-to-ranking workflow. The HR workbench at `/` is a **deprecated early prototype**, not the target UI.

## Quick start

```sh
uv sync             # install dependencies (Python 3.12+)
uv run pytest       # run the automated tests; no OpenRouter key required
uv run fastapi dev  # serve the synthetic demo API at http://127.0.0.1:8000
```

Open [the API docs](http://127.0.0.1:8000/docs) or try `GET /health` and `GET /api/v1/sample`. The demo also exposes ranking, benchmarks, override recording, and audit inspection/export under `/api/v1`.

## What’s implemented

| Module | What it does |
|---|---|
| [Job Analysis Agent](src/agentic_talent_screener/job_analysis.py) | Turns a full UTF-8 job description into an **unapproved** `RequirementProfile`. Explicit qualifications start active in the draft; inferred requirements and model suggestions start inactive. It preserves source quotes, non-scoring excerpts, and flags for human review. |
| [Resume intake (L0–L3)](src/agentic_talent_screener/resume_intake/) | Processes PDF, DOCX, and TXT files. L0 checks document structure and text visibility; L1 screens for prompt injection; L2 extracts cited claims from sanitised text in isolation; L3 verifies claims against visible source spans and assigns evidence levels. The ranking adapter hands off verified facts and risk metadata. |
| [Matching Agent](src/agentic_talent_screener/ranking/matching.py) | `LLMMatcher` uses OpenRouter to link active options and substitutes from an approved profile to one candidate’s verified facts. It receives fact statements, not resume quotes or span IDs, and returns equivalent or adjacent links with similarity and rationale. `StaticMatcher` supplies preset links for offline tests and the demo. |
| [Scoring and ranking](src/agentic_talent_screener/ranking/scorer.py) | Validates match links and scores against approved requirements. Produces a ranked pool, tiers, an optional top-N shortlist, and requirement-by-requirement explanations using the original evidence and citations. Can compare scores with and without flagged spans when those spans are present in the input. Matching failures are isolated to the affected candidate. |
| [Audit ledger](src/agentic_talent_screener/ranking/audit.py) | Records profile approvals, scoring runs, benchmark runs, and manual override reasons in a hash-chained log with JSONL export. The demo ledger is in memory and resets on restart. |
| [Evaluation harness](src/agentic_talent_screener/ranking/eval_harness.py) | Compares the ranking engine with keyword and TF-IDF baselines and a run with evidence and security checks removed. The bundled benchmark is synthetic; it does not establish real-world accuracy or robustness. |
| [FastAPI demo](src/agentic_talent_screener/api/) | Serves the sample pool, ranking, benchmark, override, and audit routes. It does not accept job descriptions or resumes for live screening. |
| [Shared OpenRouter client](src/agentic_talent_screener/llm.py) | Handles model configuration, structured JSON requests, and retries for transient failures. Used by Job Analysis, the intake security judge and extractor, and `LLMMatcher`. |

The Python handoff is: **JD → draft requirements → human approval**; **resume → verified facts**. `score_pool()` takes the approved profile, verified candidates, and a matcher. The matcher proposes links; only the scorer computes credit and rank. The approval guard is implemented, but the HR approval workflow and live API wiring are not.

## Repository layout

```text
src/agentic_talent_screener/
├── job_analysis.py        # JD-to-draft agent and CLI
├── llm.py                 # shared OpenRouter client and settings
├── resume_intake/
│   ├── pipeline.py        # intake orchestration and CLI
│   ├── l0_forensics.py    # document checks and visible source spans
│   ├── l1_injection.py    # injection screening
│   ├── l2_resume_agent.py # quarantined structured extraction
│   ├── l3_evidence_verifier.py  # claim verification and evidence levels
│   ├── contracts.py, grounding.py, hooks.py, risk.py
│   └── ranking_adapter.py # verified facts and risk metadata → ranking profile
├── ranking/
│   ├── schemas.py         # requirements, facts, links, and result types
│   ├── matching.py        # OpenRouter and static matchers
│   ├── scorer.py, config.py    # deterministic scoring and configuration
│   ├── audit.py, eval_harness.py
│   └── sample_data.py     # synthetic profiles, facts, and preset links
└── api/
    ├── app.py, config.py, dependencies.py
    ├── routers/           # health, versioned demo endpoints, legacy workbench
    └── static/            # deprecated workbench HTML
tests/                     # intake, analysis, matching, ranking, API, and client tests
synthetic_test_data/       # tiny JD for live smoke tests
docs/                     # integration guide, ranking docs, and examples
```

## Run the agents locally

Set `OPENROUTER_API_KEY` in your environment or git-ignored `.env`. `ATS_LLM_MODEL` defaults to `google/gemma-4-31b-it:free` and can be overridden with `--model` on the CLIs. `ATS_LLM_MAX_RETRIES` controls retries for transient failures. Use synthetic inputs for live smoke tests; provider and data-retention policies for real applicant data are still open.

```sh
# Draft a requirements profile from the bundled synthetic JD
uv run python -m agentic_talent_screener.job_analysis synthetic_test_data/tiny_jd.txt

# Run intake on a local synthetic resume (replace with your PDF, DOCX, or TXT path)
uv run python -m agentic_talent_screener.resume_intake.pipeline path/to/synthetic-resume.pdf
```

Use the [integration testing guide](docs/integration-testing-guide.md) for offline tests and live Qwen smoke tests of Job Analysis and Matching through the real scorer.

## What’s pending

- HR requirements review and approval workflow, the target UI, and live workflow/API integration with durable storage and access controls.
- A separate Explanation Agent, broader adversarial evaluation and the attacker agent, and evaluation of matcher accuracy and similarity calibration.
- LangGraph orchestration and Langfuse tracing.

## Further reading

- [Project specification](SPEC.md) and [resume intake specification](RESUME_INTAKE_SPEC.md)
- [Ranking specification](docs/RANKING_SPEC.md) and [HR scoring guide](docs/HR_SCORING_GUIDE.md)
- [Example draft requirement profiles](docs/examples/requirement-profiles.md)

<details>
<summary>OpenRouter API examples and exploratory notes</summary>

The project uses the shared chat-completions client. The Jev Decisions API example below is exploratory and is not wired into the system.

```sh
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <OPENROUTER_API_KEY>" \
  -d '{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
}'
```

### Using Jev with OpenRouter

Use `typesafe/jev-1.14` with the OpenRouter API:

This model does not generate text. It answers typed questions about a state (a string, object, or array) through the Decisions API and returns calibrated probabilities: a yes/no probability (noul), a pick from options you define (choice), or a position on an ordered rubric (score). Your code owns the workflow and acts on the answers, so use it for routing, ranking, verification, and other structured decisions rather than chat.

```python
import requests
import base64
import json

with open("image.png", "rb") as f:
  image_base64 = base64.b64encode(f.read()).decode("utf-8")

# The model answers narrow, typed questions about the state. Your code owns the workflow.
response = requests.post(
  url="https://openrouter.ai/api/alpha/decisions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "typesafe/jev-1.14",
    "state": [
      "Listing: vintage red bicycle, light scratches on the frame",
      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ],
    "questions": {
      "matches_listing": {
        "type": "noul",
        "instructions": "Does the attached media show the listed item?",
        "criteria": {
          "true": "The listed item is shown",
          "false": "Something else or nothing is shown"
        }
      },
      "category": {
        "type": "choice",
        "instructions": "Which category fits this listing?",
        "criteria": {
          "bikes": "Complete bicycles",
          "parts": "Components and accessories",
          "apparel": "Clothing and gear"
        }
      },
      "condition": {
        "type": "score",
        "instructions": "How worn is the item?",
        "criteria": ["Like new", "Light wear", "Heavy wear"]
      }
    }
  })
)

answers = response.json()["answers"]
# noul is a probability from 0 (no) to 1 (yes); choice and score carry the full distribution.
print(answers["matches_listing"]["noul"])
print(answers["category"]["choice"], answers["category"]["probabilities"])
print(answers["condition"]["score"])

if answers["matches_listing"]["noul"] > 0.8 and answers["category"]["choice"] == "bikes":
  pass  # publish_listing(...)

```

</details>

## Team

Neeraj Gupta · Vaibhav Gupta · Rahul Chhiber · Hardik Vyas · Anut Verma
