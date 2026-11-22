# Agentic Talent Screener

Capstone project for **AI-ML in Practice** (IIT Bombay), Team 30.

An agentic LLM system for screening job applicants that:

- **ranks the whole applicant pool on verifiable evidence** from each resume, not keyword matches;
- **explains every decision** with cited resume spans so recruiters, hiring managers and auditors can check it;
- **resists manipulation** such as prompt injection, hidden text and keyword stuffing, using layered defences, and measures that robustness with an adversarial evaluation harness;
- **keeps a human in charge.** It supports hiring decisions and never makes them.

> **Status:** implementation is in progress. See [`SPEC.md`](SPEC.md) for the working spec.

## Development

```sh
uv sync                 # install dependencies
uv run fastapi dev      # API with auto-reload on http://127.0.0.1:8000 (HR workbench at /, docs at /docs)
uv run pytest           # run tests
```

The API lives in `src/agentic_talent_screener/api/`. To add endpoints, create a router module in `api/routers/` and include it in `api/routers/v1.py` (served under `/api/v1`). Settings are read from `ATS_*` environment variables or a `.env` file (see `api/config.py`).

## Calling OpenRouter for LLM

```
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

Free model with OpenRouter - `google/gemma-4-31b-it:free`

The code calls models through `src/agentic_talent_screener/llm.py`. Put your key in the git-ignored `.env` file at the project root:

```sh
OPENROUTER_API_KEY=sk-or-...
ATS_LLM_MODEL=google/gemma-4-31b-it:free   # optional; this is the default
ATS_LLM_MAX_RETRIES=4                      # optional; retries on 429 / 5xx / network errors
```

## Draft a requirements profile

```sh
uv run python -m agentic_talent_screener.job_analysis path/to/job-description.txt
```

The Job Analysis Agent sends the full UTF-8 job description to the configured OpenRouter model (override with `--model`) and prints an unapproved `RequirementProfile` as JSON. Check the quoted qualifications, inactive suggestions, and reviewer flags before approval. This standalone command is not wired to the HR workbench; it neither approves nor scores anything. Python callers can use `job_analysis.draft_requirement_profile(jd_text)`.

See the [integration testing guide](docs/integration-testing-guide.md) for offline tests and a live smoke test using a tiny synthetic JD.

## Using Jev with Open Router


Use typesafe/jev-1.14 with the OpenRouter API:

This model does not generate text. It answers typed questions about a state (a string, object, or array) through the Decisions API and returns calibrated probabilities: a yes/no probability (noul), a pick from options you define (choice), or a position on an ordered rubric (score). Your code owns the workflow and acts on the answers, so use it for routing, ranking, verification, and other structured decisions rather than chat.

```
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

## Team

Neeraj Gupta · Vaibhav Gupta · Rahul Chhiber · Hardik Vyas · Anut Verma
