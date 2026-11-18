# Agentic Talent Screener

Capstone project for **AI-ML in Practice** (IIT Bombay), Team 30.

An agentic LLM system for screening job applicants that:

- **ranks the whole applicant pool on verifiable evidence** from each resume, not keyword matches;
- **explains every decision** with cited resume spans so recruiters, hiring managers and auditors can check it;
- **resists manipulation** such as prompt injection, hidden text and keyword stuffing, using layered defences, and measures that robustness with an adversarial evaluation harness;
- **keeps a human in charge.** It supports hiring decisions and never makes them.

> **Status:** early specification. Nothing is implemented yet. See [`SPEC.md`](SPEC.md) for the working spec.

## Development

```sh
uv sync                 # install dependencies
uv run fastapi dev      # API with auto-reload on http://127.0.0.1:8000 (docs at /docs)
uv run pytest           # run tests
```

The API lives in `src/agentic_talent_screener/api/`. To add endpoints, create a router module in `api/routers/` and include it in `api/routers/v1.py` (served under `/api/v1`). Settings are read from `ATS_*` environment variables or a `.env` file (see `api/config.py`).

## Team

Neeraj Gupta · Vaibhav Gupta · Rahul Chhiber · Hardik Vyas · Anut Verma
