# Integration testing guide

## Job Analysis Agent

Run the commands from the repository root. Use `uv sync` if dependencies are not installed.

### Run the automated tests (no API calls)

```sh
uv run pytest -q tests/test_job_analysis.py tests/test_llm.py
```

These tests mock OpenRouter. They check the request settings, draft validation, and rejection of invalid output, including numeric `min_years` on non-experience requirements. No API key or credits are needed.

### Run a live smoke test with Qwen

Set `OPENROUTER_API_KEY` in your environment or the git-ignored `.env` file. Use an account with credits for the paid model. The command below explicitly selects Qwen regardless of `ATS_LLM_MODEL`.

Run the stock CLI on the [tiny synthetic job description](../synthetic_test_data/tiny_jd.txt). This sends the fixture to OpenRouter; use synthetic input rather than confidential job descriptions or candidate data.

```sh
draft=$(mktemp)
time uv run python -m agentic_talent_screener.job_analysis \
  synthetic_test_data/tiny_jd.txt --model qwen/qwen3.8-flash > "$draft"
```

On success, the CLI exits with status 0 and writes a validated `RequirementProfile` as JSON. Inspect it with:

```sh
uv run python -m json.tool "$draft"
```

Check the following rather than expecting an exact JSON match:

- `approved` is `false`; approval metadata is empty.
- Python development is a mandatory (`M`), active `experience` requirement with `min_years: 1`.
- Familiarity with pytest is a preferred (`P`), active qualification with `min_years: null`.
- Quotes for JD-derived requirements appear verbatim in the fixture. Any inferred requirements or model suggestions start inactive.

The Job Analysis Agent requests strict structured output with a 20,000-token completion cap and a 360-second request timeout. It does not disable reasoning. One live run of the tiny fixture took about 43 seconds; timing varies.

If the CLI reports an OpenRouter call failure, check that the key is configured, the account has credits, and the model is available. A draft validation error means the agent rejected the response; do not treat a failed run as producing a usable draft. The smoke test does not exercise HR approval, resume matching, or ranking.
