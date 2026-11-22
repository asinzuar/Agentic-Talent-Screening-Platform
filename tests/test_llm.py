from __future__ import annotations

import pytest

from agentic_talent_screener import llm


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> list[float]:
    # Keep the developer's real .env and shell settings out of these tests.
    monkeypatch.chdir(tmp_path)
    for name in (
        "OPENROUTER_API_KEY", "ATS_OPENROUTER_API_KEY", "ATS_LLM_MODEL", "ATS_LLM_MAX_RETRIES", "ATS_OPENROUTER_BASE_URL"
    ):
        monkeypatch.delenv(name, raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return sleeps


class _Response:
    def __init__(self, body: dict, status_code: int = 200, headers: dict | None = None) -> None:
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise llm.requests.HTTPError(self.text)

    def json(self) -> dict:
        return self._body


def _call() -> dict:
    return llm.chat_json(
        [{"role": "user", "content": "hi"}], schema={"type": "object"}, schema_name="t", max_tokens=10, timeout=5
    )


def test_default_model_is_gemma_and_env_overrides_it(monkeypatch) -> None:
    assert llm.default_llm_model() == "google/gemma-4-31b-it:free"
    monkeypatch.setenv("ATS_LLM_MODEL", "openai/gpt-4o")
    assert llm.default_llm_model() == "openai/gpt-4o"


def test_settings_are_read_from_dotenv_file(tmp_path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\nATS_LLM_MODEL=some/model\n")
    settings = llm.LLMSettings()
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "from-file"
    assert settings.llm_model == "some/model"


def test_missing_or_empty_api_key_fails_before_any_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: pytest.fail("no request expected"))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _call()
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=\nATS_LLM_MODEL=\n")
    assert llm.default_llm_model() == llm.DEFAULT_LLM_MODEL
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _call()


def test_request_goes_to_openrouter_with_key_model_and_schema(monkeypatch) -> None:
    captured: dict = {}

    def post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)
        return _Response({"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("ATS_LLM_MODEL", "env/model")
    monkeypatch.setattr(llm.requests, "post", post)

    assert _call() == {"ok": True}
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "env/model"
    assert "reasoning" not in captured["json"]
    assert captured["json"]["response_format"]["json_schema"]["strict"] is False
    assert captured["json"]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert captured["json"]["response_format"]["json_schema"]["strict"] is False


_OK = {"choices": [{"message": {"content": '{"ok": true}'}}]}


def _scripted_post(monkeypatch, outcomes: list) -> list[int]:
    calls: list[int] = []

    def post(*_args, **_kwargs):
        calls.append(1)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(llm.requests, "post", post)
    return calls


def test_rate_limits_and_server_errors_are_retried_with_backoff(monkeypatch, _isolated_env) -> None:
    calls = _scripted_post(monkeypatch, [
        _Response({"error": "rate limited"}, 429),
        _Response({"error": "rate limited"}, 429, headers={"Retry-After": "7"}),
        _Response({"error": "bad gateway"}, 502),
        _Response(_OK),
    ])
    assert _call() == {"ok": True}
    assert len(calls) == 4
    sleeps = _isolated_env
    assert len(sleeps) == 3
    assert 1.0 <= sleeps[0] <= 2.0  # base delay with jitter
    assert sleeps[1] == 7.0  # Retry-After honoured
    assert 4.0 <= sleeps[2] <= 8.0  # backoff doubles


def test_gives_up_after_configured_retries(monkeypatch, _isolated_env) -> None:
    monkeypatch.setenv("ATS_LLM_MAX_RETRIES", "2")
    calls = _scripted_post(monkeypatch, [_Response({"error": "rate limited"}, 429)] * 3)
    with pytest.raises(RuntimeError, match="429"):
        _call()
    assert len(calls) == 3
    assert len(_isolated_env) == 2


def test_network_errors_are_retried_then_reported(monkeypatch) -> None:
    monkeypatch.setenv("ATS_LLM_MAX_RETRIES", "1")
    calls = _scripted_post(monkeypatch, [llm.requests.ConnectionError("down"), llm.requests.ConnectionError("down")])
    with pytest.raises(ConnectionError, match="Cannot reach OpenRouter"):
        _call()
    assert len(calls) == 2


def test_client_errors_are_not_retried(monkeypatch, _isolated_env) -> None:
    calls = _scripted_post(monkeypatch, [_Response({"error": "unauthorised"}, 401)])
    with pytest.raises(RuntimeError, match="401"):
        _call()
    assert len(calls) == 1
    assert _isolated_env == []


def test_missing_content_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Response({"error": {"message": "bad"}}))
    with pytest.raises(RuntimeError, match="top_level_keys"):
        _call()


def test_text_content_parts_are_normalised(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *a, **k: _Response({
            "choices": [{"message": {"content": [{"type": "text", "text": "{\"ok\": true}"}]}}]
        }),
    )

    assert _call() == {"ok": True}


def test_http_error_does_not_echo_provider_body(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *a, **k: _Response({"error": {"message": "sensitive provider text", "metadata": {"user_id": "private"}}}, 429),
    )

    with pytest.raises(RuntimeError) as error:
        _call()

    assert "sensitive provider text" not in str(error.value)
    assert "private" not in str(error.value)
    assert "error_metadata_keys" in str(error.value)
