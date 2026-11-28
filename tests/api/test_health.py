import logging

from fastapi.testclient import TestClient

from agentic_talent_screener import __version__
from agentic_talent_screener.api import create_app
from agentic_talent_screener.api.config import Settings


def test_health() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "environment": "test"}


def test_openapi_schema_is_served() -> None:
    client = TestClient(create_app())

    assert client.get("/openapi.json").status_code == 200


def test_server_appends_safe_access_and_error_logs(tmp_path, caplog) -> None:
    path = tmp_path / "server.log"
    path.write_text("Earlier server log line\n")
    server = logging.getLogger("uvicorn.error")
    access = logging.getLogger("uvicorn.access")
    query = "private-query-value"
    detail = "private-exception-detail"
    with (
        caplog.at_level(logging.INFO, logger=server.name),
        caplog.at_level(logging.INFO, logger=access.name),
        TestClient(create_app(Settings(log_dir=tmp_path))) as client,
    ):
        assert client.get("/health").status_code == 200
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:1234", "GET", f"/health?value={query}", "1.1", 200)
        try:
            raise ValueError(detail)
        except ValueError:
            server.exception("Request failed")

    text = path.read_text()
    assert "Earlier server log line" in text
    assert 'GET /health HTTP/1.1" 200' in text
    assert "Request failed" in text and "ValueError" in text
    assert query not in text and detail not in text
    assert "Talent Screener API started" in caplog.text and "Talent Screener API stopped" in text
