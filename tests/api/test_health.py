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
