import pytest

from krkn_ai.utils.prometheus import _validate_and_create_client
from krkn_ai.models.custom_errors import PrometheusConnectionError


def test_prometheus_success_first_try(monkeypatch):
    monkeypatch.delenv("MOCK_FITNESS", raising=False)

    mock_client = type("MockClient", (), {})()
    mock_client.process_query = lambda _: None

    monkeypatch.setattr(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        lambda url, token: mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client is mock_client


def test_prometheus_retry_then_success(monkeypatch):
    monkeypatch.delenv("MOCK_FITNESS", raising=False)

    calls = {"count": 0}

    def process_query(_):
        calls["count"] += 1
        if calls["count"] < 3:
            raise Exception("temporary failure")

    mock_client = type("MockClient", (), {})()
    mock_client.process_query = process_query

    monkeypatch.setattr(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        lambda url, token: mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client is mock_client
    assert calls["count"] == 3


def test_prometheus_retry_exhausted(monkeypatch):
    monkeypatch.delenv("MOCK_FITNESS", raising=False)

    def process_query(_):
        raise Exception("always down")

    mock_client = type("MockClient", (), {})()
    mock_client.process_query = process_query

    monkeypatch.setattr(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        lambda url, token: mock_client,
    )

    with pytest.raises(PrometheusConnectionError):
        _validate_and_create_client("http://prom", "token")


def test_mock_fitness_skips_prometheus(monkeypatch):
    monkeypatch.setenv("MOCK_FITNESS", "true")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    mock_client = type("MockClient", (), {})()

    monkeypatch.setattr(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        lambda url, token: mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client is mock_client
