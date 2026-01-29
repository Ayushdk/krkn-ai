import pytest
from krkn_ai.utils.prometheus import _validate_and_create_client
from krkn_ai.models.custom_errors import PrometheusConnectionError


def test_prometheus_success_first_try(mocker):
    mocker.patch("krkn_ai.utils.prometheus.env_is_truthy", return_value=False)

    mock_client = mocker.Mock()
    mock_client.process_query.return_value = None

    mocker.patch(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        return_value=mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client == mock_client
    mock_client.process_query.assert_called_once()


def test_prometheus_retry_then_success(mocker):
    mocker.patch("krkn_ai.utils.prometheus.env_is_truthy", return_value=False)

    mock_client = mocker.Mock()
    mock_client.process_query.side_effect = [
        Exception("fail-1"),
        Exception("fail-2"),
        None,
    ]

    mocker.patch(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        return_value=mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client == mock_client
    assert mock_client.process_query.call_count == 3


def test_prometheus_retry_exhausted(mocker):
    mocker.patch("krkn_ai.utils.prometheus.env_is_truthy", return_value=False)

    mock_client = mocker.Mock()
    mock_client.process_query.side_effect = Exception("always down")

    mocker.patch(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        return_value=mock_client,
    )

    with pytest.raises(PrometheusConnectionError):
        _validate_and_create_client("http://prom", "token")


def test_mock_fitness_skips_prometheus(mocker):
    mocker.patch("krkn_ai.utils.prometheus.env_is_truthy", return_value=True)

    mock_client = mocker.Mock()

    mocker.patch(
        "krkn_ai.utils.prometheus.KrknPrometheus",
        return_value=mock_client,
    )

    client = _validate_and_create_client("http://prom", "token")

    assert client == mock_client
    mock_client.process_query.assert_not_called()
