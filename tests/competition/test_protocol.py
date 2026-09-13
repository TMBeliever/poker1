import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.competition.protocol import AgentPokerClient, APIError, Config, load_env, retry_same


def test_load_env():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("AGENTPOKER_TEST_VAR=hello_world\n# Comment\nINVALID_LINE\n")
        tf.flush()
        try:
            load_env(tf.name)
            assert os.environ.get("AGENTPOKER_TEST_VAR") == "hello_world"
        finally:
            if "AGENTPOKER_TEST_VAR" in os.environ:
                del os.environ["AGENTPOKER_TEST_VAR"]
            os.remove(tf.name)


def test_client_headers():
    cfg = Config(base_url="https://test.com", key="sk_test_123")
    client = AgentPokerClient(cfg)
    headers = client._headers()
    assert headers["Authorization"] == "Bearer sk_test_123"
    assert headers["Content-Type"] == "application/json"

    cfg_no_key = Config(base_url="https://test.com", key=None)
    client_no_key = AgentPokerClient(cfg_no_key)
    assert "Authorization" not in client_no_key._headers()


def test_client_request_success():
    client = AgentPokerClient(Config(base_url="https://test.com"))
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"competitions": [{"id": "c1", "name": "Tournament"}]}'
    mock_resp.json.return_value = {"competitions": [{"id": "c1", "name": "Tournament"}]}

    with patch.object(client.s, "request", return_value=mock_resp):
        res = client.discover("active")
        assert len(res["competitions"]) == 1
        assert res["competitions"][0]["id"] == "c1"


def test_client_request_error_handling():
    client = AgentPokerClient(Config(base_url="https://test.com"))
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.content = b'{"error": {"code": "stale_action_request", "message": "Expired"}}'
    mock_resp.json.return_value = {"error": {"code": "stale_action_request", "message": "Expired"}}
    mock_resp.headers = {"Retry-After": "2"}

    with patch.object(client.s, "request", return_value=mock_resp):
        with pytest.raises(APIError) as exc_info:
            client.action({"tableId": "t1"})
        assert exc_info.value.status == 409
        assert exc_info.value.code == "stale_action_request"
        assert exc_info.value.retry_after == 2.0


def test_retry_same_logic():
    calls = []

    def failing_fn():
        calls.append(1)
        if len(calls) < 3:
            raise APIError(503, "temporarily_unavailable", "Busy", retry_after=0.01)
        return {"status": "ok"}

    res = retry_same(failing_fn, max_attempts=5, backoff=0.01)
    assert res == {"status": "ok"}
    assert len(calls) == 3
