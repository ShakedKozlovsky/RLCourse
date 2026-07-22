"""HTTP transport for MCP client — tests the real cross-network path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_build_http_transport_returns_callable() -> None:
    from marl_lab.mcp.http_transport import build_http_transport
    tr = build_http_transport("http://peer/tool", token="abc")
    assert callable(tr)


def test_http_transport_posts_with_auth_header() -> None:
    """Verify Authorization: Bearer + payload wrapping + FastMCP result unwrap."""
    from marl_lab.mcp.http_transport import build_http_transport
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": json.dumps({
        "action": 3, "q_value_for_action": 0.5, "server_role": "cop",
    })}
    mock_response.text = "..."
    with patch("httpx.post", return_value=mock_response) as mock_post:
        tr = build_http_transport("http://peer/select_action", token="secret-tk")
        payload = json.dumps({"agent_role": "cop", "observation": [0.0],
                                "episode_step": 0, "auth_token": "secret-tk"})
        result = tr(payload)
    # Verify the HTTP call
    args, kwargs = mock_post.call_args
    assert args[0] == "http://peer/select_action"
    assert kwargs["headers"]["Authorization"] == "Bearer secret-tk"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    # Verify the payload was wrapped
    sent_body = json.loads(kwargs["content"])
    assert "payload" in sent_body
    assert json.loads(sent_body["payload"])["agent_role"] == "cop"
    # Verify the result was unwrapped
    resp_dict = json.loads(result)
    assert resp_dict["action"] == 3


def test_http_transport_raises_on_non_2xx() -> None:
    from marl_lab.mcp.http_transport import build_http_transport
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    with patch("httpx.post", return_value=mock_response):
        tr = build_http_transport("http://peer", token="bad")
        with pytest.raises(RuntimeError, match="401"):
            tr("{}")


def test_http_transport_raises_on_connection_error() -> None:
    """If httpx.post throws, we wrap in a clean RuntimeError."""
    import httpx

    from marl_lab.mcp.http_transport import build_http_transport
    with patch("httpx.post", side_effect=httpx.ConnectError("connect refused")):
        tr = build_http_transport("http://not-reachable", token="tk")
        with pytest.raises(RuntimeError, match="HTTP transport failed"):
            tr("{}")


def test_http_transport_handles_bare_response_no_result_wrapper() -> None:
    """If the peer returns a raw dict (no 'result' key), pass through as text."""
    from marl_lab.mcp.http_transport import build_http_transport
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": 2, "q_value_for_action": 0.1,
                                          "server_role": "thief"}
    mock_response.text = json.dumps(mock_response.json.return_value)
    with patch("httpx.post", return_value=mock_response):
        tr = build_http_transport("http://peer", token="tk")
        result = tr("{}")
    # Should still contain the action
    assert "action" in result
