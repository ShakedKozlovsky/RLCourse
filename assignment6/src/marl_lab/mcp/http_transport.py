"""HTTP transport for the MCP client — real cross-network communication.

The core ``MCPClient`` (mcp/client.py) is transport-agnostic: it takes a
``(payload_json: str) → response_json: str`` callable. In-process tests
wrap ``server.select_action`` directly; this module provides the real
HTTP variant that talks to a live FastMCP-served ``select_action`` tool.

For the spec § 9 inter-group bonus match, this is what lets one team's
adjudicator drive the other team's MCP server."""

from __future__ import annotations

import json
from collections.abc import Callable


def build_http_transport(server_url: str, token: str,
                          timeout_s: float = 10.0) -> Callable[[str], str]:
    """Build a transport callable that POSTs the payload to ``server_url``.

    The URL is expected to be the full FastMCP tool endpoint, e.g.
    ``https://team-beta.horizon.prefect.io/mcp/tools/select_action``.
    Auth is via a Bearer token; the token is also included in the JSON
    payload so the server's ``TokenRegistry`` guard can validate it
    without HTTP-level header parsing.

    Uses ``httpx`` (already in project deps). Raises ``RuntimeError`` on
    non-2xx responses with the server's error body attached — helpful for
    debugging cross-network agreement issues at match time."""
    import httpx

    def transport(payload_json: str) -> str:
        # Payload already has auth_token embedded (SelectActionRequest field);
        # we ALSO send it as a Bearer header for FastMCP endpoints that
        # enforce header-level auth.
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # FastMCP tool endpoints expect {"payload": "<json-string>"} wrapper
        wrapped = json.dumps({"payload": payload_json})
        try:
            resp = httpx.post(server_url, content=wrapped, headers=headers,
                                timeout=timeout_s)
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP transport failed: {e}") from e
        if resp.status_code >= 300:
            raise RuntimeError(
                f"peer MCP server returned {resp.status_code}: {resp.text[:200]}"
            )
        # FastMCP wraps tool outputs; unwrap the payload
        try:
            body = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"peer returned non-JSON: {resp.text[:200]}") from e
        if isinstance(body, dict) and "result" in body:
            result = body["result"]
            return result if isinstance(result, str) else json.dumps(result)
        return resp.text
    return transport
