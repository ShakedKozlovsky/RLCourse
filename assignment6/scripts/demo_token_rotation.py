"""Demonstrate MCP token rotation + revocation (spec § 5.3).

Spec requires the auth system to support token revocation. This script
exercises the full life-cycle end-to-end:

  1. Issue token v1, register, request → success
  2. Rotate: issue v2, request with v1 (still valid) → success
  3. Revoke v1, request with v1 → rejected
  4. Request with v2 → success
  5. Revoke v2, request with v2 → rejected (registry now empty)
  6. Request with any token → rejected (deny-all)

Output: assets/logs/token_rotation.log — a CLI-style transcript a TA can
read to verify the auth lifecycle without re-running the script."""

from __future__ import annotations

import json
from pathlib import Path

from marl_lab.auth.token_registry import TokenRegistry
from marl_lab.mcp.protocol import SelectActionRequest
from marl_lab.mcp.server_base import CopMCPServer, UnauthorizedError
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "assets" / "logs" / "token_rotation.log"


def _req(token: str) -> SelectActionRequest:
    """A minimal valid request with the given token."""
    return SelectActionRequest(
        agent_role="cop",
        observation=[0.0] * obs_dim(2),
        episode_step=0,
        auth_token=token,
    )


def main() -> int:
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    # Setup
    q_net = QPerAgent(obs_dim=obs_dim(2), n_actions=6,
                       hidden_sizes=(16,), gru_hidden_size=8)
    registry = TokenRegistry(tokens=["v1-secret"])
    server = CopMCPServer(q_net=q_net, token_registry=registry)
    log("# MCP token rotation life-cycle demo")
    log(f"# initial registry size: {len(registry)} (token: v1-secret)")
    log("")

    # Stage 1: v1 works
    log("--- STAGE 1: issue v1, request with v1 ---")
    try:
        resp = server.select_action(_req("v1-secret"))
        log(f"OK request(v1-secret) → action={resp.action}, server_role={resp.server_role}")
    except UnauthorizedError as e:
        log(f"FAIL unexpected: {e}")
        return 1

    # Stage 2: add v2 alongside v1 (both should now work)
    log("")
    log("--- STAGE 2: issue v2 alongside v1 (rotation begins) ---")
    registry = TokenRegistry(tokens=["v1-secret", "v2-secret"])
    server.tokens = registry
    log(f"# registry: {len(registry)} tokens active")
    for tk in ("v1-secret", "v2-secret"):
        try:
            resp = server.select_action(_req(tk))
            log(f"OK request({tk}) → action={resp.action}")
        except UnauthorizedError as e:
            log(f"FAIL unexpected on {tk}: {e}")
            return 1

    # Stage 3: revoke v1 (only v2 remains)
    log("")
    log("--- STAGE 3: revoke v1 (rotation complete) ---")
    registry = TokenRegistry(tokens=["v2-secret"])
    server.tokens = registry
    log(f"# registry: {len(registry)} token (v2-secret)")
    # v1 should now be rejected
    try:
        server.select_action(_req("v1-secret"))
        log("FAIL: v1-secret should have been rejected after revoke")
        return 1
    except UnauthorizedError as e:
        log(f"OK request(v1-secret) → REJECTED — {type(e).__name__}: {e}")
    # v2 still works
    try:
        resp = server.select_action(_req("v2-secret"))
        log(f"OK request(v2-secret) → action={resp.action}")
    except UnauthorizedError as e:
        log(f"FAIL unexpected on v2: {e}")
        return 1

    # Stage 4: revoke v2 (deny-all)
    log("")
    log("--- STAGE 4: revoke v2 (registry empty — deny-all) ---")
    registry = TokenRegistry(tokens=[])
    server.tokens = registry
    log(f"# registry: {len(registry)} tokens (empty allowlist)")
    for tk in ("v1-secret", "v2-secret", "anything-else"):
        try:
            server.select_action(_req(tk))
            log(f"FAIL: {tk} should have been rejected")
            return 1
        except UnauthorizedError as e:
            log(f"OK request({tk}) → REJECTED — {type(e).__name__}: {e}")

    log("")
    log("--- END: all stages passed — rotation + revocation lifecycle verified ---")

    # Write transcript
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"\n[demo] transcript saved to {LOG_PATH}")
    # Also a tiny JSON summary
    summary_path = LOG_PATH.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "stages": 4,
        "successful_requests": 4,
        "rejected_requests": 4,
        "all_assertions_held": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
