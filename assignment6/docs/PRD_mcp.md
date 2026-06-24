# PRD — MCP servers + cloud architecture

> Per-mechanism PRD for the **two-MCP-servers** infrastructure. Spec § 5.3 + § 8 are the contracts.

## 1. The hard requirement

> *"Each agent — Cop and Thief — is an **autonomous AI agent**, with its own MCP server, unique and separate from the other. The two agents communicate via HTTP requests to each other's MCP server."* — spec § 5.3

This means:

| Component | Where |
|---|---|
| Cop MCP server | `src/marl_lab/mcp/cop_server.py` — runs on cop_host:cop_port |
| Thief MCP server | `src/marl_lab/mcp/thief_server.py` — runs on thief_host:thief_port |
| Shared protocol schemas | `src/marl_lab/mcp/protocol.py` — pydantic models |
| Generic client | `src/marl_lab/mcp/client.py` — handles auth + retries |
| Auth token registry | `src/marl_lab/auth/token_registry.py` |

The game adjudicator (the local `game_runner`) orchestrates one game by:
1. Initialising both MCP servers (locally or in cloud).
2. Per turn: POSTing each agent's current observation to its MCP server; collecting the action; resolving the move.
3. Aggregating into a `GameReport`; sending via Gmail.

## 2. Two-phase deployment (spec § 5.3 + § 8)

### Phase 1 — Localhost (mandatory)

```bash
# Terminal 1
marl-lab serve --role cop --port 7301
# Terminal 2
marl-lab serve --role thief --port 7302
# Terminal 3
marl-lab play --mode mcp-localhost
```

Validates: full end-to-end game runs through HTTP between two separate processes.

### Phase 2 — Cloud (mandatory but skippable with documentation)

```bash
# Deploy both servers to Prefect Cloud
marl-lab serve --role cop --deploy prefect-cloud
marl-lab serve --role thief --deploy prefect-cloud
# Returns two public URLs + auth tokens

# Game adjudicator (running anywhere) plays via the public URLs
marl-lab play --mode mcp-cloud \
              --cop-url https://api.prefect.cloud/.../cop \
              --thief-url https://api.prefect.cloud/.../thief
```

The lecturer (or another student) can verify the agents are running by hitting `GET /healthz` with a token.

## 3. Protocol (FastMCP message shapes)

```python
class MoveRequest(BaseModel):
    obs: list[float]                # flattened local observation
    hidden_token: str | None        # opaque server-side hidden-state handle
    metadata: dict                  # game_id, sub_game_id, step

class MoveResponse(BaseModel):
    action: Literal["UP", "DOWN", "LEFT", "RIGHT", "STAY", "PLACE_BARRIER"]
    next_hidden_token: str
    q_values: list[float] | None    # optional, for debug logging

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    role: Literal["cop", "thief"]
    model_version: str
    uptime_s: float

class RevokeRequest(BaseModel):
    token: str                       # admin-token guarded
```

## 4. Authentication (spec § 5.3 — token-based, revokable)

| Concern | Approach |
|---|---|
| Token issuance | Admin-only `marl-lab auth issue --label "for-grader-2026"` → prints token |
| Token storage | `MARL_MCP_ALLOWED_TOKENS` env var (comma-separated) OR `secrets/tokens.json` |
| Verification | `Authorization: Bearer <tok>` header; rejected with 401 if missing/invalid |
| Revocation | `POST /admin/revoke {"token": "..."}` with admin-token; removes from registry |
| Audit log | Every request logs `(timestamp, token_label, endpoint, status)` to `assets/logs/mcp_audit.log` |

## 5. Per-agent server tools (FastMCP `@mcp.tool` decorators)

| Tool | Purpose |
|---|---|
| `move(obs, hidden_token, metadata)` | Main inference endpoint. Returns the agent's action. |
| `reset(seed, sub_game_id)` | Clears hidden state for the new sub-game. |
| `health()` | Liveness + readiness probe. |
| `model_info()` | Returns model_version, checkpoint hash, training metadata. |
| `admin_revoke(token)` | Admin-only — removes a token from the registry. |

Both servers expose the same tools, just with different default action sets (cop has +PLACE_BARRIER).

## 6. Game adjudicator (over MCP)

```python
async def play_one_game_over_mcp(cop_url, thief_url, cop_token, thief_token, game_cfg):
    cop = McpClient(cop_url, cop_token)
    thief = McpClient(thief_url, thief_token)
    report = GameReport(...)
    for sub_id in range(1, game_cfg.num_games + 1):
        env.reset()
        await cop.reset(seed=sub_id)
        await thief.reset(seed=sub_id)
        for t in range(game_cfg.max_moves):
            cop_obs = env.partial_observation("cop")
            thief_obs = env.partial_observation("thief")
            # Query both agents in parallel
            cop_resp, thief_resp = await asyncio.gather(
                cop.move(cop_obs, hidden_token_cop),
                thief.move(thief_obs, hidden_token_thief),
            )
            obs, rewards, done = env.step({cop: cop_resp.action, thief: thief_resp.action})
            if done: break
        report.add_sub_game(...)
    return report
```

## 7. Test plan

| Test | Pass criterion |
|---|---|
| `protocol.py` schemas | pydantic round-trip serialisation |
| Cop server `GET /healthz` | 200 OK without token (health is public) |
| Cop server `POST /move` without token | 401 |
| Cop server `POST /move` with valid token | 200 + valid `MoveResponse` |
| Revoked token | 401 after `admin_revoke` |
| Client `McpClient.move()` | Sends Bearer token; raises on timeout |
| End-to-end localhost game | full 6-sub-game game runs without errors |
| Cloud-stub mode | Without `PREFECT_API_KEY`, deploy prints stub guide + does not crash |

## 8. Acceptance criteria

1. Two MCP servers run simultaneously on localhost (different ports) — both report 200 on `/healthz`.
2. The game adjudicator successfully drives a game by HTTP calls.
3. Token revocation works end-to-end.
4. README documents the 4-step deployment guide (§ 8 of the spec).
5. `assets/logs/mcp_session.log` shows real successful + rejected requests.

## 9. Non-goals

- OAuth-token authentication (overkill for course assignment; documented as future-work).
- Rate limiting (V3 § 5 mentions it; not in spec § 5.3 scope).
- Persistent server state across restarts (hidden_token lives only for the duration of the server process).
- Multi-region cloud deployment.

## 10. Citation

Anthropic, *Model Context Protocol Specification*. https://modelcontextprotocol.io, 2024.
