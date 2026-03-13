# Copyright (c) 2026 Dedalus Labs, Inc. and its contributors
# SPDX-License-Identifier: MIT

"""Smoke test tools for validating MCP handshake."""

from dedalus_mcp import tool
from dedalus_mcp.types import ToolAnnotations
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class PingResult:
    """Smoke ping response."""

    ok: bool = True
    message: str = "pong"


@tool(
    description="Smoke test ping (no downstream dispatch required)",
    tags=["smoke", "health"],
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
async def smoke_ping(message: str = "pong") -> PingResult:
    """Simple ping for testing MCP connection."""
    return PingResult(message=message)


smoke_tools = [smoke_ping]
