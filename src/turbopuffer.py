# Copyright (c) 2026 Dedalus Labs, Inc. and its contributors
# SPDX-License-Identifier: MIT

"""Turbopuffer API tools for MCP."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from dedalus_mcp import HttpMethod, HttpRequest, get_context, tool
from dedalus_mcp.auth import Connection, SecretKeys
from dedalus_mcp.types import ToolAnnotations
from pydantic import Field
from pydantic.dataclasses import dataclass

DEFAULT_BASE_URL = "https://gcp-us-central1.turbopuffer.com"

turbopuffer = Connection(
    name="turbopuffer",
    secrets=SecretKeys(api_key="TURBOPUFFER_API_KEY"),
    base_url=os.getenv("TURBOPUFFER_BASE_URL", DEFAULT_BASE_URL),
    auth_header_format="Bearer {api_key}",
)

READ_ONLY = ToolAnnotations(readOnlyHint=True)
READ_ONLY_IDEMPOTENT = ToolAnnotations(readOnlyHint=True, idempotentHint=True)


@dataclass(frozen=True)
class TurbopufferResult:
    """Normalized Turbopuffer API response."""

    success: bool
    status: int | None = None
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


def _ns(namespace: str) -> str:
    """Encode namespace for safe URL path usage."""
    return quote(namespace, safe="")


def _query_value(value: Any) -> str:
    """Normalize query values without pre-encoding."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _query(params: dict[str, Any] | None) -> str:
    """Build query string from optional values."""
    if not params:
        return ""

    parts: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(f"{key}={_query_value(item)}" for item in value)
            continue
        parts.append(f"{key}={_query_value(value)}")
    return "&".join(parts)


def _extract_error(body: Any, status: int | None) -> str:
    """Extract human-readable error message from HTTP body."""
    fallback = f"Turbopuffer request failed with status {status}" if status is not None else "Turbopuffer request failed"
    if isinstance(body, dict):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, str) and first.strip():
                return first
            if isinstance(first, dict):
                for key in ("error", "message", "detail"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
    if isinstance(body, str) and body.strip():
        return body
    return fallback


async def _req(method: HttpMethod, path: str, body: Any = None) -> TurbopufferResult:
    """Execute Turbopuffer API request via DAuth dispatch."""
    ctx = get_context()
    resp = await ctx.dispatch("turbopuffer", HttpRequest(method=method, path=path, body=body))
    if not resp.success:
        return TurbopufferResult(success=False, error=resp.error.message if resp.error else "Dispatch failed")

    response = resp.response
    if response is None:
        return TurbopufferResult(success=False, error="Empty response from downstream")

    status = response.status
    response_body = response.body
    meta = {"headers": response.headers, "status": status}
    if status >= 400:
        return TurbopufferResult(
            success=False,
            status=status,
            data=response_body,
            error=_extract_error(response_body, status),
            meta=meta,
        )
    return TurbopufferResult(success=True, status=status, data=response_body, meta=meta)


@tool(
    description="List Turbopuffer namespaces.",
    tags=["namespace", "read"],
    annotations=READ_ONLY_IDEMPOTENT,
)
async def turbopuffer_list_namespaces(
    cursor: str | None = None,
    prefix: str | None = None,
    page_size: int | None = None,
) -> TurbopufferResult:
    """List namespaces in the configured Turbopuffer account."""
    query = _query({"cursor": cursor, "prefix": prefix, "page_size": page_size})
    path = "/v1/namespaces"
    if query:
        path = f"{path}?{query}"
    return await _req(HttpMethod.GET, path)


@tool(
    description="Get metadata for a Turbopuffer namespace.",
    tags=["namespace", "read"],
    annotations=READ_ONLY_IDEMPOTENT,
)
async def turbopuffer_get_namespace_metadata(namespace: str) -> TurbopufferResult:
    """Get namespace metadata, including schema and approximate row count."""
    return await _req(HttpMethod.GET, f"/v1/namespaces/{_ns(namespace)}/metadata")


@tool(
    description="Get schema for a Turbopuffer namespace.",
    tags=["namespace", "read"],
    annotations=READ_ONLY_IDEMPOTENT,
)
async def turbopuffer_get_namespace_schema(namespace: str) -> TurbopufferResult:
    """Get namespace attribute schema."""
    return await _req(HttpMethod.GET, f"/v1/namespaces/{_ns(namespace)}/schema")


@tool(
    description="Update schema for a Turbopuffer namespace.",
    tags=["namespace", "write"],
)
async def turbopuffer_update_namespace_schema(namespace: str, schema: dict[str, Any]) -> TurbopufferResult:
    """Update namespace schema."""
    return await _req(HttpMethod.POST, f"/v1/namespaces/{_ns(namespace)}/schema", body=schema)


@tool(
    description="Create, update, or delete documents in a Turbopuffer namespace.",
    tags=["write"],
)
async def turbopuffer_write(namespace: str, write: dict[str, Any]) -> TurbopufferResult:
    """Execute a /v2 write payload against a namespace."""
    return await _req(HttpMethod.POST, f"/v2/namespaces/{_ns(namespace)}", body=write)


@tool(
    description="Query documents in a Turbopuffer namespace.",
    tags=["query", "read"],
    annotations=READ_ONLY,
)
async def turbopuffer_query(
    namespace: str,
    query: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> TurbopufferResult:
    """Run a Turbopuffer query with optional query config."""
    payload = dict(query)
    if config:
        payload.update(config)
    return await _req(HttpMethod.POST, f"/v2/namespaces/{_ns(namespace)}/query", body=payload)


@tool(
    description="Run multiple concurrent queries in a Turbopuffer namespace.",
    tags=["query", "read"],
    annotations=READ_ONLY,
)
async def turbopuffer_multi_query(
    namespace: str,
    queries: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> TurbopufferResult:
    """Run multiple concurrent queries against one namespace."""
    payload: dict[str, Any] = {"queries": queries}
    if config:
        payload.update(config)
    path = f"/v2/namespaces/{_ns(namespace)}/query?stainless_overload=multiQuery"
    return await _req(HttpMethod.POST, path, body=payload)


@tool(
    description="Explain a Turbopuffer query plan.",
    tags=["query", "read"],
    annotations=READ_ONLY_IDEMPOTENT,
)
async def turbopuffer_explain_query(
    namespace: str,
    query: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> TurbopufferResult:
    """Explain query plan for debugging and optimization."""
    payload = dict(query)
    if config:
        payload.update(config)
    return await _req(HttpMethod.POST, f"/v2/namespaces/{_ns(namespace)}/explain_query", body=payload)


@tool(
    description="Delete a Turbopuffer namespace and all its documents.",
    tags=["namespace", "write"],
)
async def turbopuffer_delete_namespace(namespace: str) -> TurbopufferResult:
    """Delete an entire namespace."""
    return await _req(HttpMethod.DELETE, f"/v2/namespaces/{_ns(namespace)}")


@tool(
    description="Signal Turbopuffer to warm cache for low-latency requests.",
    tags=["namespace", "read", "performance"],
    annotations=READ_ONLY_IDEMPOTENT,
)
async def turbopuffer_cache_warm(namespace: str) -> TurbopufferResult:
    """Warm namespace cache."""
    return await _req(HttpMethod.GET, f"/v1/namespaces/{_ns(namespace)}/hint_cache_warm")


@tool(
    description="Measure approximate-nearest-neighbor recall for a namespace.",
    tags=["query", "read", "debug"],
    annotations=READ_ONLY,
)
async def turbopuffer_measure_recall(
    namespace: str,
    num: int | None = None,
    top_k: int | None = None,
    filters: Any = None,
    include_ground_truth: bool = False,
) -> TurbopufferResult:
    """Run recall measurement using the Turbopuffer debug endpoint."""
    payload: dict[str, Any] = {"include_ground_truth": include_ground_truth}
    if num is not None:
        payload["num"] = num
    if top_k is not None:
        payload["top_k"] = top_k
    if filters is not None:
        payload["filters"] = filters
    return await _req(HttpMethod.POST, f"/v1/namespaces/{_ns(namespace)}/_debug/recall", body=payload)


@tool(
    description=(
        "Export namespace data using Turbopuffer's deprecated export endpoint. "
        "Prefer paged query APIs when possible."
    ),
    tags=["namespace", "read", "export"],
    annotations=READ_ONLY,
)
async def turbopuffer_export_documents(
    namespace: str,
    params: dict[str, Any] | None = None,
) -> TurbopufferResult:
    """Export rows from namespace (deprecated endpoint compatibility)."""
    path = f"/v1/namespaces/{_ns(namespace)}"
    query = _query(params)
    if query:
        path = f"{path}?{query}"
    return await _req(HttpMethod.GET, path)


turbopuffer_tools = [
    turbopuffer_list_namespaces,
    turbopuffer_get_namespace_metadata,
    turbopuffer_get_namespace_schema,
    turbopuffer_update_namespace_schema,
    turbopuffer_write,
    turbopuffer_query,
    turbopuffer_multi_query,
    turbopuffer_explain_query,
    turbopuffer_delete_namespace,
    turbopuffer_cache_warm,
    turbopuffer_measure_recall,
    turbopuffer_export_documents,
]
