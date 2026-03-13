# turbopuffer-mcp

A Turbopuffer MCP server built with `dedalus-mcp` and DAuth-style credential exchange.

This server follows the same DAuth north-star pattern as `x-api-mcp`:
- Credentials are provided by clients at runtime.
- The server declares a `Connection` with `SecretKeys`.
- Tool calls dispatch through Dedalus secure connection handles.

## Features

- DAuth-compatible API key authentication (`TURBOPUFFER_API_KEY`).
- Configurable Turbopuffer region/base URL via `TURBOPUFFER_BASE_URL`.
- Read + write tool coverage for the core Turbopuffer API.
- Includes a smoke ping tool for quick MCP handshake validation.

## Setup

1. Create a Turbopuffer API key.
2. Copy env template:

```bash
cp .env.example .env
```

Required:
- `TURBOPUFFER_API_KEY` (provided by the MCP client via DAuth credentials in production flows)

Optional:
- `TURBOPUFFER_BASE_URL` (defaults to `https://gcp-us-central1.turbopuffer.com`)
- `DEDALUS_AS_URL` (defaults to `https://as.dedaluslabs.ai`)
- `HOST` (defaults to `127.0.0.1`)
- `PORT` (defaults to `8080`)

## Run

```bash
uv run python src/main.py
```

## Tool Surface

- `turbopuffer_list_namespaces`
- `turbopuffer_get_namespace_metadata`
- `turbopuffer_get_namespace_schema`
- `turbopuffer_update_namespace_schema`
- `turbopuffer_write`
- `turbopuffer_query`
- `turbopuffer_multi_query`
- `turbopuffer_explain_query`
- `turbopuffer_delete_namespace`
- `turbopuffer_cache_warm`
- `turbopuffer_measure_recall`
- `turbopuffer_export_documents` (deprecated Turbopuffer endpoint compatibility)
- `smoke_ping`

## Notes

- Turbopuffer has both `/v1` and `/v2` endpoints. This server mirrors the live docs/API split:
  - Namespace listing/metadata/schema/cache-warm/recall/export use `/v1`.
  - Write/query/delete/explain use `/v2`.
- The export endpoint is deprecated in Turbopuffer docs in favor of paging query APIs, but is still exposed here for compatibility.
