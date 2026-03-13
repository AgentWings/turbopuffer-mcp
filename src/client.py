"""Demo: Turbopuffer MCP for long error trace diagnosis.

Uses turbopuffer as a vector store to semantically index error trace frames,
then retrieves only the relevant ones for LLM-based root cause analysis.

Environment variables:
    DEDALUS_API_KEY: Your Dedalus API key (dsk_*)
    TURBOPUFFER_API_KEY: Your Turbopuffer API key
"""

import asyncio
import os
import re
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from dedalus_labs import AsyncDedalus, DedalusRunner  # noqa: E402
from dedalus_mcp.auth import Connection, SecretKeys, SecretValues  # noqa: E402

MODEL = "openai/gpt-5.2"
MCP_SERVER = "liangsm/turbopuffer-mcp"

turbopuffer = Connection(
    name="turbopuffer",
    secrets=SecretKeys(api_key="TURBOPUFFER_API_KEY"),
    base_url="https://gcp-us-central1.turbopuffer.com",
)

tp_secrets = SecretValues(turbopuffer, api_key=os.getenv("TURBOPUFFER_API_KEY", ""))

# Synthetic error trace: a realistic Django + SQLAlchemy stack where a DB replica
# is down. The root cause (ConnectionRefusedError at db-replica-3) is buried in
# ~20 frames of middleware, ORM, and connection pool noise.
ERROR_TRACE = """\
Traceback (most recent call last):
  File "/app/manage.py", line 22, in main
    execute_from_command_line(sys.argv)
  File "/usr/lib/python3.12/site-packages/django/core/management/__init__.py", line 442, in execute_from_command_line
    utility.execute()
  File "/usr/lib/python3.12/site-packages/django/core/handlers/base.py", line 197, in _get_response
    response = wrapped_callback(request, *callback_args, **callback_kwargs)
  File "/app/middleware/request_tracker.py", line 42, in __call__
    response = self.get_response(request)
  File "/app/middleware/auth.py", line 58, in __call__
    response = self.get_response(request)
  File "/app/middleware/rate_limiter.py", line 31, in __call__
    response = self.get_response(request)
  File "/app/middleware/correlation_id.py", line 19, in __call__
    response = self.get_response(request)
  File "/usr/lib/python3.12/site-packages/django/core/handlers/exception.py", line 55, in inner
    response = get_response(request)
  File "/app/views/checkout.py", line 134, in post
    receipt = payment_service.process_checkout(cart, user)
  File "/app/services/checkout.py", line 67, in process_checkout
    charge = self.payment.charge(cart.total, user.payment_method)
  File "/app/services/payment.py", line 87, in charge
    result = self.db.execute(query, params)
  File "/usr/lib/python3.12/site-packages/sqlalchemy/engine/base.py", line 1412, in execute
    return meth(self, multiparams, params, _EMPTY_EXECUTION_OPTS)
  File "/usr/lib/python3.12/site-packages/sqlalchemy/engine/default.py", line 620, in do_execute
    cursor.execute(statement, parameters)
  File "/usr/lib/python3.12/site-packages/sqlalchemy/pool/base.py", line 896, in _checkout
    fairy = _ConnectionRecord.checkout(pool)
  File "/usr/lib/python3.12/site-packages/sqlalchemy/pool/base.py", line 490, in checkout
    rec = pool._do_get()
  File "/usr/lib/python3.12/site-packages/sqlalchemy/pool/impl.py", line 146, in _do_get
    self._dec_overflow()
  File "/usr/lib/python3.12/site-packages/sqlalchemy/engine/create.py", line 643, in connect
    return dialect.connect(*cargs, **cparams)
  File "/usr/lib/python3.12/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
psycopg2.OperationalError: could not connect to server: Connection refused
\tIs the server running on host "db-replica-3.internal" and accepting
\tTCP/IP connections on port 5432?
"""


@dataclass
class TraceFrame:
    id: str
    file: str
    line: int
    function: str
    code: str


_FRAME_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)\n\s+(?P<code>.+)',
)


def parse_trace(trace: str) -> tuple[list[TraceFrame], str]:
    """Parse a Python traceback into structured frames + the final exception line."""
    frames = [
        TraceFrame(
            id=f"frame-{i}",
            file=m.group("file"),
            line=int(m.group("line")),
            function=m.group("func"),
            code=m.group("code").strip(),
        )
        for i, m in enumerate(_FRAME_RE.finditer(trace))
    ]
    # Extract "ExceptionType: message" — find the last line matching the pattern
    lines = trace.strip().split("\n")
    exc_start = len(lines)
    for i, line in enumerate(lines):
        if re.match(r"^[\w.]+(?:Error|Exception|Warning):", line):
            exc_start = i
    exception = "\n".join(lines[exc_start:])
    return frames, exception


async def main() -> None:
    client = AsyncDedalus(timeout=900.0)
    runner = DedalusRunner(client)
    namespace = f"error-traces-{uuid.uuid4().hex[:8]}"

    frames, exception = parse_trace(ERROR_TRACE)
    frames_text = "\n".join(
        f"ID: {f.id} | File: {f.file} | Line: {f.line} "
        f"| Function: {f.function} | Code: {f.code}"
        for f in frames
    )

    # Phase 1: Ingest frames into turbopuffer
    print(f"[Ingest] Storing {len(frames)} frames in namespace '{namespace}'...")
    ingest = await runner.run(
        input=(
            f'Store the following error trace frames in turbopuffer namespace "{namespace}" '
            f'using the turbopuffer_write tool. Use EXACTLY this v2 write format:\n\n'
            f'turbopuffer_write(\n'
            f'  namespace="{namespace}",\n'
            f'  write={{\n'
            f'    "schema": {{\n'
            f'      "content": {{"type": "string", "full_text_search": {{"tokenizer": "word_v3", "language": "english"}}}}\n'
            f'    }},\n'
            f'    "upsert_rows": [\n'
            f'      {{"id": "frame-0", "content": "File: /app/foo.py Line: 42 Function: bar Code: do_thing()", "file": "/app/foo.py", "line_num": 42, "function": "bar"}}\n'
            f'    ]\n'
            f'  }}\n'
            f')\n\n'
            f'Store ALL frames in a single turbopuffer_write call. '
            f'The "content" field should contain the full frame text for BM25 search. '
            f'Here are the frames:\n{frames_text}'
        ),
        model=MODEL,
        mcp_servers=[MCP_SERVER],
        credentials=[tp_secrets],
    )
    print(f"[Ingest] {ingest.final_output[:200]}")

    # Phase 2: Query relevant frames and diagnose
    print("\n[Diagnose] Querying for relevant frames...")
    diagnosis = await runner.run(
        input=(
            f'The following exception was raised:\n{exception}\n\n'
            f'Query the turbopuffer namespace "{namespace}" using the turbopuffer_query tool. '
            f'The tool takes two parameters: "namespace" (string) and "query" (dict). '
            f'Pass EXACTLY these arguments:\n'
            f'  namespace: "{namespace}"\n'
            f'  query: {{"rank_by": ["content", "BM25", "connect database payment server"], '
            f'"top_k": 5, "include_attributes": ["content", "file", "line_num", "function"]}}\n\n'
            f'IMPORTANT: The parameter name is "query", not "config". Pass the dict as the "query" parameter.\n\n'
            f'Based on the returned frames, provide:\n'
            f'1. Root cause analysis\n'
            f'2. The responsible service and file\n'
            f'3. A concrete fix recommendation'
        ),
        model=MODEL,
        mcp_servers=[MCP_SERVER],
        credentials=[tp_secrets],
    )
    print(f"\n[Diagnosis]\n{diagnosis.final_output}")

    print(f"\n[Done] Namespace '{namespace}' retained in turbopuffer for inspection.")


if __name__ == "__main__":
    asyncio.run(main())
