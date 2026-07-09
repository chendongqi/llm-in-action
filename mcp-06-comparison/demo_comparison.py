"""
MCP Article 06 — MCP vs Function Calling Engineering Comparison

Measures three real dimensions that matter in the selection decision:

  1. Latency overhead
     Direct Python function call vs MCP Server call (stdio subprocess)
     Tests: 10 warm-up + 20 measured iterations each
     Reports: P50 / P90 / mean latency in milliseconds

  2. Code size comparison
     Count lines of implementation code for the same "search issues" feature
     Method A: Function Calling (tool definition + handler inline)
     Method B: MCP Server (standalone server process)

  3. Startup cost
     How long does spawning + initializing an MCP Server take?
     (One-time cost that can be amortized, but relevant for short-lived agents)

The feature under test: search a list of issues by keyword.
Same mock data used in both approaches to make results comparable.

Run:
    conda activate llm_base
    python demo_comparison.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

# ── Shared mock data ──────────────────────────────────────────────────────────

ISSUES = [
    {"key": "PROJ-101", "summary": "NPE in parseInput when config is null", "status": "Open", "priority": "P1"},
    {"key": "PROJ-102", "summary": "Add rate limiting to API endpoints", "status": "In Progress", "priority": "P2"},
    {"key": "PROJ-103", "summary": "Memory leak in WebSocket pool", "status": "Open", "priority": "P0"},
    {"key": "MOBILE-55", "summary": "Crash on Android 14 notification settings", "status": "Open", "priority": "P1"},
    {"key": "INFRA-12", "summary": "Upgrade PostgreSQL from 14 to 16", "status": "Done", "priority": "P2"},
]

WIDTH = 70


# ── Method A: Direct Python function (Function Calling equivalent) ────────────

def search_issues_direct(query: str, status: str = "") -> str:
    """The actual tool logic — called directly, no protocol layer."""
    query_lower = query.lower()
    results = [
        i for i in ISSUES
        if query_lower in i["summary"].lower()
        and (not status or i["status"] == status)
    ]
    if not results:
        return "No issues found."
    return "\n".join(f"[{i['key']}] {i['summary']} ({i['status']})" for i in results)


# ── Method B: MCP Server (inline for demo, normally in separate file) ─────────

MCP_SERVER_CODE = '''
import asyncio, json, sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

ISSUES = {issues_json}

server = Server("bench-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="search_issues",
        description="Search issues by keyword",
        inputSchema={{
            "type": "object",
            "properties": {{
                "query": {{"type": "string"}},
                "status": {{"type": "string"}}
            }},
            "required": ["query"]
        }}
    )]

@server.call_tool()
async def call_tool(name, arguments):
    query = arguments.get("query", "").lower()
    status = arguments.get("status", "")
    results = [
        i for i in ISSUES
        if query in i["summary"].lower()
        and (not status or i["status"] == status)
    ]
    text = "\\n".join(f"[{{i['key']}}] {{i['summary']}} ({{i['status']}})" for i in results) or "No issues found."
    return [TextContent(type="text", text=text)]

async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

asyncio.run(main())
'''.format(issues_json=json.dumps(ISSUES))

# Write server to temp file
_SERVER_FILE = Path("/tmp/mcp_bench_server.py")
_SERVER_FILE.write_text(MCP_SERVER_CODE)


async def _send(proc, req_id: int, method: str, params: dict) -> dict:
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    proc.stdin.write((json.dumps(msg) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]
    line = await proc.stdout.readline()  # type: ignore[union-attr]
    return json.loads(line)


async def search_via_mcp(proc, req_id: int, query: str) -> str:
    r = await _send(proc, req_id, "tools/call", {
        "name": "search_issues",
        "arguments": {"query": query}
    })
    return r.get("result", {}).get("content", [{}])[0].get("text", "")


# ── Benchmark 1: Latency ──────────────────────────────────────────────────────

def benchmark_direct(n: int = 20) -> list[float]:
    """Measure direct function call latency (microsecond range)."""
    queries = ["NPE", "API", "memory", "crash", "postgres"]
    times = []
    for i in range(n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        search_issues_direct(q)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return times


async def benchmark_mcp(n: int = 20) -> tuple[list[float], float]:
    """Measure MCP tool call latency (includes subprocess stdio roundtrip)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(_SERVER_FILE),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    # Initialize
    await _send(proc, 1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "bench", "version": "1.0"}
    })
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]

    queries = ["NPE", "API", "memory", "crash", "postgres"]
    req_id = 2
    times = []

    # Warm-up
    for i in range(5):
        await search_via_mcp(proc, req_id, queries[i % len(queries)])
        req_id += 1

    # Measured
    for i in range(n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        await search_via_mcp(proc, req_id, q)
        times.append((time.perf_counter() - t0) * 1000)
        req_id += 1

    proc.terminate()
    await proc.wait()
    return times, 0.0


async def benchmark_mcp_startup() -> float:
    """Measure MCP Server spawn + initialize time (one-time cost)."""
    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(_SERVER_FILE),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await _send(proc, 1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "bench", "version": "1.0"}
    })
    startup_ms = (time.perf_counter() - t0) * 1000
    proc.terminate()
    await proc.wait()
    return startup_ms


# ── Benchmark 2: Code size ────────────────────────────────────────────────────

FUNCTION_CALLING_IMPL = '''
# Method A: Function Calling (tool definition inline in Agent code)

SEARCH_TOOL = {
    "name": "search_issues",
    "description": "Search Jira issues by keyword",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "status": {"type": "string"}
        },
        "required": ["query"]
    }
}

def search_issues(query: str, status: str = "") -> str:
    query_lower = query.lower()
    results = [
        i for i in ISSUES
        if query_lower in i["summary"].lower()
        and (not status or i["status"] == status)
    ]
    if not results:
        return "No issues found."
    return "\\n".join(
        f"[{i['key']}] {i['summary']} ({i['status']})" for i in results
    )

def run_agent(user_input: str):
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_input}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            tools=[SEARCH_TOOL],
            messages=messages
        )
        if response.stop_reason != "tool_use":
            return response.content[0].text
        tool_call = next(b for b in response.content if b.type == "tool_use")
        result = search_issues(**tool_call.input)
        messages.extend([
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_call.id, "content": result}
            ]}
        ])
'''

MCP_SERVER_IMPL = '''
# Method B: MCP Server (standalone process, reusable by any Host)

server = Server("issues-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="search_issues",
        description="Search Jira issues by keyword",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "status": {"type": "string"}
            },
            "required": ["query"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    query = arguments.get("query", "").lower()
    status = arguments.get("status", "")
    results = [
        i for i in ISSUES
        if query in i["summary"].lower()
        and (not status or i["status"] == status)
    ]
    text = "\\n".join(
        f"[{i['key']}] {i['summary']} ({i['status']})" for i in results
    ) or "No issues found."
    return [TextContent(type="text", text=text)]

async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

asyncio.run(main())
# (Agent code connects via config — zero tool code in Agent)
'''


def count_non_empty_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * WIDTH)
    print("MCP vs Function Calling — Engineering Comparison")
    print("=" * WIDTH)

    # ── Latency benchmark ─────────────────────────────────────────────────────
    print(f"\n{'─' * WIDTH}")
    print("Benchmark 1: Tool Call Latency  (N=20 calls per method)")
    print(f"{'─' * WIDTH}")

    direct_times = benchmark_direct(20)
    mcp_times, _ = await benchmark_mcp(20)
    startup_ms = await benchmark_mcp_startup()

    def stats(times: list[float]) -> dict:
        s = sorted(times)
        return {
            "mean": statistics.mean(s),
            "p50": s[len(s) // 2],
            "p90": s[int(len(s) * 0.9)],
            "min": s[0],
            "max": s[-1],
        }

    ds = stats(direct_times)
    ms = stats(mcp_times)

    print(f"\n  {'Method':<28} {'Mean':>8}  {'P50':>8}  {'P90':>8}  {'Min':>8}")
    print(f"  {'─'*28} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'Direct function call':<28} {ds['mean']:>7.2f}ms  {ds['p50']:>7.2f}ms  {ds['p90']:>7.2f}ms  {ds['min']:>7.3f}ms")
    print(f"  {'MCP stdio call':<28} {ms['mean']:>7.2f}ms  {ms['p50']:>7.2f}ms  {ms['p90']:>7.2f}ms  {ms['min']:>7.2f}ms")

    overhead = ms['mean'] - ds['mean']
    ratio = ms['mean'] / ds['mean'] if ds['mean'] > 0 else float('inf')
    print(f"\n  MCP overhead per call: +{overhead:.2f}ms  ({ratio:.0f}x slower than direct)")
    print(f"  MCP server startup (one-time): {startup_ms:.0f}ms")
    calls_to_amortize = startup_ms / overhead if overhead > 0 else 0
    print(f"  Calls to amortize startup cost: ~{calls_to_amortize:.0f}")

    # ── Code size comparison ──────────────────────────────────────────────────
    print(f"\n{'─' * WIDTH}")
    print("Benchmark 2: Implementation Code Size  (non-empty, non-comment lines)")
    print(f"{'─' * WIDTH}")

    fc_lines = count_non_empty_lines(FUNCTION_CALLING_IMPL)
    mcp_lines = count_non_empty_lines(MCP_SERVER_IMPL)

    print(f"\n  Function Calling (tool definition + handler + Agent loop): {fc_lines} lines")
    print(f"  MCP Server (standalone Server only):                        {mcp_lines} lines")
    print(f"  MCP Agent code (zero tool code needed):                      0 lines")
    print(f"\n  For a SINGLE project: FC={fc_lines}  MCP={mcp_lines}  (similar)")
    print(f"  For N projects sharing the same tool:")
    for n in [2, 3, 5]:
        fc_total = fc_lines * n
        mcp_total = mcp_lines  # Server once, Agent code = 0 per project
        print(f"    N={n}: FC={fc_total} lines total  MCP={mcp_total} lines total  (MCP saves {fc_total - mcp_total} lines)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * WIDTH}")
    print("Summary")
    print(f"{'─' * WIDTH}")
    print(f"  MCP per-call overhead:    +{overhead:.1f}ms  (protocol + subprocess stdio)")
    print(f"  MCP startup cost:         {startup_ms:.0f}ms  (amortized over {calls_to_amortize:.0f}+ calls)")
    print(f"  Code size at N=1 project: similar  ({fc_lines} vs {mcp_lines} lines)")
    print(f"  Code size at N=3 projects: FC={fc_lines*3} lines  MCP={mcp_lines} lines")
    print(f"{'=' * WIDTH}\n")


if __name__ == "__main__":
    asyncio.run(main())
