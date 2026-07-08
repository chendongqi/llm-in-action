"""
Test client for demo_jira_server.py

Exercises all tools with both valid and invalid inputs,
so you can see both success and error responses.

Run:
    conda activate llm_base
    python demo_test_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

WIDTH = 70


def print_call(label: str, result: dict) -> None:
    content = result.get("result", {}).get("content", [])
    is_error = any(c.get("isError") for c in content)
    text = content[0].get("text", "") if content else ""
    icon = "✗" if is_error else "✓"
    print(f"\n  {icon} {label}")
    print(f"    {text[:120].replace(chr(10), ' ')}")


async def send(proc, req_id: int, method: str, params: dict) -> dict:
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    proc.stdin.write((json.dumps(msg) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]
    line = await proc.stdout.readline()  # type: ignore[union-attr]
    return json.loads(line)


async def run() -> None:
    server_path = Path(__file__).parent / "demo_jira_server.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(server_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    req_id = 0

    def next_id() -> int:
        nonlocal req_id
        req_id += 1
        return req_id

    # Initialize
    await send(proc, next_id(), "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"}
    })
    proc.stdin.write((json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"}) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]

    print("=" * WIDTH)
    print("Jira MCP Server — Test Suite")
    print("=" * WIDTH)

    # ── search_issues ─────────────────────────────────────────────────────────
    print("\n[search_issues]")

    r = await send(proc, next_id(), "tools/call", {
        "name": "search_issues", "arguments": {"query": "NPE", "project": "PROJ"}
    })
    print_call("search 'NPE' in PROJ", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "search_issues", "arguments": {"query": "crash", "status": "Open"}
    })
    print_call("search 'crash' with status=Open", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "search_issues", "arguments": {"query": "anything", "project": "INVALID"}
    })
    print_call("search with invalid project (validation error)", r)

    # ── get_issue ─────────────────────────────────────────────────────────────
    print("\n[get_issue]")

    r = await send(proc, next_id(), "tools/call", {
        "name": "get_issue", "arguments": {"issue_key": "PROJ-101"}
    })
    print_call("get PROJ-101 (P1 Bug)", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "get_issue", "arguments": {"issue_key": "PROJ-999"}
    })
    print_call("get PROJ-999 (not found → isError=true)", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "get_issue", "arguments": {"issue_key": "not-a-key"}
    })
    print_call("get malformed key (validation error → isError=true)", r)

    # ── create_issue ──────────────────────────────────────────────────────────
    print("\n[create_issue]")

    r = await send(proc, next_id(), "tools/call", {
        "name": "create_issue",
        "arguments": {
            "project": "PROJ",
            "summary": "Redis connection pool not releasing connections on timeout",
            "description": "After 30 minutes of inactivity, connections remain open and eventually exhaust the pool.",
            "priority": "P1",
            "issue_type": "Bug"
        }
    })
    print_call("create valid Bug in PROJ", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "create_issue",
        "arguments": {"project": "PROJ", "summary": ""}
    })
    print_call("create with empty summary (isError=true)", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "create_issue",
        "arguments": {"project": "PROJ", "summary": "Test", "priority": "CRITICAL"}
    })
    print_call("create with invalid priority (isError=true)", r)

    # ── update_issue ──────────────────────────────────────────────────────────
    print("\n[update_issue]")

    r = await send(proc, next_id(), "tools/call", {
        "name": "update_issue",
        "arguments": {"issue_key": "PROJ-101", "status": "In Progress", "assignee": "bob"}
    })
    print_call("update PROJ-101 status + assignee", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "update_issue",
        "arguments": {"issue_key": "PROJ-101"}
    })
    print_call("update with no fields (isError=true)", r)

    r = await send(proc, next_id(), "tools/call", {
        "name": "update_issue",
        "arguments": {"issue_key": "PROJ-101", "status": "Shipped"}
    })
    print_call("update with invalid status (isError=true)", r)

    # ── resources ─────────────────────────────────────────────────────────────
    print("\n[resources]")

    r = await send(proc, next_id(), "resources/list", {})
    resources = r.get("result", {}).get("resources", [])
    print(f"\n  Available resources: {[res['uri'] for res in resources]}")

    r = await send(proc, next_id(), "resources/read", {"uri": "jira://projects"})
    contents = r.get("result", {}).get("contents", [])
    if contents:
        projects = json.loads(contents[0]["text"])
        print(f"  Projects: {[p['key'] for p in projects]}")

    print(f"\n{'=' * WIDTH}")
    print("Test suite complete")
    print(f"{'=' * WIDTH}\n")

    proc.terminate()
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(run())
