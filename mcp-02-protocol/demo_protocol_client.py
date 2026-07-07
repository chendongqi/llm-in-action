"""
MCP Protocol Client — Shows Raw JSON-RPC Exchange

Connects to demo_mcp_server.py via stdio subprocess and
prints every JSON-RPC request/response pair in the terminal.

This lets you see the exact protocol messages without a browser:
  → initialize request/response (capability negotiation)
  → tools/list request/response (tool discovery)
  → tools/call request/response (tool execution)
  → resources/list and resources/read
  → prompts/list and prompts/get

Run:
    conda activate dev_base
    python demo_protocol_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

WIDTH = 70


def fmt_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def print_exchange(direction: str, msg: dict) -> None:
    icon = "→" if direction == "request" else "←"
    method = msg.get("method") or f"result id={msg.get('id')}"
    print(f"\n{icon} {method}")
    print("─" * WIDTH)
    print(fmt_json(msg))


async def run() -> None:
    server_path = Path(__file__).parent / "demo_mcp_server.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(server_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin and proc.stdout

    async def send(msg: dict) -> dict:
        """Send a JSON-RPC message and read the response."""
        raw = json.dumps(msg) + "\n"
        print_exchange("request", msg)
        proc.stdin.write(raw.encode())  # type: ignore[union-attr]
        await proc.stdin.drain()  # type: ignore[union-attr]
        response_line = await proc.stdout.readline()  # type: ignore[union-attr]
        response = json.loads(response_line)
        print_exchange("response", response)
        return response

    print("=" * WIDTH)
    print("MCP Protocol Demo — Raw JSON-RPC Exchange")
    print("=" * WIDTH)

    # ── Step 1: initialize ────────────────────────────────────────────────────
    print("\n\n[Step 1] Initialize — capability negotiation")
    await send({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "demo-client",
                "version": "1.0.0"
            }
        }
    })

    # Must send initialized notification after initialize
    proc.stdin.write((json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }) + "\n").encode())
    await proc.stdin.drain()

    # ── Step 2: tools/list ────────────────────────────────────────────────────
    print("\n\n[Step 2] tools/list — discover available tools")
    tools_response = await send({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })
    tool_names = [t["name"] for t in tools_response.get("result", {}).get("tools", [])]
    print(f"\n  Found tools: {tool_names}")

    # ── Step 3: tools/call — echo ─────────────────────────────────────────────
    print("\n\n[Step 3] tools/call — invoke 'echo' tool")
    await send({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {"message": "Hello from MCP client!"}
        }
    })

    # ── Step 4: tools/call — add ──────────────────────────────────────────────
    print("\n\n[Step 4] tools/call — invoke 'add' tool")
    await send({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "add",
            "arguments": {"a": 42, "b": 58}
        }
    })

    # ── Step 5: resources/list ────────────────────────────────────────────────
    print("\n\n[Step 5] resources/list — discover available resources")
    await send({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "resources/list",
        "params": {}
    })

    # ── Step 6: resources/read ────────────────────────────────────────────────
    print("\n\n[Step 6] resources/read — read server-info resource")
    await send({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/read",
        "params": {"uri": "info://server-info"}
    })

    # ── Step 7: prompts/list ──────────────────────────────────────────────────
    print("\n\n[Step 7] prompts/list — discover prompt templates")
    await send({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "prompts/list",
        "params": {}
    })

    # ── Step 8: prompts/get ───────────────────────────────────────────────────
    print("\n\n[Step 8] prompts/get — render 'summarize' template")
    await send({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/get",
        "params": {
            "name": "summarize",
            "arguments": {
                "text": "MCP defines a standard protocol for connecting AI models to tools and data sources.",
                "max_words": "20"
            }
        }
    })

    print(f"\n\n{'=' * WIDTH}")
    print("Protocol exchange complete — 8 request/response pairs shown above")
    print(f"{'=' * WIDTH}\n")

    proc.terminate()
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(run())
