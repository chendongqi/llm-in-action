"""
MCP Protocol Demo — Minimal Server

Demonstrates the complete MCP protocol exchange:
  initialize handshake → capabilities negotiation
  tools/list          → tool discovery
  tools/call          → tool invocation
  resources/list      → resource discovery
  resources/read      → resource read
  prompts/list        → prompt template discovery
  prompts/get         → prompt template rendering

This server exposes:
  Tools:     echo (repeats input), add (adds two numbers)
  Resources: info://server-info (returns server metadata)
  Prompts:   summarize (a reusable prompt template)

Two ways to observe the protocol:

  Method A — Protocol client (terminal, no extra install):
    python demo_protocol_client.py
    Prints every JSON-RPC request/response pair in the terminal.

  Method B — MCP Inspector (visual browser UI):
    npx @modelcontextprotocol/inspector python demo_mcp_server.py
    Open the URL shown in terminal, click through the steps.

Run server standalone (stays alive, waiting for connections):
    conda activate llm_base
    python demo_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

# ── Server setup ──────────────────────────────────────────────────────────────

SERVER_NAME = "mcp-protocol-demo"
SERVER_VERSION = "1.0.0"

server = Server(SERVER_NAME)

# ── Tools ─────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Repeats the input message back. Useful for testing connectivity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="add",
            description="Adds two numbers and returns the result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "echo":
        message = arguments.get("message", "")
        return [TextContent(type="text", text=f"Echo: {message}")]

    if name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return [TextContent(type="text", text=f"{a} + {b} = {a + b}")]

    raise ValueError(f"Unknown tool: {name}")


# ── Resources ─────────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="info://server-info",  # type: ignore[arg-type]
            name="Server Information",
            description="Metadata about this MCP server: name, version, capabilities",
            mimeType="application/json"
        ),
    ]


@server.read_resource()  # type: ignore[arg-type]
async def read_resource(uri: str) -> str:
    if str(uri) == "info://server-info":
        info = {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "capabilities": {
                "tools": ["echo", "add"],
                "resources": ["info://server-info"],
                "prompts": ["summarize"]
            }
        }
        return json.dumps(info, indent=2)

    raise ValueError(f"Unknown resource: {uri}")


# ── Prompts ───────────────────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="summarize",
            description="Summarize a piece of text concisely",
            arguments=[
                PromptArgument(
                    name="text",
                    description="The text to summarize",
                    required=True
                ),
                PromptArgument(
                    name="max_words",
                    description="Maximum words in the summary (default: 50)",
                    required=False
                )
            ]
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    if name == "summarize":
        text = (arguments or {}).get("text", "")
        max_words = (arguments or {}).get("max_words", "50")
        return GetPromptResult(
            description="Summarization prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Summarize the following text in at most {max_words} words:\n\n{text}"
                    )
                )
            ]
        )

    raise ValueError(f"Unknown prompt: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
