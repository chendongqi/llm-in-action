"""
MCP Protocol Demo

Three demos:
  Demo 1 — Traditional Function Calling: tools defined inline, hardwired to one agent
  Demo 2 — MCP Server Discovery: client connects, lists tools, calls them directly
  Demo 3 — LLM + MCP: LangChain agent uses MCP tools via MultiServerMCPClient

Run:
    conda activate dev_base
    python mcp_demo.py
"""

import asyncio
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ── LLM init ──────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

SERVER_SCRIPT = str(Path(__file__).parent / "tools_server.py")
PYTHON_BIN = sys.executable


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Traditional Function Calling
# ══════════════════════════════════════════════════════════════════════════════
#
# Tools are defined as Python functions, then wrapped into LangChain Tool
# objects and bound to the LLM. Every agent that needs these tools must
# repeat this wiring.

print("\n" + "=" * 70)
print("Demo 1: Traditional Function Calling")
print("=" * 70)

import json
import math
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import create_react_agent


@lc_tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789 +-*/.()** ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are allowed"
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@lc_tool
def text_stats(text: str) -> str:
    """Return word count, sentence count, and character count."""
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    chars = len(text)
    return json.dumps({"words": words, "sentences": sentences, "chars": chars})


@lc_tool
def weather_mock(city: str) -> str:
    """Return mock weather data for a city."""
    mock_data = {
        "beijing":  {"temp": 25, "condition": "sunny",   "humidity": 40},
        "shanghai": {"temp": 22, "condition": "cloudy",  "humidity": 75},
        "shenzhen": {"temp": 30, "condition": "rainy",   "humidity": 90},
        "default":  {"temp": 20, "condition": "unknown", "humidity": 60},
    }
    data = mock_data.get(city.lower(), mock_data["default"])
    return json.dumps({"city": city, **data})


traditional_tools = [calculator, text_stats, weather_mock]
traditional_agent = create_react_agent(model=llm, tools=traditional_tools)

questions = [
    "What is 2 ** 10 + 100 / 4?",
    "Analyze this text: 'Python is elegant. It is readable. Everyone loves it!'",
    "What's the weather in Beijing?",
]

print("\nAgent with inline tool definitions (traditional approach):")
print("-" * 50)
for q in questions:
    result = traditional_agent.invoke({"messages": [HumanMessage(q)]})
    answer = result["messages"][-1].content
    print(f"Q: {q}")
    print(f"A: {answer[:120]}")
    print()

print(f"Tools defined inline in THIS file: {[t.name for t in traditional_tools]}")
print("Problem: if Agent B also needs these tools, you copy-paste or re-import.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — MCP Server Discovery
# ══════════════════════════════════════════════════════════════════════════════
#
# MCP server runs as a subprocess. The client connects via stdio, calls
# initialize(), then list_tools() to discover what's available — no
# hard-coded tool list on the client side.

print("\n" + "=" * 70)
print("Demo 2: MCP Server Tool Discovery")
print("=" * 70)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo_mcp_discovery() -> None:
    server_params = StdioServerParameters(
        command=PYTHON_BIN,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── tool discovery ────────────────────────────────────────────────
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"\nServer: demo-tools")
            print(f"Discovered {len(tools)} tools:")
            for t in tools:
                print(f"  ● {t.name:<16} — {t.description}")

            # ── direct tool calls ─────────────────────────────────────────────
            print("\nDirect tool calls (no LLM):")

            r1 = await session.call_tool("calculator", {"expression": "2 ** 10 + 100 / 4"})
            print(f"  calculator('2 ** 10 + 100 / 4')  → {r1.content[0].text}")  # type: ignore[index]

            r2 = await session.call_tool("weather_mock", {"city": "Shanghai"})
            print(f"  weather_mock('Shanghai')          → {r2.content[0].text}")  # type: ignore[index]

            r3 = await session.call_tool(
                "text_stats",
                {"text": "Python is elegant. It is readable. Everyone loves it!"}
            )
            print(f"  text_stats(...)                   → {r3.content[0].text}")  # type: ignore[index]

    print("\nKey insight: client discovered tools dynamically — no hardcoded list.")
    print("Any agent connecting to this server gets the same tool catalog.")


asyncio.run(demo_mcp_discovery())


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — LLM + MCP via MultiServerMCPClient
# ══════════════════════════════════════════════════════════════════════════════
#
# langchain-mcp-adapters converts MCP tool schemas into LangChain Tool objects
# automatically. The LLM agent uses them exactly like inline tools — but they
# live in the MCP server, not in the agent code.

print("\n" + "=" * 70)
print("Demo 3: LLM Agent Using MCP Tools")
print("=" * 70)

from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore[import-untyped]


async def demo_llm_mcp() -> None:
    # langchain-mcp-adapters >= 0.1.0: MultiServerMCPClient is not a context manager
    client = MultiServerMCPClient(
        {
            "demo-tools": {
                "command": PYTHON_BIN,
                "args": [SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await client.get_tools()
    print(f"\nMCP tools loaded into LangChain: {[t.name for t in mcp_tools]}")
    print("Agent sees the same tools — defined in server, not in agent code.\n")

    agent = create_react_agent(model=llm, tools=mcp_tools)

    multi_q = (
        "Please answer all three questions:\n"
        "1. What is sqrt(144) + 2 ** 8?\n"
        "2. What's the weather in Shenzhen?\n"
        "3. Count the stats for: 'MCP is a protocol. It standardizes tools. "
        "Agents love it!'"
    )

    print(f"Question: {multi_q}\n")
    result = await agent.ainvoke({"messages": [HumanMessage(multi_q)]})
    answer = result["messages"][-1].content
    print(f"Answer:\n{answer}")

    print("\n" + "-" * 50)
    print("Comparison summary:")
    print()
    rows = [
        ("Tool definition",   "In agent code (repeated)",  "In MCP server (once)"),
        ("Tool discovery",    "Hardcoded import",          "Dynamic list_tools()"),
        ("Multi-agent reuse", "Copy-paste or re-import",   "All agents connect to same server"),
        ("Update a tool",     "Edit every agent",          "Edit server only"),
        ("Cross-language",    "Python only",               "Any language (JSON-RPC)"),
    ]
    print(f"{'Dimension':<22} {'Traditional':<30} {'MCP'}")
    print("-" * 74)
    for row in rows:
        print(f"{row[0]:<22} {row[1]:<30} {row[2]}")


asyncio.run(demo_llm_mcp())

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
