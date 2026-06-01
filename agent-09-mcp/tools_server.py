"""
MCP Server: exposes three tools via stdio transport.

This file runs as a standalone subprocess — do not run it directly.
The demo script (mcp_demo.py) launches it automatically.

Tools exposed:
  calculator   — evaluate a simple arithmetic expression
  text_stats   — count words, sentences, and chars in a text
  weather_mock — return mock weather data for a city
"""

import json
import math
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression (e.g. '2 ** 10', '100 / 7')."""
    allowed = set("0123456789 +-*/.()** ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are allowed"
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def text_stats(text: str) -> str:
    """Return word count, sentence count, and character count for the given text."""
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    chars = len(text)
    return json.dumps({"words": words, "sentences": sentences, "chars": chars})


@mcp.tool()
def weather_mock(city: str) -> str:
    """Return mock weather data for a city (demo only — not real data)."""
    mock_data = {
        "beijing":  {"temp": 25, "condition": "sunny",   "humidity": 40},
        "shanghai": {"temp": 22, "condition": "cloudy",  "humidity": 75},
        "shenzhen": {"temp": 30, "condition": "rainy",   "humidity": 90},
        "default":  {"temp": 20, "condition": "unknown", "humidity": 60},
    }
    data = mock_data.get(city.lower(), mock_data["default"])
    return json.dumps({"city": city, **data})


if __name__ == "__main__":
    mcp.run(transport="stdio")
