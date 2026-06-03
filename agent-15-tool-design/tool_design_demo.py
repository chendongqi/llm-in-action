"""
Agent Tool Design Demo

Three design principles with live comparison:
  Demo 1 — Description Quality: vague vs precise docstrings affect LLM tool selection
  Demo 2 — Error Handling:      raise exceptions vs return error strings (agent recovery)
  Demo 3 — Tool Granularity:    fat omnibus tool vs focused fine-grained tools

Run:
    conda activate dev_base
    python tool_design_demo.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

MOCK_WEATHER = {
    "beijing":  {"temp": 25, "condition": "sunny"},
    "shanghai": {"temp": 22, "condition": "cloudy"},
    "shenzhen": {"temp": 30, "condition": "rainy"},
}

MOCK_PRODUCTS = {
    "wonderbot basic": {"price": 99,  "api_calls": 10_000},
    "wonderbot pro":   {"price": 299, "api_calls": 100_000},
}


def get_final_answer(output: dict) -> str:
    for m in reversed(output["messages"]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return str(m.content)
    return ""


def tools_called(output: dict) -> list[str]:
    result = []
    for m in output["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            result.extend(tc["name"] for tc in m.tool_calls)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Description Quality
# ══════════════════════════════════════════════════════════════════════════════
#
# Hypothesis: a precise, action-oriented docstring helps the LLM pick the right
# tool; a vague one causes the LLM to answer from its own knowledge instead.

print("\n" + "=" * 70)
print("Demo 1: Description Quality — vague vs precise docstrings")
print("=" * 70)


# ── Vague version ─────────────────────────────────────────────────────────────

@lc_tool
def weather_vague(city: str) -> str:
    """Get data."""
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return json.dumps({"city": city, **data})


# ── Precise version ───────────────────────────────────────────────────────────

@lc_tool
def weather_precise(city: str) -> str:
    """Get current weather for a city.

    Returns temperature (Celsius) and condition (sunny / cloudy / rainy / unknown).
    Use this whenever the user asks about weather, temperature, or sky conditions
    for a specific city. Pass the city name as a plain string, e.g. 'Beijing'.
    """
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return json.dumps({"city": city, **data})


agent_vague   = create_react_agent(model=llm, tools=[weather_vague])
agent_precise = create_react_agent(model=llm, tools=[weather_precise])

WEATHER_QUERIES = [
    "What's the weather in Beijing today?",
    "Is it raining in Shanghai right now?",
    "What temperature should I expect in Shenzhen?",
    "Should I bring an umbrella to Beijing?",
    "How's the sky in Shanghai?",
]

print(f"\nRunning {len(WEATHER_QUERIES)} weather queries on both agents:\n")
print(f"  {'Query':<48} {'Vague':<10} {'Precise'}")
print(f"  {'-'*48} {'-'*10} {'-'*10}")

vague_hits, precise_hits = 0, 0
for q in WEATHER_QUERIES:
    r_v = agent_vague.invoke({"messages":   [HumanMessage(q)]})
    r_p = agent_precise.invoke({"messages": [HumanMessage(q)]})
    called_v = bool(tools_called(r_v))
    called_p = bool(tools_called(r_p))
    vague_hits   += int(called_v)
    precise_hits += int(called_p)
    print(f"  {q:<48} {'✓ called' if called_v else '✗ skipped':<10} {'✓ called' if called_p else '✗ skipped'}")

print(f"\nTool call rate — Vague: {vague_hits}/{len(WEATHER_QUERIES)}  "
      f"Precise: {precise_hits}/{len(WEATHER_QUERIES)}")
print("Takeaway: a precise docstring is a signal to the LLM about WHEN and HOW to call the tool.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — Error Handling: raise vs return
# ══════════════════════════════════════════════════════════════════════════════
#
# Tools that raise exceptions crash the agent.
# Tools that return descriptive error strings let the LLM recover gracefully.

print("\n" + "=" * 70)
print("Demo 2: Error Handling — raise exception vs return error string")
print("=" * 70)


# ── Raises on unknown city ────────────────────────────────────────────────────

@lc_tool
def weather_raises(city: str) -> str:
    """Get current weather for a city."""
    if city.lower() not in MOCK_WEATHER:
        raise ValueError(f"City '{city}' not found in database.")
    data = MOCK_WEATHER[city.lower()]
    return json.dumps({"city": city, **data})


# ── Returns a helpful error string ────────────────────────────────────────────

@lc_tool
def weather_returns_error(city: str) -> str:
    """Get current weather for a city.

    Returns temperature and condition on success, or a descriptive error
    message if the city is not in the database.
    """
    data = MOCK_WEATHER.get(city.lower())
    if data is None:
        available = list(MOCK_WEATHER.keys())
        return (f"City '{city}' not found. "
                f"Available cities: {available}. "
                f"Please ask the user to confirm the city name.")
    return json.dumps({"city": city, **data})


ERROR_CASES = [
    ("known city",    "What's the weather in Beijing?"),
    ("unknown city",  "What's the weather in Atlantis?"),
    ("typo city",     "What's the weather in Shanghia?"),
]

agent_raises = create_react_agent(model=llm, tools=[weather_raises])
agent_returns = create_react_agent(model=llm, tools=[weather_returns_error])

print(f"\nRunning {len(ERROR_CASES)} cases comparing exception vs error-string:\n")

for label, query in ERROR_CASES:
    print(f"[{label}] {query}")

    # Agent that raises
    try:
        r_raise = agent_raises.invoke({"messages": [HumanMessage(query)]})
        ans_raise = get_final_answer(r_raise)
        crashed = False
    except Exception as e:
        ans_raise = f"[CRASHED] {type(e).__name__}: {e}"
        crashed = True

    # Agent that returns error
    r_return = agent_returns.invoke({"messages": [HumanMessage(query)]})
    ans_return = get_final_answer(r_return)

    print(f"  raises : {ans_raise[:120]}")
    print(f"  returns: {ans_return[:120]}")
    print(f"  outcome: {'CRASH' if crashed else 'handled'} / graceful")
    print()

print("Takeaway: return error strings. The LLM can reason about them and recover.")
print("Raised exceptions propagate up and crash the agent run.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — Tool Granularity: omnibus vs fine-grained
# ══════════════════════════════════════════════════════════════════════════════
#
# A fat tool that handles everything forces the LLM to pass free-text queries.
# Fine-grained tools accept structured parameters — easier for the LLM to use correctly.

print("\n" + "=" * 70)
print("Demo 3: Tool Granularity — omnibus tool vs fine-grained tools")
print("=" * 70)


# ── Fat omnibus tool ──────────────────────────────────────────────────────────

@lc_tool
def omnibus_lookup(query: str) -> str:
    """Look up weather, product info, or evaluate math. Pass the full user question."""
    q = query.lower()

    # Weather: try to extract city
    for city in MOCK_WEATHER:
        if city in q:
            data = MOCK_WEATHER[city]
            return json.dumps({"city": city, **data})

    # Product: try to extract product name
    for name in MOCK_PRODUCTS:
        if name in q or name.split()[-1] in q:
            data = MOCK_PRODUCTS[name]
            return json.dumps({"product": name, **data})

    # Math: try to eval
    import re
    match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', query)
    if match:
        expr = match.group().strip()
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return f"{expr} = {result}"
        except Exception:
            pass

    return f"Could not process query: '{query}'"


# ── Fine-grained tools ────────────────────────────────────────────────────────

@lc_tool
def get_weather(city: str) -> str:
    """Get current weather (temperature and condition) for a city by name."""
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return json.dumps({"city": city, **data})


@lc_tool
def get_product_info(product_name: str) -> str:
    """Get pricing and monthly API call limit for a WonderBot plan.

    Pass the full product name, e.g. 'wonderbot pro' or 'wonderbot basic'.
    """
    data = MOCK_PRODUCTS.get(product_name.lower())
    if data is None:
        return f"Product '{product_name}' not found. Available: {list(MOCK_PRODUCTS)}"
    return json.dumps({"product": product_name, **data})


@lc_tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression.

    Pass a plain math expression, e.g. '299 * 12' or '25 - 22'.
    Only digits and operators (+ - * / . ()) are allowed.
    """
    import math
    allowed = set("0123456789 +-*/.()** ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric operators allowed"
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


agent_fat   = create_react_agent(model=llm, tools=[omnibus_lookup])
agent_fine  = create_react_agent(model=llm, tools=[get_weather, get_product_info, calculator])

GRANULARITY_CASES = [
    (
        "single — weather",
        "What's the weather in Beijing?",
    ),
    (
        "single — product",
        "How much does WonderBot Pro cost?",
    ),
    (
        "multi-step — weather + calc",
        "What is the temperature in Beijing and Shanghai? Calculate the difference.",
    ),
    (
        "multi-step — product + calc",
        "What is the WonderBot Pro monthly price? Calculate 299 * 12 for annual cost.",
    ),
]

print(f"\nRunning {len(GRANULARITY_CASES)} cases on fat vs fine-grained:\n")

for label, query in GRANULARITY_CASES:
    print(f"[{label}]")
    print(f"  Query: {query}")

    r_fat  = agent_fat.invoke({"messages":  [HumanMessage(query)]})
    r_fine = agent_fine.invoke({"messages": [HumanMessage(query)]})

    called_fat  = tools_called(r_fat)
    called_fine = tools_called(r_fine)
    ans_fat  = get_final_answer(r_fat)[:100]
    ans_fine = get_final_answer(r_fine)[:100]

    print(f"  Fat    tools={called_fat}  →  {ans_fat}")
    print(f"  Fine   tools={called_fine}  →  {ans_fine}")
    print()

print("Takeaway: fine-grained tools with typed parameters are easier for the LLM")
print("to parameterize correctly. Fat tools push parsing work onto the tool itself,")
print("and the LLM often passes ambiguous or over-complete queries.")


# ── Design principle summary ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Tool Design Principles Summary")
print("=" * 70)
print()
print(f"{'Principle':<22} {'Bad':<32} {'Good'}")
print("-" * 75)
print(f"{'Description':<22} {'\"Get data.\"':<32} {'What, when, how + param format'}")
print(f"{'Error handling':<22} {'raise ValueError(...)':<32} {'return \"Error: ...\" string'}")
print(f"{'Granularity':<22} {'omnibus(query: str)':<32} {'separate typed-param tools'}")
print(f"{'Parameter naming':<22} {'lookup(q: str)':<32} {'get_weather(city: str)'}")
print(f"{'Return format':<22} {'raw dict / None':<32} {'JSON string or error string'}")
print()
print("Golden rule: design tools for the LLM, not for humans.")
print("The LLM reads the docstring to decide whether and how to call the tool.")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
