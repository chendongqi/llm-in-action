"""
Agent Cost & Performance Optimization Demo

Four optimization strategies with measured impact:
  Demo 1 — Token Cost Breakdown:  measure where tokens go; minimal vs verbose system prompt
  Demo 2 — Model Routing:         direct LLM for simple queries, full agent for complex ones
  Demo 3 — Parallel Tool Calls:   asyncio vs sequential; wall-clock time comparison
  Demo 4 — Tool Result Cache:     TTL-based memoization; avoid redundant tool + LLM calls

Run:
    conda activate dev_base
    python cost_demo.py
"""

import asyncio
import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

enc = tiktoken.get_encoding("cl100k_base")

MOCK_WEATHER = {
    "beijing":  {"temp": 25, "condition": "sunny"},
    "shanghai": {"temp": 22, "condition": "cloudy"},
    "shenzhen": {"temp": 30, "condition": "rainy"},
}
MOCK_PRODUCTS = {
    "wonderbot basic": {"price": 99,  "api_calls": 10_000},
    "wonderbot pro":   {"price": 299, "api_calls": 100_000},
}


def count_tokens(text: str) -> int:
    return len(enc.encode(str(text)))


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


# ── Tools ──────────────────────────────────────────────────────────────────────

@lc_tool
def get_weather(city: str) -> str:
    """Get current weather (temperature and condition) for a city.

    Use when the user asks about weather or temperature for a specific city.
    Pass the city name as a plain string, e.g. 'Beijing'.
    """
    time.sleep(0.05)  # simulate network latency
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return f"{city}: {data['temp']}°C, {data['condition']}"


@lc_tool
def get_product_info(product_name: str) -> str:
    """Get pricing and monthly API call limit for a WonderBot plan.

    Use when the user asks about product pricing or plans.
    Pass the full product name, e.g. 'wonderbot pro'.
    """
    time.sleep(0.05)  # simulate network latency
    data = MOCK_PRODUCTS.get(product_name.lower())
    if data is None:
        return f"Product '{product_name}' not found. Available: {list(MOCK_PRODUCTS)}"
    return f"{product_name}: ${data['price']}/month, {data['api_calls']:,} API calls/month"


@lc_tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression.

    Pass a plain math expression, e.g. '299 * 12' or '25 - 22'.
    """
    allowed = set("0123456789 +-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric operators allowed"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [get_weather, get_product_info, calculator]


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1: Token Cost Breakdown — minimal vs verbose system prompt
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 1: Token Cost Breakdown — system prompt size matters")
print("=" * 70)

MINIMAL_PROMPT = "You are a helpful assistant."

VERBOSE_PROMPT = """\
You are an extremely helpful, knowledgeable, and professional AI assistant \
for WonderLab's enterprise software platform. You specialize in providing \
accurate weather information for cities worldwide, detailed product \
recommendations for WonderBot plans, and precise arithmetic calculations. \
You always respond in a clear, structured manner, provide context for your \
answers, explain your reasoning step by step, and make sure the user fully \
understands the information provided. When the user's request is ambiguous, \
ask clarifying questions before proceeding. Always be thorough, \
comprehensive, and leave no important detail unexplained.\
"""

prompt_tokens_min  = count_tokens(MINIMAL_PROMPT)
prompt_tokens_verb = count_tokens(VERBOSE_PROMPT)

print(f"\nSystem prompt token counts:")
print(f"  Minimal  ({prompt_tokens_min:>3} tokens): '{MINIMAL_PROMPT}'")
print(f"  Verbose  ({prompt_tokens_verb:>3} tokens): '{VERBOSE_PROMPT[:60]}...'")
print(f"  Extra tokens per call: {prompt_tokens_verb - prompt_tokens_min}")

# Run the same query on both agents and measure latency
agent_min  = create_react_agent(model=llm, tools=TOOLS, prompt=MINIMAL_PROMPT)
agent_verb = create_react_agent(model=llm, tools=TOOLS, prompt=VERBOSE_PROMPT)

QUERY = "What's the weather in Beijing?"

print(f"\nQuery: '{QUERY}'  (2 runs each)")
print(f"\n  {'Agent':<12} {'Run 1':>8} {'Run 2':>8} {'Avg':>8}  Answer")
print(f"  {'-'*65}")

for label, agent in [("Minimal", agent_min), ("Verbose", agent_verb)]:
    times = []
    last_answer = ""
    for _ in range(2):
        t0 = time.time()
        r = agent.invoke({"messages": [HumanMessage(QUERY)]})
        times.append(time.time() - t0)
        last_answer = get_final_answer(r)
    avg = sum(times) / len(times)
    print(f"  {label:<12} {times[0]:>7.2f}s {times[1]:>7.2f}s {avg:>7.2f}s  {last_answer[:40]}")

print(f"\nTakeaway: every call pays the system-prompt token cost.")
print(f"  - Keep system prompts concise; move static reference docs to retrieval")
print(f"  - Claude / OpenAI support explicit prompt caching (cache_control) to")
print(f"    amortize repeated static context — up to ~90% cost reduction on")
print(f"    the cached portion when the same prefix is reused across calls")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2: Model Routing — direct LLM vs full agent
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 2: Model Routing — skip agent overhead for simple queries")
print("=" * 70)

ROUTING_SYSTEM = """\
Classify the user query. Reply with ONLY one word:
- "direct"  if the query can be answered from general knowledge (no real-time data needed)
- "agent"   if the query requires a tool call (weather, product pricing, calculation)

Examples:
  "What is the capital of France?"    → direct
  "Explain machine learning briefly." → direct
  "What's the weather in Shanghai?"   → agent
  "How much does WonderBot Pro cost?" → agent
  "What is 299 * 12?"                 → agent\
"""


def classify_query(query: str) -> str:
    resp = llm.invoke([
        SystemMessage(ROUTING_SYSTEM),
        HumanMessage(query),
    ])
    raw = str(resp.content).strip().lower()
    return "agent" if "agent" in raw else "direct"


full_agent = create_react_agent(model=llm, tools=TOOLS)


def routed_run(query: str) -> dict:
    t_route_start = time.time()
    route = classify_query(query)
    t_route = time.time() - t_route_start

    t_exec_start = time.time()
    if route == "direct":
        resp = llm.invoke([HumanMessage(query)])
        answer = str(resp.content)
        tool_names: list[str] = []
    else:
        r = full_agent.invoke({"messages": [HumanMessage(query)]})
        answer = get_final_answer(r)
        tool_names = tools_called(r)
    t_exec = time.time() - t_exec_start

    return {
        "route":      route,
        "route_ms":   t_route * 1000,
        "exec_ms":    t_exec * 1000,
        "total_ms":   (t_route + t_exec) * 1000,
        "tools":      tool_names,
        "answer":     answer[:80],
    }


ROUTING_CASES = [
    ("What is the capital of France?",                    "→ direct expected"),
    ("Explain machine learning in one sentence.",          "→ direct expected"),
    ("What's the weather in Shanghai right now?",          "→ agent  expected"),
    ("How much does WonderBot Pro cost per month?",        "→ agent  expected"),
    ("What is 299 multiplied by 12?",                      "→ agent  expected"),
]

print()
print(f"  {'Query':<50} {'Route':<8} {'Total':>8}  Tools")
print(f"  {'-'*80}")

for query, note in ROUTING_CASES:
    r = routed_run(query)
    tool_str = str(r["tools"]) if r["tools"] else "[]"
    print(f"  {query:<50} {r['route']:<8} {r['total_ms']:>6.0f}ms  {tool_str}")

print(f"\nTakeaway: routing cuts agent overhead (~2-4 LLM turns) to 1 call")
print(f"for queries that don't need tools. At scale, this matters.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3: Parallel Tool Calls — asyncio vs sequential
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 3: Parallel Tool Calls — sequential vs async wall-clock time")
print("=" * 70)

LATENCY_MS = 100  # simulated per-tool I/O latency


async def fetch_weather_async(city: str) -> str:
    await asyncio.sleep(LATENCY_MS / 1000)
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return f"{city}: {data['temp']}°C, {data['condition']}"


async def run_parallel(cities: list[str]) -> list[str]:
    return await asyncio.gather(*[fetch_weather_async(c) for c in cities])


def run_sequential(cities: list[str]) -> list[str]:
    results = []
    for city in cities:
        time.sleep(LATENCY_MS / 1000)
        data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
        results.append(f"{city}: {data['temp']}°C, {data['condition']}")
    return results


CITIES = ["Beijing", "Shanghai", "Shenzhen"]
N_RUNS = 3

print(f"\n{len(CITIES)} tool calls, each with {LATENCY_MS}ms simulated I/O latency ({N_RUNS} runs each):")

seq_times = []
for _ in range(N_RUNS):
    t0 = time.time()
    run_sequential(CITIES)
    seq_times.append((time.time() - t0) * 1000)

par_times = []
for _ in range(N_RUNS):
    t0 = time.time()
    asyncio.run(run_parallel(CITIES))
    par_times.append((time.time() - t0) * 1000)

avg_seq = sum(seq_times) / N_RUNS
avg_par = sum(par_times) / N_RUNS
speedup = avg_seq / avg_par

print(f"\n  Sequential  avg: {avg_seq:>7.1f}ms  (expected ~{LATENCY_MS * len(CITIES)}ms)")
print(f"  Parallel    avg: {avg_par:>7.1f}ms  (expected ~{LATENCY_MS}ms)")
print(f"  Speedup        : {speedup:.1f}x  ({(1 - avg_par / avg_seq) * 100:.0f}% faster)")
print()
print("Note: LangGraph's create_react_agent already handles parallel tool calls")
print("when the LLM emits multiple tool_calls in a single response. No asyncio")
print("boilerplate needed — the LLM just has to decide to call them together.")
print("Write tools as async functions to benefit automatically.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 4: Tool Result Cache — TTL-based memoization
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 4: Tool Result Cache — skip repeat tool calls")
print("=" * 70)

_cache: dict[str, tuple[str, float]] = {}  # key → (result, timestamp)
CACHE_TTL_S = 60.0

_stats = {"hits": 0, "misses": 0}


def get_weather_cached(city: str) -> tuple[str, bool]:
    """Return (result, cache_hit). Caches results for CACHE_TTL_S seconds."""
    key = f"weather:{city.lower()}"
    now = time.time()

    if key in _cache:
        result, ts = _cache[key]
        if now - ts < CACHE_TTL_S:
            _stats["hits"] += 1
            return result, True  # cache hit

    # Cache miss — call the real tool (simulate latency)
    _stats["misses"] += 1
    time.sleep(LATENCY_MS / 1000)
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    result = f"{city}: {data['temp']}°C, {data['condition']}"
    _cache[key] = (result, now)
    return result, False


CACHE_CASES = [
    ("Beijing",  "1st call"),
    ("Shanghai", "1st call"),
    ("Beijing",  "2nd call — should hit cache"),
    ("Shenzhen", "1st call"),
    ("Shanghai", "3rd call — should hit cache"),
    ("Beijing",  "4th call — should hit cache"),
]

print(f"\n  {'City':<12} {'Status':<14} {'Time':>8}  Note")
print(f"  {'-'*60}")

for city, note in CACHE_CASES:
    t0 = time.time()
    result, hit = get_weather_cached(city)
    elapsed_ms = (time.time() - t0) * 1000
    status = "HIT  ✓" if hit else "MISS"
    print(f"  {city:<12} {status:<14} {elapsed_ms:>6.1f}ms  {note}")

total = len(CACHE_CASES)
hit_rate = _stats["hits"] / total * 100
print(f"\n  Cache hits: {_stats['hits']}/{total} = {hit_rate:.0f}%")
print(f"  Misses:     {_stats['misses']}/{total}")
print()
print(f"  Miss avg latency : ~{LATENCY_MS}ms  (tool call)")
print(f"  Hit  avg latency : < 1ms         (dict lookup)")
print()
print("Takeaway: cache idempotent tools (same input → same output).")
print("Set TTL based on data freshness: weather 5-15 min, product pricing hours.")
print("Never cache tools with side effects (write, send, delete).")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Cost & Performance Optimization — Strategy Summary")
print("=" * 70)
print()
print(f"{'Strategy':<28} {'Reduces':<28} {'Typical gain'}")
print("-" * 72)
print(f"{'Trim system prompt':<28} {'Fixed input tokens/call':<28} {'10-40% input cost'}")
print(f"{'Prompt caching (Claude API)':<28} {'Repeated static context':<28} {'~90% on cached prefix'}")
print(f"{'Model routing':<28} {'Agent overhead':<28} {'40-60% calls skip agent'}")
print(f"{'Parallel tool calls':<28} {'Wall-clock latency':<28} {'~Nx speedup for N tools'}")
print(f"{'Tool result caching':<28} {'Redundant tool + LLM calls':<28} {'Depends on hit rate'}")
print()
print("Golden rule: measure first, then optimize.")
print("Log token counts and per-call latency — they tell you exactly")
print("where optimization effort has the highest return.")
print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
