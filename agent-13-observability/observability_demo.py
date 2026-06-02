"""
Agent Observability Demo

Three observability patterns:
  Demo 1 — Live Trace:      real-time event stream during execution
  Demo 2 — Step Timeline:   per-step latency breakdown (where the time goes)
  Demo 3 — Audit Log:       JSON-serializable trace for production debugging

Run:
    conda activate dev_base
    python observability_demo.py
"""

import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
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

# ── Mock data & tools ──────────────────────────────────────────────────────────

MOCK_WEATHER = {
    "beijing":  {"temp": 25, "condition": "sunny"},
    "shanghai": {"temp": 22, "condition": "cloudy"},
    "shenzhen": {"temp": 30, "condition": "rainy"},
}

MOCK_PRODUCTS = {
    "wonderbot basic": {"price": 99,  "api_calls": 10_000},
    "wonderbot pro":   {"price": 299, "api_calls": 100_000},
}


@lc_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return json.dumps({"city": city, **data})


@lc_tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    import math
    allowed = set("0123456789 +-*/.()** ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric operators allowed"
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@lc_tool
def get_product_info(product_name: str) -> str:
    """Get pricing and API limits for WonderBot plans."""
    data = MOCK_PRODUCTS.get(product_name.lower(), None)
    if data is None:
        return f"Product '{product_name}' not found. Available: {list(MOCK_PRODUCTS)}"
    return json.dumps({"product": product_name, **data})


agent = create_react_agent(model=llm, tools=[get_weather, calculator, get_product_info])


def get_final_answer(output: dict) -> str:
    for m in reversed(output["messages"]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return str(m.content)
    return ""


# ── Tracer ─────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step_type: str       # "llm" | "tool"
    name: str
    input_preview: str
    output_preview: str
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class AgentTracer(BaseCallbackHandler):
    """Captures LLM calls and tool invocations with per-step timing."""

    def __init__(self, verbose: bool = True, trace_id: str = "") -> None:
        super().__init__()
        self.verbose = verbose
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.steps: list[StepRecord] = []
        self._llm_t0: float = 0.0
        self._tool_t0: float = 0.0
        self._tool_name: str = ""
        self._tool_input: str = ""

    def reset(self) -> None:
        self.steps = []
        self.trace_id = str(uuid.uuid4())[:8]

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: object) -> None:
        self._llm_t0 = time.time()
        if self.verbose:
            print(f"  [LLM →] reasoning...")

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        t1 = time.time()
        output = ""
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg:
                content = msg.content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            output = part.get("text", "")[:80]
                            break
                else:
                    output = str(content)[:80]
        except (IndexError, AttributeError):
            pass

        step = StepRecord(
            step_type="llm", name="LLM",
            input_preview="", output_preview=output,
            start_time=self._llm_t0, end_time=t1,
        )
        self.steps.append(step)
        if self.verbose:
            print(f"  [LLM ←] {step.duration_ms:.0f}ms  |  {output[:70]}")

    # ── Tool events ───────────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: object) -> None:
        self._tool_t0 = time.time()
        self._tool_name = serialized.get("name", "tool")
        self._tool_input = str(input_str)[:60]
        if self.verbose:
            print(f"  [TOOL→] {self._tool_name}({self._tool_input})")

    def on_tool_end(self, output: str, **kwargs: object) -> None:
        t1 = time.time()
        step = StepRecord(
            step_type="tool", name=self._tool_name,
            input_preview=self._tool_input,
            output_preview=str(output)[:80],
            start_time=self._tool_t0, end_time=t1,
        )
        self.steps.append(step)
        if self.verbose:
            print(f"  [TOOL←] {step.output_preview}  [{step.duration_ms:.0f}ms]")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Live Trace
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 1: Live Trace — real-time event stream")
print("=" * 70)

QUERY_1 = (
    "What is the weather in Beijing and Shanghai? "
    "Calculate the temperature difference."
)
print(f"\nQuery: {QUERY_1}\n")

tracer1 = AgentTracer(verbose=True, trace_id="demo1")
result1 = agent.invoke(
    {"messages": [HumanMessage(QUERY_1)]},
    config={"callbacks": [tracer1]},
)
answer1 = get_final_answer(result1)

print(f"\nFinal answer: {answer1[:200]}")
llm_steps  = sum(1 for s in tracer1.steps if s.step_type == "llm")
tool_steps = sum(1 for s in tracer1.steps if s.step_type == "tool")
print(f"\nTrace summary: {len(tracer1.steps)} steps  "
      f"({llm_steps} LLM calls, {tool_steps} tool calls)")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — Step Timeline
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 2: Step Timeline — where does the latency live?")
print("=" * 70)

QUERY_2 = "Tell me the WonderBot Pro price and calculate 299 * 12 for annual cost."
print(f"\nQuery: {QUERY_2}\n")

tracer2 = AgentTracer(verbose=False)
result2 = agent.invoke(
    {"messages": [HumanMessage(QUERY_2)]},
    config={"callbacks": [tracer2]},
)
answer2 = get_final_answer(result2)

total_ms = sum(s.duration_ms for s in tracer2.steps)
llm_ms   = sum(s.duration_ms for s in tracer2.steps if s.step_type == "llm")
tool_ms  = sum(s.duration_ms for s in tracer2.steps if s.step_type == "tool")
bar_scale = (40 / total_ms) if total_ms > 0 else 1

print("Step-by-step breakdown:\n")
for i, step in enumerate(tracer2.steps, 1):
    bar   = "█" * max(int(step.duration_ms * bar_scale), 1)
    label = "LLM reasoning" if step.step_type == "llm" else f"tool: {step.name}"
    print(f"  Step {i}  {label:<25} [{step.duration_ms:>6.0f}ms]  {bar}")

print(f"\n  {'─' * 60}")
print(f"  Total : {total_ms:.0f}ms")
print(f"  LLM   : {llm_ms:.0f}ms  ({llm_ms / total_ms * 100:.1f}% of wall time)")
print(f"  Tools : {tool_ms:.0f}ms  ({tool_ms / total_ms * 100:.1f}% of wall time)")
print(f"\nFinal answer: {answer2[:150]}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — Structured Audit Log
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 3: Audit Log — JSON-serializable trace for production")
print("=" * 70)

QUERY_3 = "What's the weather in Shenzhen and how much does WonderBot Basic cost?"
print(f"\nQuery: {QUERY_3}\n")

tracer3 = AgentTracer(verbose=False)
result3 = agent.invoke(
    {"messages": [HumanMessage(QUERY_3)]},
    config={"callbacks": [tracer3]},
)
answer3 = get_final_answer(result3)


def build_audit_log(tracer: AgentTracer, query: str, answer: str) -> dict:
    steps_log = []
    for s in tracer.steps:
        entry: dict = {"type": s.step_type, "duration_ms": round(s.duration_ms, 1)}
        if s.step_type == "tool":
            entry["tool"]   = s.name
            entry["input"]  = s.input_preview
            entry["output"] = s.output_preview
        else:
            entry["output_preview"] = s.output_preview
        steps_log.append(entry)

    total    = sum(s.duration_ms for s in tracer.steps)
    llm_tot  = sum(s.duration_ms for s in tracer.steps if s.step_type == "llm")
    tool_tot = total - llm_tot

    return {
        "trace_id": tracer.trace_id,
        "query":    query,
        "answer":   answer,
        "steps":    steps_log,
        "summary": {
            "step_count":      len(tracer.steps),
            "tool_call_count": sum(1 for s in tracer.steps if s.step_type == "tool"),
            "total_ms":        round(total, 1),
            "llm_ms":          round(llm_tot, 1),
            "tool_ms":         round(tool_tot, 1),
        },
    }


audit = build_audit_log(tracer3, QUERY_3, answer3)
print("Audit log (JSON):\n")
print(json.dumps(audit, indent=2, ensure_ascii=False))


# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Observability Patterns Summary")
print("=" * 70)
print()
print(f"{'Pattern':<20} {'When to use':<38} {'Output format'}")
print("-" * 72)
print(f"{'Live trace':<20} {'Development, debugging':<38} {'Printed event stream'}")
print(f"{'Step timeline':<20} {'Performance analysis':<38} {'ASCII latency chart'}")
print(f"{'Audit log':<20} {'Production, compliance':<38} {'JSON record per request'}")
print()
print("Key insight: LLM accounts for ~99% of latency. Tools are effectively free.")
print("Reduce Agent latency by minimising LLM calls, not tool calls.")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
