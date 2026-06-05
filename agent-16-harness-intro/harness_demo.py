"""
Agent Harness Engineering Demo

Five elements of a controllable agent execution framework:
  Element 1 — Action Space:       whitelist blocks unauthorized tool calls
  Element 2 — Human Checkpoint:   LangGraph interrupt pauses before risky ops
  Element 3 — Execution Boundary: max-step cap prevents runaway agents
  Element 4 — Audit Log:          append-only record woven through all elements
  Element 5 — Rollback:           snapshot / restore on write failure

Run:
    conda activate dev_base
    python harness_demo.py
"""

import copy
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

# ── Mock data ──────────────────────────────────────────────────────────────────

MOCK_REPORTS = {
    "q1_sales":       "Q1 Sales: $1.2M revenue, 340 customers, 15% growth YoY.",
    "security_audit": "Security Audit: 3 critical issues, 7 high, 12 medium found.",
}

SYSTEM_CONFIG: dict = {"version": "1.0", "timeout": 30, "max_retries": 3}


# ══════════════════════════════════════════════════════════════════════════════
# Element 4: Audit Log  (used by all other elements)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEntry:
    ts: str
    action: str
    risk: str
    result: str
    note: str = ""

AUDIT: list[AuditEntry] = []


def audit(action: str, risk: str, result: str, note: str = "") -> None:
    AUDIT.append(AuditEntry(time.strftime("%H:%M:%S"), action, risk, result, note))


# ══════════════════════════════════════════════════════════════════════════════
# Element 1: Action Space Registry
# ══════════════════════════════════════════════════════════════════════════════
#
# Every allowed operation is declared here.
# Anything absent from the registry is automatically blocked.

ACTION_SPACE: dict[str, dict] = {
    "read_report":   {"risk": "safe",  "needs_approval": False},
    "write_report":  {"risk": "risky", "needs_approval": True},
    # "delete_records" is intentionally absent → will be blocked
}


# ── Tools ──────────────────────────────────────────────────────────────────────

@lc_tool
def read_report(report_name: str) -> str:
    """Read a business report by name.

    Returns report content as plain text.
    Use this when the user asks to view, read, or check a report.
    Available reports: q1_sales, security_audit.
    Pass the report name as a plain string, e.g. 'q1_sales'.
    """
    data = MOCK_REPORTS.get(report_name.lower().replace(" ", "_"))
    if data is None:
        return f"Report '{report_name}' not found. Available: {list(MOCK_REPORTS)}"
    audit("read_report", "safe", "executed", f"report={report_name}")
    return data


@lc_tool
def write_report(filename: str, content: str) -> str:
    """Save a report summary to a file.

    This is a WRITE operation that modifies the filesystem.
    Use only when the user explicitly asks to save or write output to a file.
    Pass filename (e.g. 'output.txt') and the content string.
    """
    audit("write_report", "risky", "executed", f"file={filename}")
    return f"Report saved to '{filename}' ({len(content)} chars)."


@lc_tool
def delete_records(table: str) -> str:
    """Delete all records from a database table.

    DANGEROUS: this operation is irreversible.
    Pass the table name as a string, e.g. 'users'.
    """
    # This tool is NOT in ACTION_SPACE — the harness blocks it before execution.
    return "should never reach here"


ALL_TOOLS = [read_report, write_report, delete_records]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
LLM_WITH_TOOLS = llm.bind_tools(ALL_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# Core: custom LangGraph graph with harness checks in the tools node
# ══════════════════════════════════════════════════════════════════════════════

class HarnessState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: HarnessState) -> dict:
    return {"messages": [LLM_WITH_TOOLS.invoke(state["messages"])]}


def harness_tools_node(state: HarnessState) -> dict:
    """Execute tool calls with action-space and human-checkpoint enforcement."""
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]

        # ── Element 1: Action Space check ─────────────────────────────────────
        if name not in ACTION_SPACE:
            audit(name, "blocked", "BLOCKED", "not in action space")
            result_text = (
                f"ERROR: '{name}' is not in the allowed action space. "
                f"Allowed tools: {list(ACTION_SPACE)}."
            )

        # ── Element 2: Human Checkpoint ───────────────────────────────────────
        elif ACTION_SPACE[name]["needs_approval"]:
            # interrupt() pauses the graph here and returns the resume value
            decision = interrupt({
                "tool":    name,
                "args":    args,
                "message": (
                    f"Agent wants to call '{name}' with args {args}. "
                    "Approve? Send 'approved' to proceed, anything else to reject."
                ),
            })
            if decision == "approved":
                audit(name, "risky", "executed", "human approved")
                result_text = str(TOOL_MAP[name].invoke(args))
            else:
                audit(name, "risky", "rejected", f"human rejected (decision='{decision}')")
                result_text = f"Operation '{name}' was rejected by the human reviewer."

        # ── Safe operation: auto-execute ──────────────────────────────────────
        else:
            result_text = str(TOOL_MAP[name].invoke(args))

        results.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))

    return {"messages": results}


def should_continue(state: HarnessState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    return "tools" if (isinstance(last, AIMessage) and last.tool_calls) else "__end__"


harness_graph = StateGraph(HarnessState)
harness_graph.add_node("agent", agent_node)
harness_graph.add_node("tools", harness_tools_node)
harness_graph.add_edge(START, "agent")
harness_graph.add_conditional_edges("agent", should_continue)
harness_graph.add_edge("tools", "agent")

harness_app = harness_graph.compile(checkpointer=MemorySaver())


def run_harness(query: str, thread_id: str, auto_decision: str = "approved") -> None:
    """Run the harness agent; auto_decision simulates the human reviewer."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n  Query: '{query}'")

    result = harness_app.invoke(
        {"messages": [HumanMessage(query)]},
        config=config,
    )

    state = harness_app.get_state(config)
    if state.next:
        # Graph is paused at interrupt() — display checkpoint info
        try:
            interrupt_data = state.tasks[0].interrupts[0].value
            print(f"  [HARNESS] ⚠️  Checkpoint triggered:")
            print(f"            Tool : {interrupt_data.get('tool')}")
            print(f"            Args : {interrupt_data.get('args')}")
        except (IndexError, AttributeError):
            print(f"  [HARNESS] ⚠️  Checkpoint triggered (details unavailable)")
        print(f"  [HARNESS] Simulating human decision: '{auto_decision}'")

        result = harness_app.invoke(Command(resume=auto_decision), config=config)

    final = [
        m for m in result["messages"]
        if isinstance(m, AIMessage) and not m.tool_calls
    ]
    answer = final[-1].content if final else "(no response)"
    print(f"  Answer: {str(answer)[:150]}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1: Action Space — unauthorized tool is blocked
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 1: Action Space — unauthorized tool is blocked")
print("=" * 70)

run_harness(
    "Delete all records from the users table.",
    thread_id="demo1-blocked",
)

# ══════════════════════════════════════════════════════════════════════════════
# Demo 2: Human Checkpoint — safe operation (no interrupt)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 2: Human Checkpoint — safe read operation, no pause needed")
print("=" * 70)

run_harness(
    "What is in the q1_sales report?",
    thread_id="demo2-safe",
)

# ══════════════════════════════════════════════════════════════════════════════
# Demo 3: Human Checkpoint — risky write operation, human APPROVES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 3: Human Checkpoint — risky write operation, human APPROVES")
print("=" * 70)

run_harness(
    "Save the q1_sales report summary to 'output.txt'.",
    thread_id="demo3-approve",
    auto_decision="approved",
)

# ══════════════════════════════════════════════════════════════════════════════
# Demo 4: Human Checkpoint — risky write operation, human REJECTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 4: Human Checkpoint — risky write operation, human REJECTS")
print("=" * 70)

run_harness(
    "Write a file called 'override.txt' with content 'Access granted'.",
    thread_id="demo4-reject",
    auto_decision="rejected",
)

# ══════════════════════════════════════════════════════════════════════════════
# Demo 5: Execution Boundary — max-step cap prevents runaway agents
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 5: Execution Boundary — max-step cap")
print("=" * 70)

boundary_agent = create_react_agent(
    model=llm,
    tools=[read_report, write_report],
)


def run_bounded(query: str, max_steps: int) -> dict:
    """Wrap agent execution with a hard ceiling on tool-call steps."""
    messages = [HumanMessage(query)]
    steps = 0

    while True:
        result = boundary_agent.invoke({"messages": messages})
        msgs = result["messages"]
        steps += sum(1 for m in msgs if isinstance(m, AIMessage) and m.tool_calls)

        last = msgs[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            audit("agent_run", "system", "completed", f"steps={steps}")
            return {"status": "completed", "steps": steps, "answer": str(last.content)[:120]}

        if steps >= max_steps:
            audit("agent_run", "system", "stopped_max_steps", f"limit={max_steps}")
            return {
                "status": "stopped_max_steps",
                "steps": steps,
                "answer": f"[Harness] Execution stopped: {max_steps}-step limit reached.",
            }
        messages = msgs


print()
BOUNDARY_CASES = [
    ("What is in the q1_sales report?", 5, "simple query — should complete normally"),
    (
        "Read both q1_sales and security_audit, then save a combined report to combined.txt",
        1,
        "multi-step — intentionally tight limit",
    ),
]

for query, limit, label in BOUNDARY_CASES:
    print(f"[{label}]  max_steps={limit}")
    r = run_bounded(query, limit)
    print(f"  Status : {r['status']}  |  Steps used: {r['steps']}")
    print(f"  Answer : {r['answer']}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Demo 6: Rollback — snapshot / restore on write failure
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Demo 6: Rollback — snapshot / restore on write failure")
print("=" * 70)


@contextmanager
def rollback_on_failure(state: dict, op_name: str):
    """Capture a deep copy before the block; restore it if the block raises."""
    snapshot = copy.deepcopy(state)
    print(f"\n  Snapshot taken before '{op_name}': {snapshot}")
    try:
        yield state
        audit(op_name, "write", "committed")
        print(f"  '{op_name}' committed — no rollback needed.")
    except Exception as exc:
        state.clear()
        state.update(snapshot)
        audit(op_name, "write", "rolled_back", str(exc))
        print(f"  '{op_name}' FAILED ({exc})")
        print(f"  State restored: {state}")


print()
print("Test A — successful update:")
with rollback_on_failure(SYSTEM_CONFIG, "update_timeout"):
    SYSTEM_CONFIG["timeout"] = 60
print(f"  Final state: {SYSTEM_CONFIG}")

print("\nTest B — failed update (rollback triggered):")
try:
    with rollback_on_failure(SYSTEM_CONFIG, "bad_version_bump"):
        SYSTEM_CONFIG["version"] = "2.0"
        raise ValueError("Version 2.0 is incompatible with current schema")
except ValueError:
    pass
print(f"  Final state: {SYSTEM_CONFIG}")


# ══════════════════════════════════════════════════════════════════════════════
# Audit Trail summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Complete Audit Trail")
print("=" * 70)
print()
print(f"{'Time':10} {'Risk':10} {'Result':16} Action  (note)")
print("-" * 70)
for e in AUDIT:
    print(f"{e.ts:10} {e.risk:10} {e.result:16} {e.action}  {e.note}")

print("\n" + "=" * 70)
print("Harness Engineering — Five Elements")
print("=" * 70)
print()
print(f"{'Element':<24} {'What it prevents':<38} Demo")
print("-" * 70)
print(f"{'1. Action Space':<24} {'Unauthorized tool execution':<38} Demo 1")
print(f"{'2. Human Checkpoint':<24} {'Unsupervised risky operations':<38} Demo 2-4")
print(f"{'3. Execution Boundary':<24} {'Runaway / infinite-loop agents':<38} Demo 5")
print(f"{'4. Audit Log':<24} {'Untraceable agent behavior':<38} All demos")
print(f"{'5. Rollback':<24} {'Unrecoverable write failures':<38} Demo 6")
print()
print("Golden rule: autonomous ≠ uncontrolled.")
print("A harness lets an agent be trusted, not just capable.")
print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
