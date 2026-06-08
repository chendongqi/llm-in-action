"""
Agent Harness — Production Module Demo

Shows the harness as a proper Python package (harness/) with:
  Part 1 — Standalone module usage (no LangGraph, pure Python)
  Part 2 — LangGraph integration: harness.execute() inside the tools node
  Part 3 — Complete audit trail + integrity verification

Scenario: a code-review assistant that can read tickets, draft fixes,
          create PRs (admin), and optionally merge to main (irreversible).

Run:
    conda activate dev_base
    python demo_production.py
"""

import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing import Annotated
from typing_extensions import TypedDict

from harness import (
    AgentHarness,
    BudgetExhaustedError,
    HumanApprovalRequired,
    PermissionError,
    PermissionLevel,
    RegisteredAction,
)

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)


# ── Mock handlers ──────────────────────────────────────────────────────────────

TICKETS = {
    "BUG-101": {"title": "NullPointerException in auth module", "priority": "P1",
                "status": "open"},
    "BUG-202": {"title": "Slow query in analytics dashboard",   "priority": "P2",
                "status": "open"},
}
DRAFTS: dict[str, str] = {}
PRS: list[dict]         = []
MERGED: list[str]       = []


def _read_ticket(ticket_id: str) -> dict:
    time.sleep(0.02)
    return TICKETS.get(ticket_id, {"error": f"Ticket '{ticket_id}' not found"})


def _write_draft(ticket_id: str, patch: str) -> str:
    time.sleep(0.02)
    DRAFTS[ticket_id] = patch
    return f"Draft saved for {ticket_id} ({len(patch)} chars)"


def _create_pr(ticket_id: str, title: str) -> str:
    time.sleep(0.02)
    pr_num = len(PRS) + 1
    PRS.append({"id": pr_num, "ticket": ticket_id, "title": title})
    return f"PR #{pr_num} created: '{title}'"


def _merge_to_main(pr_id: int) -> str:
    time.sleep(0.02)
    MERGED.append(str(pr_id))
    return f"PR #{pr_id} merged to main"


def sep(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Standalone module usage (no LangGraph)
# ══════════════════════════════════════════════════════════════════════════════

sep("Part 1 — Standalone AgentHarness.execute()")

harness = AgentHarness(budget=50, log_path="/tmp/prod_audit.jsonl")

# Register actions
harness.registry.register(RegisteredAction(
    "read_ticket", PermissionLevel.READ, 1, "Read a Jira ticket", _read_ticket))
harness.registry.register(RegisteredAction(
    "write_draft", PermissionLevel.WRITE, 3, "Write draft fix to branch", _write_draft))
harness.registry.register(RegisteredAction(
    "create_pr", PermissionLevel.ADMIN, 8, "Open a pull request", _create_pr))
harness.registry.register(RegisteredAction(
    "merge_to_main", PermissionLevel.IRREVERSIBLE, 20, "Merge PR to main", _merge_to_main))

print("\n1.1  Registered actions:")
for name in harness.registry.names():
    reg = harness.registry.get(name)
    print(f"  {name:<18}  level={reg.level.name:<14}  cost={reg.budget_cost}")

# 1.2 Normal flow: read → write draft → create PR
print("\n1.2  Normal flow: read → write draft → create PR")
r1 = harness.execute("read_ticket", ticket_id="BUG-101")
print(f"  read_ticket:  {r1}")

r2 = harness.execute("write_draft", ticket_id="BUG-101",
                     patch="- fix: add null check before auth.getUser()")
print(f"  write_draft:  {r2}")

r3 = harness.execute("create_pr", ticket_id="BUG-101",
                     title="fix: null check in auth module (BUG-101)")
print(f"  create_pr:    {r3}")
print(f"  {harness.budget.summary()}")

# 1.3 Unregistered action
print("\n1.3  Unregistered action — should raise PermissionError")
try:
    harness.execute("delete_all_data")
except PermissionError as e:
    print(f"  Blocked: {e}")

# 1.4 IRREVERSIBLE action → HumanApprovalRequired
print("\n1.4  IRREVERSIBLE action — raises HumanApprovalRequired")
try:
    harness.execute("merge_to_main", pr_id=1)
except HumanApprovalRequired as e:
    print(f"  Intercepted: {e}")
    print(f"  Action: {e.action_name}  Args: {e.action_args}")
    print("  Human approves → calling approve_and_execute()")
    result = harness.approve_and_execute("merge_to_main", pr_id=1)
    print(f"  Executed: {result}")

# 1.5 Budget exhaustion
print("\n1.5  Budget exhaustion (write_draft cost=3, budget now limited)")
small_harness = AgentHarness(budget=5, log_path="/tmp/budget_test.jsonl")
small_harness.registry.register(RegisteredAction(
    "write_draft", PermissionLevel.WRITE, 3, "Write draft", _write_draft))

for i in range(1, 4):
    try:
        small_harness.execute("write_draft", ticket_id=f"T-{i}", patch="patch")
        print(f"  write #{i}: OK  — {small_harness.budget.summary()}")
    except BudgetExhaustedError as e:
        print(f"  write #{i}: BLOCKED — {e}")

# 1.6 Rollback
print("\n1.6  Rollback: write_draft fails mid-transaction")
rollback_harness = AgentHarness(budget=30, log_path="/tmp/rollback_test.jsonl")
rollback_harness.registry.register(RegisteredAction(
    "write_draft", PermissionLevel.WRITE, 3, "Write draft", _write_draft))

original_drafts = dict(DRAFTS)

def _failing_write(ticket_id: str, patch: str) -> str:
    DRAFTS[ticket_id] = patch  # modify state
    raise RuntimeError("Disk full — write failed")

rollback_harness.registry.register(RegisteredAction(
    "write_fail", PermissionLevel.WRITE, 3, "Failing write", _failing_write))

# Load current DRAFTS into harness state for rollback demo
rollback_harness._state.update({"drafts": dict(DRAFTS)})
snap = dict(DRAFTS)  # snapshot before the transaction
print(f"  DRAFTS before: {list(DRAFTS)}")

try:
    with rollback_harness.rollback.transaction(rollback_harness._state, "failing_patch"):
        _failing_write("BUG-999", "bad patch")
except RuntimeError as e:
    DRAFTS.clear()
    DRAFTS.update(snap)  # restore from snapshot
    print(f"  Write failed: {e}")
    print(f"  DRAFTS after rollback: {list(DRAFTS)}  ← BUG-999 not present ✓")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — LangGraph integration
# ══════════════════════════════════════════════════════════════════════════════

sep("Part 2 — LangGraph integration: harness inside tools node")

# Define LangChain tools (what the LLM sees)
@lc_tool
def read_ticket(ticket_id: str) -> str:
    """Read a Jira ticket by ID. Returns title, priority, and status."""
    data = TICKETS.get(ticket_id, {"error": "not found"})
    return str(data)


@lc_tool
def write_draft(ticket_id: str, patch: str) -> str:
    """Save a draft code fix for the given ticket ID."""
    DRAFTS[ticket_id] = patch
    return f"Draft saved for {ticket_id}"


@lc_tool
def create_pr(ticket_id: str, title: str) -> str:
    """Create a pull request for the given ticket."""
    pr_num = len(PRS) + 1
    PRS.append({"id": pr_num, "ticket": ticket_id, "title": title})
    return f"PR #{pr_num} created"


@lc_tool
def merge_to_main(pr_id: int) -> str:
    """Merge a pull request to the main branch. This is irreversible."""
    MERGED.append(str(pr_id))
    return f"PR #{pr_id} merged"


LG_TOOLS = [read_ticket, write_draft, create_pr, merge_to_main]
TOOL_MAP = {t.name: t for t in LG_TOOLS}

# Separate harness instance for LangGraph demo
lg_harness = AgentHarness(budget=60, log_path="/tmp/lg_audit.jsonl")
lg_harness.registry.register(RegisteredAction(
    "read_ticket",  PermissionLevel.READ,        1,  "Read ticket",   _read_ticket))
lg_harness.registry.register(RegisteredAction(
    "write_draft",  PermissionLevel.WRITE,        3,  "Write draft",   _write_draft))
lg_harness.registry.register(RegisteredAction(
    "create_pr",    PermissionLevel.ADMIN,         8,  "Create PR",     _create_pr))
lg_harness.registry.register(RegisteredAction(
    "merge_to_main",PermissionLevel.IRREVERSIBLE, 20, "Merge to main", _merge_to_main))


class HState(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
bound_model = llm.bind_tools(LG_TOOLS)


def agent_node(state: HState) -> dict:
    return {"messages": [bound_model.invoke(state["messages"])]}


def tools_node(state: HState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        try:
            reg = lg_harness.registry.get(name)
            lg_harness.budget.spend(name, reg.budget_cost)

            if reg.level == PermissionLevel.IRREVERSIBLE:
                decision = interrupt({
                    "tool": name, "args": args,
                    "message": f"IRREVERSIBLE '{name}'. Approve?",
                })
                if decision != "approved":
                    lg_harness.budget.refund(name, reg.budget_cost)
                    lg_harness.audit.log(name, "checkpoint", str(args), "HUMAN_REJECTED")
                    results.append(ToolMessage(
                        content=f"'{name}' rejected by human.",
                        tool_call_id=tc["id"],
                    ))
                    continue

            if reg.level in (PermissionLevel.WRITE, PermissionLevel.ADMIN):
                with lg_harness.rollback.transaction(lg_harness._state, name):
                    output = TOOL_MAP[name].invoke(args)
            else:
                output = TOOL_MAP[name].invoke(args)

            lg_harness.audit.log(name, "agent", str(args), "EXECUTED",
                                 {"level": reg.level.name})
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

        except PermissionError as e:
            lg_harness.audit.log(name, "registry", str(args), "BLOCKED")
            results.append(ToolMessage(content=str(e), tool_call_id=tc["id"]))
        except BudgetExhaustedError as e:
            results.append(ToolMessage(content=str(e), tool_call_id=tc["id"]))

    return {"messages": results}


def router(state: HState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


g = StateGraph(HState)
g.add_node("agent", agent_node)
g.add_node("tools", tools_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", router)
g.add_edge("tools", "agent")
app = g.compile(checkpointer=memory)


def run_agent(query: str, thread_id: str, auto_approve: bool = True) -> str:
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [HumanMessage(query)]}, config=cfg)

    state = app.get_state(cfg)
    while state.next and state.tasks and state.tasks[0].interrupts:
        iv = state.tasks[0].interrupts[0].value
        tool_name = iv.get("tool", "?") if isinstance(iv, dict) else str(iv)
        decision = "approved" if auto_approve else "rejected"
        print(f"  [LG Checkpoint] '{tool_name}' → '{decision}'")
        app.invoke(Command(resume=decision), config=cfg)
        state = app.get_state(cfg)

    for m in reversed(app.get_state(cfg).values["messages"]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return str(m.content)
    return "(no answer)"


print("\n2.1  Read a ticket (READ, auto-executes)")
ans = run_agent("What is in ticket BUG-202?", "lg-1")
print(f"  Answer: {ans[:80]}")
print(f"  {lg_harness.budget.summary()}")

print("\n2.2  Write draft + create PR (WRITE + ADMIN)")
ans = run_agent(
    "Save a draft fix for BUG-202: 'add index on analytics.event_ts', "
    "then create a PR titled 'perf: index on event_ts (BUG-202)'.",
    "lg-2",
)
print(f"  Answer: {ans[:80]}")
print(f"  {lg_harness.budget.summary()}")

print("\n2.3  Merge to main (IRREVERSIBLE, auto-approved)")
ans = run_agent("Merge PR 2 to main.", "lg-3", auto_approve=True)
print(f"  Answer: {ans[:80]}")
print(f"  {lg_harness.budget.summary()}")

print("\n2.4  Merge to main (IRREVERSIBLE, rejected by human)")
ans = run_agent("Merge PR 1 to main.", "lg-4", auto_approve=False)
print(f"  Answer: {ans[:80]}")
print(f"  Merged list: {MERGED}")


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Audit trail + integrity verification
# ══════════════════════════════════════════════════════════════════════════════

sep("Part 3 — Audit Trail + Integrity Verification")

print(f"\n  Total LangGraph audit entries: {len(lg_harness.audit)}")
print(f"\n  Last 8 entries:")
print(f"  {'Time':<9} {'Action':<18} {'Actor':<12} {'Result':<22} Level")
print(f"  {'-'*72}")
for e in lg_harness.audit.tail(8):
    lvl = e.get("metadata", {}).get("level", "")
    print(f"  {e['ts']:<9} {e['action']:<18} {e['actor']:<12} {e['result']:<22} {lvl}")

ok = lg_harness.audit.verify_integrity()
print(f"\n  verify_integrity() = {ok}  {'✓' if ok else '✗ — TAMPERED!'}")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
