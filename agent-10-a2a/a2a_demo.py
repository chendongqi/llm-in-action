"""
A2A Protocol Demo

Three demos:
  Demo 1 — Direct Agent Calls: agents as Python functions, hardwired coupling
  Demo 2 — A2A-style: AgentCard registry, capability discovery, task delegation
  Demo 3 — LLM-Driven Discovery: orchestrator reads AgentCards and decides routing

Run:
    conda activate dev_base
    python a2a_demo.py
"""

import os
import uuid
import warnings
from dataclasses import dataclass
from typing import Callable

warnings.filterwarnings("ignore", category=DeprecationWarning)

from a2a.types import (  # type: ignore[import-untyped]
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)


def _ask(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(system), HumanMessage(user)])
    return str(resp.content)


# ── helpers to build a2a proto objects ───────────────────────────────────────

def make_skill(skill_id: str, name: str, description: str, tags: list[str]) -> AgentSkill:
    s = AgentSkill()
    s.id = skill_id
    s.name = name
    s.description = description
    s.tags.extend(tags)
    return s


def make_card(name: str, description: str, skills: list[AgentSkill]) -> AgentCard:
    c = AgentCard()
    c.name = name
    c.description = description
    c.skills.extend(skills)
    return c


def make_task(input_text: str, context_id: str = "") -> Task:
    t = Task()
    t.id = str(uuid.uuid4())
    t.context_id = context_id or str(uuid.uuid4())
    t.status.CopyFrom(TaskStatus())
    t.status.state = TaskState.TASK_STATE_SUBMITTED
    msg = Message()
    msg.message_id = str(uuid.uuid4())
    msg.role = Role.ROLE_USER
    part = Part()
    part.text = input_text
    msg.parts.append(part)
    t.history.append(msg)
    return t


def complete_task(task: Task, result_text: str) -> Task:
    """Mark task completed and attach result as agent message."""
    task.status.state = TaskState.TASK_STATE_COMPLETED
    msg = Message()
    msg.message_id = str(uuid.uuid4())
    msg.role = Role.ROLE_AGENT
    part = Part()
    part.text = result_text
    msg.parts.append(part)
    task.history.append(msg)
    return task


def task_input(task: Task) -> str:
    for msg in task.history:
        if msg.role == Role.ROLE_USER and msg.parts:
            return msg.parts[0].text
    return ""


def task_output(task: Task) -> str:
    for msg in reversed(task.history):
        if msg.role == Role.ROLE_AGENT and msg.parts:
            return msg.parts[0].text
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Direct Agent Calls
# ══════════════════════════════════════════════════════════════════════════════
#
# Agents are Python functions. The orchestrator calls them directly.
# Simple, but: each agent is a hard dependency — no discovery, no decoupling.

print("\n" + "=" * 70)
print("Demo 1: Direct Agent Calls (no protocol)")
print("=" * 70)

QUESTION = "Explain the tradeoffs between Python and Go for a new microservice."


def research_agent_fn(question: str) -> str:
    return _ask(
        "You are a research specialist. List 4 factual key points about the topic.",
        question,
    )


def analysis_agent_fn(research: str) -> str:
    return _ask(
        "You are an analyst. Given the research, summarize the tradeoffs in 3 bullets.",
        f"Research:\n{research}",
    )


def writing_agent_fn(analysis: str) -> str:
    return _ask(
        "You are a technical writer. Compose a 100-word decision guide from the analysis.",
        f"Analysis:\n{analysis}",
    )


def direct_orchestrator(question: str) -> str:
    """Hardwired pipeline — each agent is a direct function call."""
    print("  → calling research_agent (direct)")
    research = research_agent_fn(question)
    print("  → calling analysis_agent (direct)")
    analysis = analysis_agent_fn(research)
    print("  → calling writing_agent (direct)")
    answer = writing_agent_fn(analysis)
    return answer


print(f"\nQuestion: {QUESTION}\n")
direct_answer = direct_orchestrator(QUESTION)
print(f"\nAnswer (first 200 chars):\n{direct_answer[:200]}...")
print("\nDirect calls work — but research_agent, analysis_agent, writing_agent")
print("are hardcoded dependencies. Swap one out? Edit the orchestrator.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — A2A-style: AgentCard Registry + Task Delegation
# ══════════════════════════════════════════════════════════════════════════════
#
# Each agent publishes an AgentCard with its capabilities.
# The registry enables discovery by tag. The orchestrator finds the right
# agent at runtime — no hardcoded names in the orchestrator.

print("\n" + "=" * 70)
print("Demo 2: A2A Agent Cards + Registry + Task Delegation")
print("=" * 70)


@dataclass
class AgentEntry:
    card: AgentCard
    handler: Callable[[Task], Task]


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentEntry] = {}

    def register(self, card: AgentCard, handler: Callable[[Task], Task]) -> None:
        self._agents[card.name] = AgentEntry(card=card, handler=handler)
        print(f"  [registry] registered: {card.name}")

    def discover(self, tag: str) -> list[AgentCard]:
        """Return all agents whose skills include the given tag."""
        results = []
        for entry in self._agents.values():
            for skill in entry.card.skills:
                if tag in list(skill.tags):
                    results.append(entry.card)
                    break
        return results

    def delegate(self, agent_name: str, input_text: str) -> Task:
        """Create a task and execute it via the registered handler."""
        if agent_name not in self._agents:
            raise KeyError(f"Agent '{agent_name}' not registered")
        task = make_task(input_text)
        task.status.state = TaskState.TASK_STATE_WORKING
        return self._agents[agent_name].handler(task)


registry = AgentRegistry()

print("\nRegistering agents with AgentCards:")

# ── research agent ────────────────────────────────────────────────────────────
research_card = make_card(
    name="research-agent",
    description="Gathers factual background on technical topics",
    skills=[make_skill("research", "Research", "Collect key facts on a topic", ["research", "facts"])],
)


def research_handler(task: Task) -> Task:
    result = research_agent_fn(task_input(task))
    return complete_task(task, result)


registry.register(research_card, research_handler)

# ── analysis agent ────────────────────────────────────────────────────────────
analysis_card = make_card(
    name="analysis-agent",
    description="Analyzes research notes and extracts tradeoffs and conclusions",
    skills=[make_skill("analysis", "Analysis", "Summarize tradeoffs from research", ["analysis", "tradeoffs"])],
)


def analysis_handler(task: Task) -> Task:
    result = analysis_agent_fn(task_input(task))
    return complete_task(task, result)


registry.register(analysis_card, analysis_handler)

# ── writing agent ─────────────────────────────────────────────────────────────
writing_card = make_card(
    name="writing-agent",
    description="Composes clear technical prose from analysis output",
    skills=[make_skill("writing", "Writing", "Write decision guides and summaries", ["writing", "prose"])],
)


def writing_handler(task: Task) -> Task:
    result = writing_agent_fn(task_input(task))
    return complete_task(task, result)


registry.register(writing_card, writing_handler)

print(f"\nDiscovery demo — agents with 'research' capability:")
researchers = registry.discover("research")
for c in researchers:
    print(f"  Found: {c.name} — {c.description}")

print(f"\nDiscovery demo — agents with 'writing' capability:")
writers = registry.discover("writing")
for c in writers:
    print(f"  Found: {c.name} — {c.description}")


def a2a_orchestrator(question: str) -> str:
    """Discover agents at runtime, delegate via tasks — no hardcoded names."""
    researchers = registry.discover("research")
    assert researchers, "no research agent found"
    print(f"  → delegating to {researchers[0].name} (discovered via tag)")
    t1 = registry.delegate(researchers[0].name, question)
    research = task_output(t1)

    analysts = registry.discover("analysis")
    assert analysts, "no analysis agent found"
    print(f"  → delegating to {analysts[0].name} (discovered via tag)")
    t2 = registry.delegate(analysts[0].name, research)
    analysis = task_output(t2)

    writers = registry.discover("writing")
    assert writers, "no writing agent found"
    print(f"  → delegating to {writers[0].name} (discovered via tag)")
    t3 = registry.delegate(writers[0].name, analysis)
    return task_output(t3)


print(f"\nQuestion: {QUESTION}\n")
a2a_answer = a2a_orchestrator(QUESTION)
print(f"\nAnswer (first 200 chars):\n{a2a_answer[:200]}...")
print("\nOrchestrator used registry.discover() — never named any agent explicitly.")
print("Swap writing-agent for a new one? Register it with 'writing' tag. Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — LLM-Driven Agent Discovery
# ══════════════════════════════════════════════════════════════════════════════
#
# The orchestrator asks the LLM to read the available AgentCards and decide
# which agents to call for a given task, in what order.
# This is the A2A vision: agents finding and working with each other
# based on capability descriptions, not hardcoded references.

print("\n" + "=" * 70)
print("Demo 3: LLM-Driven Agent Discovery")
print("=" * 70)

# Build a catalog description from the registered cards
catalog_lines = []
for name, entry in registry._agents.items():
    skills_str = "; ".join(
        f"{s.name}({', '.join(s.tags)})" for s in entry.card.skills
    )
    catalog_lines.append(f"  {name}: {entry.card.description} [skills: {skills_str}]")
catalog = "\n".join(catalog_lines)

TASK_DESCRIPTION = (
    "A developer asked: 'Should I use Python or Go for a new high-throughput API service?' "
    "Produce a concise decision guide."
)

print(f"\nTask: {TASK_DESCRIPTION}")
print(f"\nAvailable agents:\n{catalog}")

# LLM reads the catalog and produces an execution plan
plan_raw = _ask(
    f"You are an orchestrator. Available agents:\n{catalog}\n\n"
    "Given the task, output a JSON list of agent names to call in order. "
    "Example: [\"research-agent\", \"analysis-agent\", \"writing-agent\"]\n"
    "Output ONLY the JSON array, no explanation.",
    f"Task: {TASK_DESCRIPTION}",
)

print(f"\nLLM execution plan: {plan_raw.strip()}")

# Parse and execute the plan
import json, re

match = re.search(r"\[.*?\]", plan_raw, re.DOTALL)
plan: list[str] = json.loads(match.group()) if match else []

print(f"\nExecuting {len(plan)} agents:")
context = TASK_DESCRIPTION
for agent_name in plan:
    if agent_name not in registry._agents:
        print(f"  [skip] unknown agent: {agent_name}")
        continue
    print(f"  → delegating to {agent_name}")
    task = registry.delegate(agent_name, context)
    context = task_output(task)

print(f"\nFinal answer (first 250 chars):\n{context[:250]}...")

# ── comparison summary ────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Protocol positioning comparison:")
print()
rows = [
    ("Problem solved",  "Agent ↔ Tool",              "Agent ↔ Agent"),
    ("Discovery",       "list_tools() from Server",  "discover() from Registry"),
    ("Unit of work",    "Tool call (sync)",          "Task (async-ready)"),
    ("Coupling",        "Agent uses tool directly",  "Orchestrator delegates task"),
    ("Cross-service",   "Tool is a process",         "Agent is a service"),
    ("Who decides?",    "Agent decides (tool use)",  "Orchestrator decides (delegation)"),
]
print(f"{'Dimension':<20} {'MCP':<30} {'A2A'}")
print("-" * 72)
for row in rows:
    print(f"{row[0]:<20} {row[1]:<30} {row[2]}")

print()
print("Protocol selection guide:")
print("  Intra-team, same codebase     → direct function call")
print("  Agent needs external tools    → MCP (tools as service)")
print("  Agent delegates to specialists → A2A (agents as service)")
print("  Large open network of agents  → ANP (decentralized discovery)")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
