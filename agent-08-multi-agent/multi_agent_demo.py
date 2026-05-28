"""
Multi-Agent Architecture Design Patterns Demo

Three patterns:
  Demo 1 — Supervisor pattern: LLM classifies task + deterministic routing
  Demo 2 — Pipeline pattern: fixed sequential agent chain
  Demo 3 — Supervisor adaptability: same graph, different path per task type

Run:
    conda activate dev_base
    python multi_agent_demo.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# ── LLM init ──────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)


def _ask(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(system), HumanMessage(user)])
    return str(resp.content)


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Supervisor Pattern
# ══════════════════════════════════════════════════════════════════════════════
#
# Topology:
#   classify → supervisor → [researcher | writer | reviewer | FINISH]
#                                ↓           ↓         ↓
#                            supervisor ← ← ← ← ← ←
#
# Two-phase design:
#   Phase 1: LLM classifies task type once (simple_fact vs full_article)
#   Phase 2: Python routing based on task_type + called list (deterministic)
#
# This hybrid keeps LLM as the "decision maker" while eliminating routing loops.

print("\n" + "=" * 70)
print("Demo 1: Supervisor Pattern")
print("=" * 70)

WORKERS = ["researcher", "writer", "reviewer"]


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    task_type: str   # "simple_fact" or "full_article"
    called: list[str]
    next: str


def classify_node(state: SupervisorState) -> SupervisorState:
    """LLM decides task type once — sets routing strategy for all subsequent steps."""
    decision = _ask(
        "Classify this task into exactly one category:\n"
        "  simple_fact  — a factual question with a direct short answer\n"
        "  full_article — needs research, writing, and editorial review\n"
        "Output one word only: simple_fact / full_article",
        f"Task: {state['task']}",
    ).strip().lower()

    task_type = "full_article" if "full_article" in decision else "simple_fact"
    print(f"  [classify] task_type = {task_type}")
    return {**state, "task_type": task_type}


def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Deterministic Python routing — no LLM call, no risk of loops."""
    called = state["called"]
    task_type = state["task_type"]

    if "researcher" not in called:
        next_worker = "researcher"
    elif task_type == "simple_fact":
        next_worker = "FINISH"          # simple questions stop after research
    elif "writer" not in called:
        next_worker = "writer"
    elif "reviewer" not in called:
        next_worker = "reviewer"
    else:
        next_worker = "FINISH"

    print(f"  [supervisor] → {next_worker}")
    return {**state, "next": next_worker}


def researcher_node(state: SupervisorState) -> SupervisorState:
    print("  [researcher] working...")
    notes = _ask(
        "You are a research assistant. Gather 3–5 concise key facts on the topic.",
        f"Topic: {state['task']}",
    )
    msg = AIMessage(content=f"[Researcher] {notes}")
    return {**state, "messages": [msg], "called": state["called"] + ["researcher"]}


def writer_node(state: SupervisorState) -> SupervisorState:
    print("  [writer] working...")
    research = next(
        (str(m.content) for m in reversed(state["messages"]) if "[Researcher]" in str(m.content)),
        "No research available.",
    )
    draft = _ask(
        "You are a technical writer. Write a concise 150-word article based on the research notes.",
        f"Research notes:\n{research}",
    )
    msg = AIMessage(content=f"[Writer] {draft}")
    return {**state, "messages": [msg], "called": state["called"] + ["writer"]}


def reviewer_node(state: SupervisorState) -> SupervisorState:
    print("  [reviewer] working...")
    draft = next(
        (str(m.content) for m in reversed(state["messages"]) if "[Writer]" in str(m.content)),
        "No draft available.",
    )
    review = _ask(
        "You are a technical reviewer. Give 2–3 specific improvement suggestions.",
        f"Draft:\n{draft}",
    )
    msg = AIMessage(content=f"[Reviewer] {review}")
    return {**state, "messages": [msg], "called": state["called"] + ["reviewer"]}


def route_supervisor(state: SupervisorState) -> str:
    return state["next"]


def build_supervisor_graph() -> CompiledStateGraph:
    g = StateGraph(SupervisorState)
    g.add_node("classify", classify_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END,
        },
    )
    g.add_edge("researcher", "supervisor")
    g.add_edge("writer", "supervisor")
    g.add_edge("reviewer", "supervisor")
    return g.compile()


supervisor_app = build_supervisor_graph()

ARTICLE_TASK = "Write a short article about Python list comprehensions"
print(f"\nTask: {ARTICLE_TASK}")
print("-" * 50)

article_result = supervisor_app.invoke(
    {
        "messages": [HumanMessage(content=ARTICLE_TASK)],
        "task": ARTICLE_TASK,
        "task_type": "",
        "called": [],
        "next": "",
    }
)

print("\nExecution trace summary:")
for msg in article_result["messages"]:
    if isinstance(msg, AIMessage):
        content = str(msg.content)
        role_prefix = content.split("]")[0] + "]" if "]" in content else "[AI]"
        preview = content[len(role_prefix):].strip()[:80].replace("\n", " ")
        print(f"  {role_prefix} {preview}...")

print(f"\nWorkers called: {article_result['called']}")
print(f"Task type     : {article_result['task_type']}")
print("Result        : researcher → writer → reviewer → FINISH")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — Pipeline Pattern
# ══════════════════════════════════════════════════════════════════════════════
#
# Topology (fixed, linear):
#   outline_agent → draft_agent → polish_agent → END
#
# No routing decisions needed. Each stage passes its output to the next.

print("\n" + "=" * 70)
print("Demo 2: Pipeline Pattern")
print("=" * 70)


class PipelineState(TypedDict):
    topic: str
    outline: str
    draft: str
    polished: str
    stage_log: list[str]


def outline_agent(state: PipelineState) -> PipelineState:
    print("  [outline_agent] working...")
    outline = _ask(
        "You are an outline specialist. Create a 5-point article outline (numbered list).",
        f"Topic: {state['topic']}",
    )
    log = state["stage_log"] + [f"[outline_agent] {len(outline)} chars"]
    return {**state, "outline": outline, "stage_log": log}


def draft_agent(state: PipelineState) -> PipelineState:
    print("  [draft_agent] working...")
    draft = _ask(
        "You are a writer. Expand the outline into a 200-word article draft.",
        f"Outline:\n{state['outline']}",
    )
    log = state["stage_log"] + [f"[draft_agent] {len(draft)} chars"]
    return {**state, "draft": draft, "stage_log": log}


def polish_agent(state: PipelineState) -> PipelineState:
    print("  [polish_agent] working...")
    polished = _ask(
        "You are an editor. Polish the draft: improve flow, fix grammar, make it engaging.",
        f"Draft:\n{state['draft']}",
    )
    log = state["stage_log"] + [f"[polish_agent] {len(polished)} chars"]
    return {**state, "polished": polished, "stage_log": log}


pipeline_graph = StateGraph(PipelineState)
pipeline_graph.add_node("outline_agent", outline_agent)
pipeline_graph.add_node("draft_agent", draft_agent)
pipeline_graph.add_node("polish_agent", polish_agent)

pipeline_graph.set_entry_point("outline_agent")
pipeline_graph.add_edge("outline_agent", "draft_agent")
pipeline_graph.add_edge("draft_agent", "polish_agent")
pipeline_graph.add_edge("polish_agent", END)

pipeline_app = pipeline_graph.compile()

print(f"\nTask: {ARTICLE_TASK}")
print("-" * 50)

pipeline_result = pipeline_app.invoke(
    {
        "topic": ARTICLE_TASK,
        "outline": "",
        "draft": "",
        "polished": "",
        "stage_log": [],
    }
)

print("\nExecution trace:")
for log_entry in pipeline_result["stage_log"]:
    print(f"  {log_entry}")

print("\nFinal polished article (first 300 chars):")
print(pipeline_result["polished"][:300] + "...")
print("\nPipeline completed in 3 fixed stages: outline → draft → polish")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — Supervisor Adaptability
# ══════════════════════════════════════════════════════════════════════════════
#
# Same graph, different execution path based on task classification.
# Simple factual question → only researcher is called (writer + reviewer skipped).
# This is the key advantage of the Supervisor pattern over Pipeline.

print("\n" + "=" * 70)
print("Demo 3: Supervisor Adaptability — Same Graph, Different Path")
print("=" * 70)

SIMPLE_TASK = "What year was Python created?"
print(f"\nTask: {SIMPLE_TASK}")
print("-" * 50)

simple_result = supervisor_app.invoke(
    {
        "messages": [HumanMessage(content=SIMPLE_TASK)],
        "task": SIMPLE_TASK,
        "task_type": "",
        "called": [],
        "next": "",
    }
)

print(f"\nWorkers called : {simple_result['called']}")
print(f"Task type      : {simple_result['task_type']}")
print("Result         : researcher → FINISH  (writer + reviewer skipped)")

# ── Comparison summary ────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print(f"{'Task':<40} {'Workers Called':<28} {'Steps'}")
print("-" * 72)
print(f"{'Write article (full_article)':<40} {'researcher→writer→reviewer':<28} 3")
print(f"{'Factual question (simple_fact)':<40} {'researcher':<28} 1")
print("\nSame supervisor graph, different paths based on LLM classification.")

print("\n" + "-" * 60)
print("Pattern decision matrix:")
print()
rows = [
    ("Execution path",   "Fixed, hardwired",       "Dynamic, classification-driven"),
    ("Best for",         "ETL, doc processing",    "Research, open Q&A, mixed tasks"),
    ("Debuggability",    "High (linear trace)",    "Medium (path varies per task)"),
    ("LLM calls/turn",  "N (one per stage)",      "N + 1 (one classify call extra)"),
    ("Flexibility",      "Low",                    "High"),
    ("Predictability",   "High",                   "Lower"),
    ("Implementation",   "Trivial",                "Medium"),
]
print(f"{'Dimension':<22} {'Pipeline':<26} {'Supervisor'}")
print("-" * 70)
for row in rows:
    print(f"{row[0]:<22} {row[1]:<26} {row[2]}")

print()
print("Rule of thumb:")
print("  Know exactly what steps you need → Pipeline")
print("  Need to adapt the steps per task  → Supervisor")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
