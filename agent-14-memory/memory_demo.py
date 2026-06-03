"""
Agent Memory Demo

Three memory patterns:
  Demo 1 — Short-term:   LangGraph MemorySaver keeps context within a session thread
  Demo 2 — Long-term:    User-specific facts persist across sessions (key-value store)
  Demo 3 — Compression:  Summarize history when token count exceeds threshold

Run:
    conda activate dev_base
    python memory_demo.py
"""

import json
import os
import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── Tools ──────────────────────────────────────────────────────────────────────

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


def get_final_answer(output: dict) -> str:
    for m in reversed(output["messages"]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            return str(m.content)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Demo 1 — Short-term Memory: MemorySaver checkpointer
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 1: Short-term Memory — MemorySaver (within-session context)")
print("=" * 70)

checkpointer = MemorySaver()
stateful_agent = create_react_agent(
    model=llm,
    tools=[get_weather, calculator, get_product_info],
    checkpointer=checkpointer,
)

THREAD_A: RunnableConfig = {"configurable": {"thread_id": "thread-alice"}}
THREAD_B: RunnableConfig = {"configurable": {"thread_id": "thread-bob"}}

print("\n── Turn 1 (Thread A): introduce name and city ──")
r1 = stateful_agent.invoke(
    {"messages": [HumanMessage("Hi, I'm Alice. I live in Beijing.")]},
    config=THREAD_A,
)
print(f"Agent: {get_final_answer(r1)[:150]}")

print("\n── Turn 2 (Thread A): ask about local weather (expects agent to remember Beijing) ──")
r2 = stateful_agent.invoke(
    {"messages": [HumanMessage("What's the weather like where I live today?")]},
    config=THREAD_A,
)
ans2 = get_final_answer(r2)
tools2 = [tc["name"] for m in r2["messages"]
          if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
print(f"Agent: {ans2[:200]}")
print(f"Tools used: {tools2}")

print("\n── Same question in Thread B (no prior context) ──")
r3 = stateful_agent.invoke(
    {"messages": [HumanMessage("What's the weather like where I live today?")]},
    config=THREAD_B,
)
ans3 = get_final_answer(r3)
tools3 = [tc["name"] for m in r3["messages"]
          if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
print(f"Agent: {ans3[:200]}")
print(f"Tools used: {tools3}")

print(f"\nThread A: remembered city → called {tools2}")
print(f"Thread B: no context → {tools3 if tools3 else 'asked for clarification'}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 2 — Long-term Memory: cross-session key-value store
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 2: Long-term Memory — cross-session user fact store")
print("=" * 70)

# Simulated persistent store (production: replace with database or vector store)
LONG_TERM_STORE: dict[str, dict[str, str]] = {}


def save_user_facts(user_id: str, facts: dict[str, str]) -> None:
    if user_id not in LONG_TERM_STORE:
        LONG_TERM_STORE[user_id] = {}
    LONG_TERM_STORE[user_id].update(facts)


def load_user_facts(user_id: str) -> dict[str, str]:
    return LONG_TERM_STORE.get(user_id, {})


def extract_facts(conversation: str) -> dict[str, str]:
    """Ask the LLM to extract key-value facts from a conversation snippet."""
    resp = llm.invoke([
        SystemMessage(
            "Extract key facts about the user from this conversation. "
            "Return ONLY a JSON object with short lowercase keys. "
            'Example: {"city": "Shanghai", "plan": "wonderbot pro"}\n'
            "Output the JSON object only, no explanation."
        ),
        HumanMessage(f"Conversation:\n{conversation}"),
    ])
    text = str(resp.content).strip()
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


USER_ID = "user-alice"
base_agent = create_react_agent(model=llm, tools=[get_weather, calculator, get_product_info])

print(f"\n── Session 1: user mentions preferences ──")
session1_turns = [
    "I'm Alice. I'm based in Shanghai and my team uses WonderBot Pro.",
    "We mainly use the API for data processing — about 50,000 calls a month.",
]

session1_log = []
for msg in session1_turns:
    print(f"User: {msg}")
    result = base_agent.invoke({"messages": [HumanMessage(msg)]})
    ans = get_final_answer(result)
    print(f"Agent: {ans[:100]}")
    session1_log.append(f"User: {msg}")
    session1_log.append(f"Agent: {ans[:100]}")

facts = extract_facts("\n".join(session1_log))
save_user_facts(USER_ID, facts)
print(f"\nExtracted and saved to long-term store: {facts}")

print(f"\n── Session 2: new conversation, long-term memory injected ──")
stored = load_user_facts(USER_ID)
facts_text = "; ".join(f"{k}={v}" for k, v in stored.items())
personalized_prompt = (
    "You are a helpful assistant. "
    f"Known facts about this user: {facts_text}. "
    "Use these facts to personalize your responses without asking the user to repeat themselves."
)

personalized_agent = create_react_agent(
    model=llm,
    tools=[get_weather, calculator, get_product_info],
    prompt=personalized_prompt,
)

q_s2 = "What's the weather like in my city today?"
print(f"User: {q_s2}")
r_s2 = personalized_agent.invoke({"messages": [HumanMessage(q_s2)]})
ans_s2 = get_final_answer(r_s2)
tools_s2 = [tc["name"] for m in r_s2["messages"]
            if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
print(f"Agent: {ans_s2[:200]}")
print(f"Tools used: {tools_s2}")
print(f"\nAgent used stored city='{stored.get('city', '?')}' without user repeating it.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 3 — History Compression
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Demo 3: History Compression — summarize when context grows too long")
print("=" * 70)

COMPRESSION_THRESHOLD = 250   # tokens; low value to trigger compression in demo


def summarize_messages(messages: list) -> str:
    """Compress a message list into a compact summary."""
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {str(m.content)[:150]}"
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    )
    resp = llm.invoke([
        SystemMessage(
            "Summarize the following conversation in 2-3 sentences. "
            "Preserve all key facts: names, cities, numbers, product names, preferences."
        ),
        HumanMessage(f"Conversation:\n{history_text}"),
    ])
    return str(resp.content)


turns = [
    "Hi, I'm Bob. My startup is in Shenzhen and we're evaluating AI API tools.",
    "We're looking at WonderBot Pro. How much does it cost?",
    "We have 8 developers. Will 100,000 API calls a month be enough?",
    "What's the weather in Shenzhen today? We have an outdoor team event.",
    "What would it cost us annually? Use 299 * 12.",
]

messages: list = []

print(f"\nBuilding conversation across {len(turns)} turns (threshold = {COMPRESSION_THRESHOLD} tokens):\n")

for i, turn in enumerate(turns, 1):
    messages.append(HumanMessage(turn))
    result = base_agent.invoke({"messages": messages})
    ai_ans = get_final_answer(result)
    messages.append(AIMessage(ai_ans))

    total_tokens = sum(count_tokens(str(m.content)) for m in messages)
    print(f"  Turn {i}: {len(messages)} messages | ~{total_tokens} tokens")

    if total_tokens > COMPRESSION_THRESHOLD and i < len(turns):
        print(f"\n  ── token threshold exceeded: compressing history ──")
        summary = summarize_messages(messages)
        summary_tokens = count_tokens(summary)
        tokens_before = total_tokens
        messages = [SystemMessage(f"Conversation summary so far: {summary}")]
        print(f"  Before: ~{tokens_before} tokens")
        print(f"  After : ~{summary_tokens} tokens  (compressed {tokens_before - summary_tokens} tokens)")
        print(f"  Summary: {summary[:200]}")
        print()

print(f"\nFinal history: {len(messages)} messages | ~{sum(count_tokens(str(m.content)) for m in messages)} tokens")

print(f"\n── Verify: agent recalls key facts after compression ──")
messages.append(HumanMessage("Quickly summarize: who am I, what city, and what's the annual API cost?"))
result_final = base_agent.invoke({"messages": messages})
final_ans = get_final_answer(result_final)
print(f"Agent: {final_ans[:300]}")


# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Memory Pattern Summary")
print("=" * 70)
print()
print(f"{'Pattern':<18} {'Scope':<20} {'Storage':<22} {'Use case'}")
print("-" * 75)
print(f"{'Short-term':<18} {'Single session':<20} {'In-memory checkpointer':<22} {'Multi-turn Q&A'}")
print(f"{'Long-term':<18} {'Cross-session':<20} {'DB / vector store':<22} {'Personalization'}")
print(f"{'Compression':<18} {'Within session':<20} {'Summarized history':<22} {'Long conversations'}")
print()
print("Production recommendation: combine all three.")
print("  MemorySaver  →  per-thread short-term context")
print("  Key-value DB →  persistent user facts across sessions")
print("  Summarizer   →  token guard when conversation grows long")

print("\n" + "=" * 70)
print("All demos complete.")
print("=" * 70)
