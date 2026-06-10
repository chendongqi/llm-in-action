"""
Web Agent Demo

A LangGraph-based research agent that browses the web to answer questions.

Tools:
  web_search(query)   — DuckDuckGo search, returns top 5 results
  fetch_page(url)     — Fetch and clean a page, truncated to token budget

Three engineering guards demonstrated:
  1. Token budget   — page content truncated before passing to LLM
  2. Step limit     — MAX_STEPS prevents infinite navigation loops
  3. Error handling — bad / hallucinated URLs return a safe error string

Run:
    conda activate dev_base
    python demo_web_agent.py
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

# ── Engineering constants ─────────────────────────────────────────────────────

PAGE_TOKEN_BUDGET = 800   # max tokens of page text sent to LLM per fetch
MAX_STEPS         = 8     # loop guard: stop after N agent→tool cycles
WIDTH             = 70

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) "
        "Gecko/20100101 Firefox/115.0"
    )
}


# ── Token utilities ───────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Rough estimate: ~3 chars per token for English/Chinese mix."""
    return max(1, len(text) // 3)


def truncate_to_budget(text: str, budget: int = PAGE_TOKEN_BUDGET) -> str:
    if count_tokens(text) <= budget:
        return text
    cutoff = budget * 3
    return text[:cutoff] + f"\n\n[... content truncated to ~{budget}-token budget ...]"


# ── HTML cleaning ─────────────────────────────────────────────────────────────

def clean_html(html: str) -> str:
    """Strip navigation/ads/scripts; return readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    lines = [
        line.strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web with DuckDuckGo.
    Returns up to 5 results, each with title, snippet, and URL.
    Use the URLs from results to call fetch_page — never invent URLs.
    """
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=12,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for i, block in enumerate(soup.select(".result"), 1):
            if i > 5:
                break
            title   = (block.select_one(".result__title")   or soup.new_tag("x")).get_text(strip=True)
            snippet = (block.select_one(".result__snippet") or soup.new_tag("x")).get_text(strip=True)
            url_raw = (block.select_one(".result__url")     or soup.new_tag("x")).get_text(strip=True)
            url = f"https://{url_raw}" if url_raw and not url_raw.startswith("http") else url_raw
            results.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as exc:
        return f"Search error: {exc}"


@tool
def fetch_page(url: str) -> str:
    """
    Fetch a web page and return its cleaned text (truncated to token budget).
    Only call with real URLs obtained from web_search results.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        full_text = clean_html(resp.text)
        orig_tokens = count_tokens(full_text)
        displayed = truncate_to_budget(full_text)
        shown_tokens = min(orig_tokens, PAGE_TOKEN_BUDGET)
        return (
            f"[URL: {url}]\n"
            f"[Size: {orig_tokens} tokens → showing {shown_tokens} tokens "
            f"(budget={PAGE_TOKEN_BUDGET})]\n\n"
            f"{displayed}"
        )
    except requests.HTTPError as exc:
        return f"HTTP {exc.response.status_code} — could not fetch {url}"
    except requests.ConnectionError:
        return f"Connection error — {url} may not exist or be unreachable"
    except Exception as exc:
        return f"Error fetching {url}: {type(exc).__name__}: {exc}"


# ── Agent graph ───────────────────────────────────────────────────────────────

TOOLS   = [web_search, fetch_page]
TOOL_MAP = {t.name: t for t in TOOLS}
bound_llm = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = f"""You are a web research agent. Answer the user's question by browsing the web.

Workflow:
1. Call web_search to find relevant pages.
2. Call fetch_page on promising URLs to read content.
3. If you find the answer, give a clear, concise final response.
4. If a page doesn't help, try a different search query.

Strict rules:
- Only use URLs from web_search results — never invent or guess URLs.
- If fetch_page returns an error, try a different URL or search query.
- You have at most {MAX_STEPS} total steps. Be efficient.
- Once you have enough information, stop browsing and answer directly."""


class WState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


def agent_node(state: WState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = bound_llm.invoke(msgs)
    return {"messages": [response], "steps": state["steps"] + 1}


def tools_node(state: WState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:  # type: ignore[union-attr]
        output = TOOL_MAP[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return {"messages": results}


def router(state: WState) -> str:
    if state["steps"] >= MAX_STEPS:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    g = StateGraph(WState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


graph = build_graph()


# ── Research runner ───────────────────────────────────────────────────────────

def run_research(label: str, query: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"[{label}]")
    print(f"Q: {query}")
    print(f"{'─' * WIDTH}")

    t0 = time.time()
    state = graph.invoke(
        {"messages": [HumanMessage(content=query)], "steps": 0},
        config={"recursion_limit": MAX_STEPS * 3},
    )
    elapsed = time.time() - t0

    # Print execution trace
    tool_calls_total = 0
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = str(tc["args"])[:72]
                print(f"  → {tc['name']}({args_str})")
                tool_calls_total += 1
        elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            print(f"\n  Answer: {msg.content}")

    steps = state["steps"]
    print(f"\n  Steps: {steps}/{MAX_STEPS}  |  Tool calls: {tool_calls_total}  |  Time: {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Web Agent Demo")
    print(f"Model: glm-4-flash  |  Token budget/page: {PAGE_TOKEN_BUDGET}  |  Max steps: {MAX_STEPS}")
    print("=" * WIDTH)

    # ── Part 1: Single-hop factual lookup ─────────────────────────────────────
    print("\n\n=== Part 1: Single-hop Research ===")

    run_research(
        "Python latest version",
        "What is the latest stable release version of Python, and when was it released?",
    )

    run_research(
        "LangGraph latest version",
        "What is the latest published version of the langgraph Python package on PyPI?",
    )

    # ── Part 2: Multi-hop research ────────────────────────────────────────────
    print("\n\n=== Part 2: Multi-hop Research ===")

    run_research(
        "Multi-hop: find info requiring 2+ pages",
        "Who created LangGraph and what company is behind it? "
        "Also find the GitHub repository URL.",
    )

    # ── Part 3: Engineering guards ────────────────────────────────────────────
    print("\n\n=== Part 3: Engineering Guards ===")

    # Guard 1: URL hallucination — fetch_page handles bad URLs safely
    print(f"\n{'─' * WIDTH}")
    print("[Guard 1] URL error handling (bad / hallucinated URL)")
    bad_url = "https://totally-made-up-domain-xyz99999.org/docs/nonexistent"
    result = fetch_page.invoke({"url": bad_url})
    print(f"  fetch_page({bad_url[:50]}...)")
    print(f"  → {result}")

    # Guard 2: Token budget — show truncation in action
    print(f"\n{'─' * WIDTH}")
    print(f"[Guard 2] Token budget enforcement (budget={PAGE_TOKEN_BUDGET} tokens/page)")
    result2 = fetch_page.invoke({"url": "https://pypi.org/project/langgraph/"})
    header_line = result2.split("\n")[1] if "\n" in result2 else result2[:120]
    print(f"  fetch_page(pypi.org/project/langgraph/)")
    print(f"  → {header_line}")

    # Guard 3: Step limit — show that graph terminates at MAX_STEPS
    print(f"\n{'─' * WIDTH}")
    print(f"[Guard 3] Step limit (MAX_STEPS={MAX_STEPS}) — agent cannot loop forever")
    print(f"  Graph router returns END when state['steps'] >= {MAX_STEPS}")
    print(f"  Even if tool_calls remain, execution stops.")

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
