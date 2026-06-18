"""
Skill Workflow Chaining Demo

Implements and benchmarks 4 Skill chaining patterns:

  Pattern 1: Sequential Chain    A → B → C
    blog pipeline: keyword-extractor → outline-planner → article-writer

  Pattern 2: Parallel Fan-out    A → {B1, B2, B3} → C
    competitor analysis: split → 3 parallel analyzers → merge report
    Measures: wall-clock time vs sequential equivalent

  Pattern 3: Conditional Routing A → Router → B1 | B2 | B3
    content router: classify request type → route to appropriate writer

  Pattern 4: Feedback Loop       A → Evaluator → [pass] output
                                              ↓
                                          [fail] → A (retry, max 3)
    quality gate: write → quality check → rewrite if score < threshold

Each pattern uses real LLM calls. Pattern 2 logs the speedup ratio.
Pattern 4 logs the number of iterations required.

Run:
    conda activate dev_base
    python demo_skill_workflow.py
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

WIDTH = 70


# ── Skill implementations (LLM-backed) ───────────────────────────────────────

def call_skill(system_prompt: str, user_input: str) -> str:
    return str(llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]).content)


SKILLS = {
    "keyword_extractor": (
        "Extract 3-5 key technical concepts from the topic. "
        "Return as a comma-separated list, nothing else."
    ),
    "outline_planner": (
        "Given keywords, create a 4-section article outline. "
        "Format: one H2 heading per line. Return only the headings."
    ),
    "article_writer": (
        "Write a focused 300-word technical article section. "
        "Use the provided outline. Include one code example."
    ),
    "competitor_product": (
        "Analyze the company's PRODUCT from a competitor perspective. "
        "3 sentences. Focus on features and UX."
    ),
    "competitor_market": (
        "Analyze the company's MARKET POSITION from a competitor perspective. "
        "3 sentences. Focus on positioning and customer segments."
    ),
    "competitor_tech": (
        "Analyze the company's TECHNOLOGY STACK from a competitor perspective. "
        "3 sentences. Focus on infrastructure and engineering choices."
    ),
    "report_merger": (
        "Merge 3 competitor analysis sections (Product, Market, Tech) into "
        "a 200-word executive summary. Identify the top strategic threat."
    ),
    "content_classifier": (
        "Classify the user's writing request into exactly one category. "
        "Reply with only one word: 'technical', 'marketing', or 'general'."
    ),
    "technical_writer": (
        "Write a technical explanation. Be precise, include code if relevant. "
        "200 words max."
    ),
    "marketing_writer": (
        "Write engaging marketing copy. Focus on benefits and CTA. "
        "150 words max."
    ),
    "general_writer": (
        "Write clear, accessible content on the topic. 150 words max."
    ),
    "quality_evaluator": (
        "Evaluate this technical article on a 1-10 scale. "
        "Return JSON only: {\"score\": <int>, \"feedback\": \"<one sentence>\"}. "
        "Score >= 7 means passing quality. Be strict."
    ),
    "quality_writer": (
        "Write a high-quality, well-structured technical article. "
        "Include: clear intro, 2 H2 sections, 1 code block, conclusion. "
        "300 words minimum. Apply the feedback if provided."
    ),
}


# ── Pattern 1: Sequential Chain ───────────────────────────────────────────────

class BlogState(TypedDict):
    topic: str
    keywords: str
    outline: str
    article: str


def kw_node(state: BlogState) -> dict:
    kw = call_skill(SKILLS["keyword_extractor"], state["topic"])
    return {"keywords": kw}


def outline_node(state: BlogState) -> dict:
    outline = call_skill(SKILLS["outline_planner"],
                         f"Topic: {state['topic']}\nKeywords: {state['keywords']}")
    return {"outline": outline}


def write_node(state: BlogState) -> dict:
    article = call_skill(SKILLS["article_writer"],
                         f"Topic: {state['topic']}\nOutline:\n{state['outline']}")
    return {"article": article}


def run_sequential(topic: str) -> tuple[dict, float]:
    g = StateGraph(BlogState)
    g.add_node("keywords", kw_node)
    g.add_node("outline",  outline_node)
    g.add_node("write",    write_node)
    g.set_entry_point("keywords")
    g.add_edge("keywords", "outline")
    g.add_edge("outline",  "write")
    g.add_edge("write",    END)
    app = g.compile()

    t0 = time.time()
    result = app.invoke({"topic": topic, "keywords": "", "outline": "", "article": ""})
    return result, time.time() - t0


# ── Pattern 2: Parallel Fan-out ───────────────────────────────────────────────

def run_parallel(company: str) -> tuple[dict, float, float]:
    """Returns (merged_result, parallel_time, sequential_equivalent)."""
    dimensions = [
        ("product", SKILLS["competitor_product"]),
        ("market",  SKILLS["competitor_market"]),
        ("tech",    SKILLS["competitor_tech"]),
    ]

    # Parallel execution
    results: dict[str, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(call_skill, prompt, f"Analyze company: {company}"): dim
            for dim, prompt in dimensions
        }
        for future in as_completed(futures):
            dim = futures[future]
            results[dim] = future.result()
    parallel_time = time.time() - t0

    # Merge
    merge_input = (
        f"Company: {company}\n\n"
        f"Product analysis:\n{results['product']}\n\n"
        f"Market analysis:\n{results['market']}\n\n"
        f"Tech analysis:\n{results['tech']}"
    )
    merged = call_skill(SKILLS["report_merger"], merge_input)
    total_time = time.time() - t0

    return {"merged": merged, "details": results}, total_time, parallel_time


# ── Pattern 3: Conditional Routing ───────────────────────────────────────────

def run_conditional(request: str) -> tuple[str, str, float]:
    """Returns (category, output, elapsed)."""
    t0 = time.time()

    category = call_skill(SKILLS["content_classifier"], request).strip().lower()
    # Normalize
    if "tech" in category:
        category = "technical"
    elif "market" in category:
        category = "marketing"
    else:
        category = "general"

    skill_key = {
        "technical": "technical_writer",
        "marketing": "marketing_writer",
        "general":   "general_writer",
    }[category]

    output = call_skill(SKILLS[skill_key], request)
    return category, output, time.time() - t0


# ── Pattern 4: Feedback Loop ─────────────────────────────────────────────────

MAX_RETRIES = 3
QUALITY_THRESHOLD = 7  # score out of 10


def parse_eval(raw: str) -> tuple[int, str]:
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        data = json.loads(raw)
        return int(data.get("score", 5)), data.get("feedback", "")
    except (json.JSONDecodeError, ValueError):
        m = re.search(r'"score"\s*:\s*(\d+)', raw)
        score = int(m.group(1)) if m else 5
        fb_m = re.search(r'"feedback"\s*:\s*"([^"]+)"', raw)
        feedback = fb_m.group(1) if fb_m else "no feedback"
        return score, feedback


def run_feedback_loop(topic: str) -> tuple[str, int, list[dict], float]:
    """Returns (final_output, iterations_used, history, elapsed)."""
    t0 = time.time()
    history: list[dict] = []
    feedback = ""
    output = ""
    iteration = 0

    for iteration in range(1, MAX_RETRIES + 1):
        prompt = topic if not feedback else f"{topic}\n\nApply this feedback: {feedback}"
        output = call_skill(SKILLS["quality_writer"], prompt)

        eval_raw = call_skill(SKILLS["quality_evaluator"], output)
        score, feedback = parse_eval(eval_raw)

        history.append({"iteration": iteration, "score": score, "feedback": feedback})

        if score >= QUALITY_THRESHOLD:
            break

    return output, iteration, history, time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Skill Workflow Chaining Demo")
    print("Patterns: Sequential | Parallel | Conditional | Feedback Loop")
    print("=" * WIDTH)

    # Pattern 1: Sequential
    print(f"\n{'─' * WIDTH}")
    print("Pattern 1: Sequential Chain  (keyword → outline → write)")
    print(f"{'─' * WIDTH}")
    topic = "Python async/await: from coroutines to production-ready patterns"
    result, elapsed = run_sequential(topic)
    print(f"  Topic:    {topic}")
    print(f"  Keywords: {result['keywords'][:80]}")
    print(f"  Outline:  {result['outline'][:120].replace(chr(10), ' | ')}")
    print(f"  Article:  {result['article'][:120].replace(chr(10), ' ')}...")
    print(f"  Time: {elapsed:.1f}s (3 sequential LLM calls)")

    # Pattern 2: Parallel
    print(f"\n{'─' * WIDTH}")
    print("Pattern 2: Parallel Fan-out  (3 analyzers → merge)")
    print(f"{'─' * WIDTH}")
    company = "Notion"
    para_result, total_time, fan_time = run_parallel(company)
    print(f"  Company: {company}")
    print(f"  Product:  {para_result['details']['product'][:80]}...")
    print(f"  Market:   {para_result['details']['market'][:80]}...")
    print(f"  Tech:     {para_result['details']['tech'][:80]}...")
    print(f"  Merged:   {para_result['merged'][:120].replace(chr(10), ' ')}...")
    print(f"  Fan-out time: {fan_time:.1f}s  |  Total (incl. merge): {total_time:.1f}s")
    print(f"  Sequential equiv: ~{fan_time * 3:.1f}s  |  Speedup: ~{(fan_time * 3) / total_time:.1f}x")

    # Pattern 3: Conditional
    print(f"\n{'─' * WIDTH}")
    print("Pattern 3: Conditional Routing  (classify → route)")
    print(f"{'─' * WIDTH}")
    routing_tests = [
        "Explain how Kubernetes pod scheduling works with a code example",
        "Write a compelling product description for our new AI writing tool",
        "What is machine learning and why does it matter?",
    ]
    for req in routing_tests:
        category, output, elapsed = run_conditional(req)
        print(f"  Input:  {req[:60]}...")
        print(f"  Route:  {category}  ({elapsed:.1f}s)")
        print(f"  Output: {output[:80].replace(chr(10), ' ')}...")
        print()

    # Pattern 4: Feedback Loop
    print(f"{'─' * WIDTH}")
    print(f"Pattern 4: Feedback Loop  (write → evaluate → retry, max {MAX_RETRIES}, threshold={QUALITY_THRESHOLD}/10)")
    print(f"{'─' * WIDTH}")
    loop_topic = "Write a technical article about Redis Cluster sharding strategy"
    final, iterations, history, elapsed = run_feedback_loop(loop_topic)
    print(f"  Topic: {loop_topic}")
    for h in history:
        status = "✓ PASS" if h["score"] >= QUALITY_THRESHOLD else "✗ fail"
        print(f"    Iteration {h['iteration']}: score={h['score']}/10  {status}")
        print(f"              feedback: {h['feedback'][:80]}")
    print(f"  Final score: {history[-1]['score']}/10  |  Iterations: {iterations}/{MAX_RETRIES}")
    print(f"  Output: {final[:120].replace(chr(10), ' ')}...")
    print(f"  Time: {elapsed:.1f}s")

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
