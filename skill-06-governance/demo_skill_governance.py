"""
Skill Governance Demo

Three governance mechanisms demonstrated with real data:

  Part 1: Embedding-based Skill Routing
    Build a 6-Skill registry, embed descriptions, route 8 test queries
    via cosine similarity, report routing accuracy + confidence scores.

  Part 2: Prompt Compression Impact
    Run the same task against a verbose Skill prompt (328 tokens) and a
    compressed version (~180 tokens). Measure: token reduction %, quality
    delta (LLM-as-Judge), and whether compression degrades output.

  Part 3: Cost Report
    Run 4 Skills on representative tasks, compute cost-per-call estimates
    and project monthly cost at different usage volumes.

Run:
    conda activate dev_base
    python demo_skill_governance.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.0,
)

embedder = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    openai_api_key=os.environ.get("EMBEDDING_API_KEY", ""),  # type: ignore[arg-type]
    openai_api_base=os.environ.get("EMBEDDING_API_BASE", ""),  # type: ignore[call-arg]
    check_embedding_ctx_length=False,
)

WIDTH = 70


# ── Skill Registry ────────────────────────────────────────────────────────────

REGISTRY = [
    {
        "id": "rnd-technical-writer",
        "description": "Write technical blog articles and tutorials for software engineers. "
                       "Use when: user asks to write a tech article, tutorial, or explainer.",
        "domain": "content",
        "prompt": "You are an expert technical writer. Write a Markdown technical article.",
    },
    {
        "id": "competitor-analyzer",
        "description": "Analyze a competitor company across product, market, and technology dimensions. "
                       "Use when: user asks to analyze or compare a company.",
        "domain": "strategy",
        "prompt": "You are a competitive intelligence analyst. Produce a structured competitor analysis.",
    },
    {
        "id": "bug-root-cause",
        "description": "Diagnose software bugs and identify root causes from stack traces and logs. "
                       "Use when: user shares an error message, stack trace, or bug description.",
        "domain": "engineering",
        "prompt": "You are a senior software engineer. Diagnose the bug and suggest fixes.",
    },
    {
        "id": "meeting-summarizer",
        "description": "Summarize meeting transcripts and extract action items with owners. "
                       "Use when: user provides meeting notes or a conversation log to summarize.",
        "domain": "productivity",
        "prompt": "You are a meeting assistant. Summarize key decisions and list action items.",
    },
    {
        "id": "sql-query-builder",
        "description": "Generate SQL queries from natural language descriptions. "
                       "Use when: user describes a data retrieval need in plain language.",
        "domain": "engineering",
        "prompt": "You are a database expert. Write an efficient SQL query for the user's need.",
    },
    {
        "id": "marketing-copywriter",
        "description": "Write marketing copy, product descriptions, and promotional content. "
                       "Use when: user needs compelling copy for a product, feature, or campaign.",
        "domain": "marketing",
        "prompt": "You are an expert copywriter. Write engaging marketing copy.",
    },
]

# Test queries with expected Skill id
ROUTING_TESTS = [
    {"query": "Write a deep-dive article about Kubernetes pod scheduling",
     "expected": "rnd-technical-writer"},
    {"query": "我们的主要竞争对手 Notion 最近有什么动态",
     "expected": "competitor-analyzer"},
    {"query": "Traceback: AttributeError: 'NoneType' object has no attribute 'split'",
     "expected": "bug-root-cause"},
    {"query": "帮我整理一下今天产品评审会的要点和后续任务",
     "expected": "meeting-summarizer"},
    {"query": "Get all orders placed in the last 7 days with customer name and total amount",
     "expected": "sql-query-builder"},
    {"query": "Write a product description for our new AI-powered code review tool",
     "expected": "marketing-copywriter"},
    # Edge / ambiguous cases
    {"query": "分析一下 Python 3.12 的性能改进",
     "expected": "rnd-technical-writer"},
    {"query": "List all users who haven't logged in for 30 days",
     "expected": "sql-query-builder"},
]


# ── Part 1: Embedding routing ─────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def route_by_embedding(query: str, skill_embeddings: list[tuple[str, list[float]]]) -> tuple[str, float]:
    q_emb = embedder.embed_query(query)
    best_id, best_score = "", -1.0
    for skill_id, emb in skill_embeddings:
        score = cosine_similarity(q_emb, emb)
        if score > best_score:
            best_score = score
            best_id = skill_id
    return best_id, best_score


def run_routing_demo() -> None:
    print(f"\n{'─' * WIDTH}")
    print("Part 1: Embedding-based Skill Routing")
    print(f"Registry: {len(REGISTRY)} Skills  |  Test queries: {len(ROUTING_TESTS)}")
    print(f"{'─' * WIDTH}")

    # Embed all Skill descriptions
    print("  Building registry embeddings...")
    skill_embeddings = []
    for skill in REGISTRY:
        emb = embedder.embed_query(skill["description"])
        skill_embeddings.append((skill["id"], emb))
    print(f"  Done ({len(skill_embeddings)} embeddings)")

    correct = 0
    print(f"\n  {'Query':<45} {'Routed to':<25} {'Score':>6}  {'OK?'}")
    print(f"  {'─'*45} {'─'*25} {'─'*6}  {'─'*4}")

    for test in ROUTING_TESTS:
        routed_id, score = route_by_embedding(test["query"], skill_embeddings)
        ok = "✓" if routed_id == test["expected"] else "✗"
        if routed_id == test["expected"]:
            correct += 1
        print(f"  {test['query'][:44]:<44}  {routed_id:<25} {score:.3f}  {ok}")

    accuracy = correct / len(ROUTING_TESTS)
    print(f"\n  Routing accuracy: {correct}/{len(ROUTING_TESTS)} = {accuracy:.0%}")


# ── Part 2: Prompt compression ────────────────────────────────────────────────

VERBOSE_PROMPT = """You are a highly experienced senior technical writer with over 10 years of expertise
in software engineering documentation, developer education, and technical communication.
Your primary responsibility is to create comprehensive, accurate, and engaging technical articles
for software engineers and developers.

When writing a technical article, you should:
1. Start with a clear and compelling introduction that explains why this topic matters
2. Break down complex concepts into digestible sections with clear headings
3. Include practical code examples that readers can actually run and test
4. Explain not just HOW things work but also WHY they work that way
5. Add helpful notes, warnings, or tips where appropriate
6. Conclude with a summary of key takeaways

Your writing style should be:
- Professional yet approachable and conversational
- Technically accurate without being overly academic
- Clear and concise, avoiding unnecessary jargon
- Supportive of readers at different skill levels

Format requirements:
- Use YAML frontmatter with title, description, and tags
- Use H2 headings for main sections
- Include at least one code block with proper syntax highlighting
- Keep total length between 400-600 words
- End with a brief summary section"""

COMPRESSED_PROMPT = """You are an expert technical writer for software engineers.

Write a Markdown technical article:
- YAML frontmatter (title, description, tags)
- H2 sections covering core concept, implementation, and practical tips
- At least 1 code block with syntax highlighting
- 400-600 words
- End with a summary"""

COMPRESSION_TASKS = [
    "Write a technical article about Python context managers and the `with` statement",
    "Write a technical article about Redis pub/sub messaging patterns",
]

JUDGE_PROMPT = """Evaluate this AI-generated technical article on technical accuracy, depth, clarity,
and practical value. Each dimension 1–5. Return JSON only (no fences):
{{"technical_accuracy":<1-5>,"depth":<1-5>,"clarity":<1-5>,"practical_value":<1-5>}}"""

WEIGHTS = {"technical_accuracy": 0.35, "depth": 0.25, "clarity": 0.20, "practical_value": 0.20}


def count_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def judge(article: str) -> float:
    raw = str(llm.invoke([HumanMessage(content=JUDGE_PROMPT + f"\n\nArticle:\n{article[:2000]}")]).content)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        s = json.loads(raw)
    except json.JSONDecodeError:
        return 3.0
    return sum(s.get(k, 3) * w for k, w in WEIGHTS.items())


def run_compression_demo() -> None:
    print(f"\n{'─' * WIDTH}")
    print("Part 2: Prompt Compression Impact")
    verbose_tokens = count_tokens(VERBOSE_PROMPT)
    compressed_tokens = count_tokens(COMPRESSED_PROMPT)
    reduction = 1 - compressed_tokens / verbose_tokens
    print(f"Verbose: ~{verbose_tokens} tokens  |  Compressed: ~{compressed_tokens} tokens  |  Reduction: {reduction:.0%}")
    print(f"{'─' * WIDTH}")

    results = []
    for task in COMPRESSION_TASKS:
        print(f"\n  Task: {task[:60]}...")

        t0 = time.time()
        v_out = str(llm.invoke([SystemMessage(content=VERBOSE_PROMPT), HumanMessage(content=task)]).content)
        v_time = time.time() - t0
        v_score = judge(v_out)

        t0 = time.time()
        c_out = str(llm.invoke([SystemMessage(content=COMPRESSED_PROMPT), HumanMessage(content=task)]).content)
        c_time = time.time() - t0
        c_score = judge(c_out)

        delta = c_score - v_score
        results.append({"delta": delta, "v_score": v_score, "c_score": c_score})

        print(f"    Verbose    → score={v_score:.2f}/5  time={v_time:.1f}s")
        print(f"    Compressed → score={c_score:.2f}/5  time={c_time:.1f}s")
        print(f"    Quality delta: {delta:+.2f}  ({'no significant change' if abs(delta) < 0.15 else 'notable change'})")

    avg_delta = sum(r["delta"] for r in results) / len(results)
    print(f"\n  Avg quality delta: {avg_delta:+.2f}")
    print(f"  Prompt token reduction: {reduction:.0%}  |  Quality impact: {avg_delta:+.2f}/5")


# ── Part 3: Cost report ───────────────────────────────────────────────────────

COST_SKILLS = [
    {
        "id": "rnd-technical-writer",
        "prompt": COMPRESSED_PROMPT,
        "tasks": [
            "Write a technical article about async/await patterns",
            "Write a technical article about Docker networking",
        ],
    },
    {
        "id": "competitor-analyzer",
        "prompt": "You are a competitive intelligence analyst. Produce a 200-word structured analysis.",
        "tasks": [
            "Analyze Notion as a competitor",
            "Analyze Linear as a competitor",
        ],
    },
    {
        "id": "meeting-summarizer",
        "prompt": "Summarize key decisions and list action items with owners.",
        "tasks": [
            "Meeting notes: Team discussed Q3 roadmap. Decision: ship auth by Aug. Owner: Alice. Delay search to Q4.",
        ],
    },
    {
        "id": "sql-query-builder",
        "prompt": "You are a database expert. Write an efficient SQL query.",
        "tasks": [
            "Get all users who signed up in the last 30 days and have made at least 2 purchases",
        ],
    },
]

# Approximate cost per token (USD) — using glm-4-flash as proxy
COST_PER_TOKEN = 0.000001  # $1 / 1M tokens (input)


@dataclass
class SkillCostRecord:
    skill_id: str
    calls: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency: float
    cost_per_call: float


def run_cost_demo() -> None:
    print(f"\n{'─' * WIDTH}")
    print("Part 3: Cost Report")
    print(f"{'─' * WIDTH}")

    records: list[SkillCostRecord] = []

    for skill_cfg in COST_SKILLS:
        input_tokens_list = []
        output_tokens_list = []
        latencies = []

        for task in skill_cfg["tasks"]:
            full_input = skill_cfg["prompt"] + task
            t0 = time.time()
            output = str(llm.invoke([
                SystemMessage(content=skill_cfg["prompt"]),
                HumanMessage(content=task),
            ]).content)
            latencies.append(time.time() - t0)
            input_tokens_list.append(count_tokens(full_input))
            output_tokens_list.append(count_tokens(output))

        avg_in = sum(input_tokens_list) / len(input_tokens_list)
        avg_out = sum(output_tokens_list) / len(output_tokens_list)
        cost_per_call = (avg_in + avg_out) * COST_PER_TOKEN
        avg_lat = sum(latencies) / len(latencies)

        records.append(SkillCostRecord(
            skill_id=skill_cfg["id"],
            calls=len(skill_cfg["tasks"]),
            avg_input_tokens=avg_in,
            avg_output_tokens=avg_out,
            avg_latency=avg_lat,
            cost_per_call=cost_per_call,
        ))
        print(f"  [{skill_cfg['id']}]  in={avg_in:.0f}t  out={avg_out:.0f}t  "
              f"${cost_per_call*100:.4f}/100calls  p50={avg_lat:.1f}s")

    # Monthly projection
    MONTHLY_VOLUME = {"rnd-technical-writer": 200, "competitor-analyzer": 50,
                      "meeting-summarizer": 300, "sql-query-builder": 500}

    print(f"\n  Monthly cost projection (estimated volumes):")
    total_monthly = 0.0
    for r in records:
        vol = MONTHLY_VOLUME.get(r.skill_id, 100)
        monthly = r.cost_per_call * vol
        total_monthly += monthly
        print(f"    {r.skill_id:<28} {vol:>5} calls/mo  ${monthly:.4f}")
    print(f"    {'─'*50}")
    print(f"    {'Total':<28} {'':>5}           ${total_monthly:.4f}/mo")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Skill Governance Demo")
    print("Routing | Prompt Compression | Cost Report")
    print("=" * WIDTH)

    run_routing_demo()
    run_compression_demo()
    run_cost_demo()

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
