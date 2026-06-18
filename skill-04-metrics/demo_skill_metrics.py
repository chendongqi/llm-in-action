"""
Skill Metrics Demo

Demonstrates the L1/L2/L3 three-layer metrics framework for AI Skills.

Layer definitions:
  L3 — System Health:   latency, token cost, availability, error rate
  L2 — Output Quality:  format compliance (rule-based) + LLM-as-Judge scoring
  L1 — Business Value:  task completion rate, adoption rate (simulated)

Workflow:
  1. Run 6 real Skill invocations with varied inputs → collect L3 data
  2. Score each output with LLM-as-Judge → collect L2 data
  3. Apply simulated L1 user feedback
  4. Compute all metrics, render health dashboard, check alert thresholds

Subject Skill: rnd-technical-writer
  Given a topic, writes a Markdown technical article with frontmatter,
  H2 sections, and at least one code block.

Run:
    conda activate dev_base
    python demo_skill_metrics.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass

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

WIDTH = 70

# ── Skill definition ──────────────────────────────────────────────────────────

SKILL_PROMPT = """You are an expert technical writer.
Write a complete Markdown technical article.

Output must include:
- YAML frontmatter (title, description, tags)
- At least 3 H2 sections (## heading)
- At least 1 fenced code block
- A brief summary section at the end

Target length: 500-900 words. Write in the same language as the request."""

# ── Test inputs ───────────────────────────────────────────────────────────────

SKILL_INPUTS = [
    # Normal cases
    {"id": "T01", "input": "Write a technical article about Python asyncio event loop internals"},
    {"id": "T02", "input": "写一篇关于 Redis 缓存穿透、击穿、雪崩的技术文章"},
    {"id": "T03", "input": "Write a technical article about Docker multi-stage builds"},
    # Edge cases
    {"id": "T04", "input": "写一篇关于 LangGraph 状态管理的入门教程"},
    {"id": "T05", "input": "Write a short technical note about HTTP/2 multiplexing"},
    {"id": "T06", "input": "写一篇介绍 Rust 所有权模型的文章，面向有 Python 背景的读者"},
]

# ── L3 data collection ────────────────────────────────────────────────────────

@dataclass
class L3Metrics:
    call_id: str
    latency_s: float
    token_estimate: int
    success: bool
    error: str = ""


def count_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def invoke_skill(call_id: str, user_input: str) -> tuple[str, L3Metrics]:
    t0 = time.time()
    try:
        output = str(llm.invoke([
            SystemMessage(content=SKILL_PROMPT),
            HumanMessage(content=user_input),
        ]).content)
        latency = time.time() - t0
        tokens = count_tokens(SKILL_PROMPT + user_input + output)
        return output, L3Metrics(call_id, latency, tokens, True)
    except Exception as exc:
        latency = time.time() - t0
        return "", L3Metrics(call_id, latency, 0, False, str(exc))


# ── L2 quality scoring ────────────────────────────────────────────────────────

@dataclass
class L2Metrics:
    call_id: str
    format_compliant: bool
    format_issues: list[str]
    quality_score: float          # weighted LLM-as-Judge score
    judge_scores: dict


JUDGE_PROMPT = """Evaluate this AI-generated technical article on 4 dimensions (1–5 each).

Article:
{article}

Dimensions:
1. technical_accuracy — factually correct and precise?
2. depth              — goes beyond surface-level?
3. clarity            — well-organized, easy to follow?
4. practical_value    — includes actionable examples or code?

Respond in valid JSON only (no markdown fences):
{{
  "technical_accuracy": <1-5>,
  "depth": <1-5>,
  "clarity": <1-5>,
  "practical_value": <1-5>
}}"""

JUDGE_WEIGHTS = {
    "technical_accuracy": 0.35,
    "depth": 0.25,
    "clarity": 0.20,
    "practical_value": 0.20,
}


def check_format(article: str) -> tuple[bool, list[str]]:
    issues = []
    if "---" not in article[:300]:
        issues.append("missing frontmatter")
    h2_count = len(re.findall(r"^## ", article, re.MULTILINE))
    if h2_count < 3:
        issues.append(f"only {h2_count} H2 sections (need ≥3)")
    if "```" not in article:
        issues.append("no code block")
    word_count = len(article.split())
    if word_count < 200:
        issues.append(f"too short: {word_count} words")
    return len(issues) == 0, issues


def judge_quality(article: str) -> dict:
    prompt = JUDGE_PROMPT.format(article=article[:2500])
    raw = str(llm.invoke([HumanMessage(content=prompt)]).content)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"technical_accuracy": 3, "depth": 3, "clarity": 3, "practical_value": 3}


def score_l2(call_id: str, article: str) -> L2Metrics:
    compliant, issues = check_format(article)
    scores = judge_quality(article)
    weighted = sum(scores.get(k, 3) * w for k, w in JUDGE_WEIGHTS.items())
    return L2Metrics(call_id, compliant, issues, round(weighted, 2), scores)


# ── L1 simulated feedback ─────────────────────────────────────────────────────

# Simulated user feedback (in production: from actual user interactions)
# Scale: {"adopted": bool, "rating": 1-5, "task_complete": bool}
SIMULATED_L1 = {
    "T01": {"adopted": True,  "rating": 4, "task_complete": True},
    "T02": {"adopted": True,  "rating": 5, "task_complete": True},
    "T03": {"adopted": False, "rating": 3, "task_complete": True},
    "T04": {"adopted": True,  "rating": 4, "task_complete": True},
    "T05": {"adopted": True,  "rating": 3, "task_complete": True},
    "T06": {"adopted": False, "rating": 2, "task_complete": False},
}


# ── Metric aggregation ────────────────────────────────────────────────────────

@dataclass
class SkillHealthReport:
    skill_name: str
    # L3
    availability: float
    p90_latency: float
    p99_latency: float
    avg_tokens: float
    error_rate: float
    # L2
    format_compliance: float
    avg_quality_score: float
    # L1
    task_completion_rate: float
    adoption_rate: float
    avg_rating: float


def compute_metrics(
    l3_data: list[L3Metrics],
    l2_data: list[L2Metrics],
    l1_data: dict,
) -> SkillHealthReport:
    # L3
    n = len(l3_data)
    successes = [m for m in l3_data if m.success]
    availability = len(successes) / n
    latencies = sorted(m.latency_s for m in l3_data)
    p90_idx = max(0, int(0.90 * n) - 1)
    p99_idx = max(0, int(0.99 * n) - 1)
    avg_tokens = statistics.mean(m.token_estimate for m in successes) if successes else 0

    # L2
    compliant = sum(1 for m in l2_data if m.format_compliant)
    format_compliance = compliant / len(l2_data) if l2_data else 0
    avg_quality = statistics.mean(m.quality_score for m in l2_data) if l2_data else 0

    # L1
    feedback_list = list(l1_data.values())
    task_completion = sum(1 for f in feedback_list if f["task_complete"]) / len(feedback_list)
    adoption = sum(1 for f in feedback_list if f["adopted"]) / len(feedback_list)
    avg_rating = statistics.mean(f["rating"] for f in feedback_list)

    return SkillHealthReport(
        skill_name="rnd-technical-writer",
        availability=availability,
        p90_latency=latencies[p90_idx],
        p99_latency=latencies[p99_idx],
        avg_tokens=avg_tokens,
        error_rate=1 - availability,
        format_compliance=format_compliance,
        avg_quality_score=avg_quality,
        task_completion_rate=task_completion,
        adoption_rate=adoption,
        avg_rating=avg_rating,
    )


# ── Alert check ───────────────────────────────────────────────────────────────

THRESHOLDS = {
    # L3
    "availability":        {"min": 0.99, "label": "Availability < 99%",        "severity": "CRITICAL"},
    "p90_latency":         {"max": 30.0, "label": "P90 latency > 30s",          "severity": "WARNING"},
    "p99_latency":         {"max": 60.0, "label": "P99 latency > 60s",          "severity": "WARNING"},
    # L2
    "format_compliance":   {"min": 0.95, "label": "Format compliance < 95%",    "severity": "WARNING"},
    "avg_quality_score":   {"min": 3.8,  "label": "Quality score < 3.8/5",      "severity": "WARNING"},
    # L1
    "task_completion_rate":{"min": 0.75, "label": "Task completion < 75%",      "severity": "CRITICAL"},
    "adoption_rate":       {"min": 0.60, "label": "Adoption rate < 60%",        "severity": "WARNING"},
    "avg_rating":          {"min": 4.0,  "label": "Avg rating < 4.0",           "severity": "WARNING"},
}


def check_alerts(report: SkillHealthReport) -> list[dict]:
    alerts = []
    for metric, rule in THRESHOLDS.items():
        value = getattr(report, metric)
        if "min" in rule and value < rule["min"]:
            alerts.append({"metric": metric, "value": value,
                           "threshold": rule["min"], "label": rule["label"],
                           "severity": rule["severity"]})
        elif "max" in rule and value > rule["max"]:
            alerts.append({"metric": metric, "value": value,
                           "threshold": rule["max"], "label": rule["label"],
                           "severity": rule["severity"]})
    return alerts


# ── Dashboard render ──────────────────────────────────────────────────────────

def render_dashboard(report: SkillHealthReport, alerts: list[dict]) -> None:
    print(f"\n{'═' * WIDTH}")
    print(f"  Skill Health Dashboard: {report.skill_name}")
    print(f"{'═' * WIDTH}")

    def status(value: float, threshold: float, mode: str = "min") -> str:
        ok = value >= threshold if mode == "min" else value <= threshold
        return "✓" if ok else "✗"

    # L3
    print(f"\n  ── L3: System Health ──")
    print(f"  {'Metric':<28} {'Value':>10}  {'Threshold':>12}  Status")
    print(f"  {'─'*28} {'─'*10}  {'─'*12}  {'─'*6}")
    print(f"  {'Availability':<28} {report.availability:>9.1%}  {'>99%':>12}  "
          f"{status(report.availability, 0.99)}")
    print(f"  {'P90 Latency':<28} {report.p90_latency:>9.1f}s  {'<30s':>12}  "
          f"{status(report.p90_latency, 30.0, 'max')}")
    print(f"  {'P99 Latency':<28} {report.p99_latency:>9.1f}s  {'<60s':>12}  "
          f"{status(report.p99_latency, 60.0, 'max')}")
    print(f"  {'Avg Tokens/call':<28} {report.avg_tokens:>9.0f}   {'(budget)':>12}")

    # L2
    print(f"\n  ── L2: Output Quality ──")
    print(f"  {'Format Compliance':<28} {report.format_compliance:>9.1%}  {'>95%':>12}  "
          f"{status(report.format_compliance, 0.95)}")
    print(f"  {'Quality Score (L-J)':<28} {report.avg_quality_score:>9.2f}   {'>3.8/5':>12}  "
          f"{status(report.avg_quality_score, 3.8)}")

    # L1
    print(f"\n  ── L1: Business Value ──")
    print(f"  {'Task Completion Rate':<28} {report.task_completion_rate:>9.1%}  {'>75%':>12}  "
          f"{status(report.task_completion_rate, 0.75)}")
    print(f"  {'Adoption Rate':<28} {report.adoption_rate:>9.1%}  {'>60%':>12}  "
          f"{status(report.adoption_rate, 0.60)}")
    print(f"  {'Avg User Rating':<28} {report.avg_rating:>9.2f}   {'>4.0/5':>12}  "
          f"{status(report.avg_rating, 4.0)}")

    # Alerts
    print(f"\n  ── Alerts ──")
    if not alerts:
        print("  No alerts — all metrics within thresholds")
    else:
        for a in alerts:
            icon = "🔴" if a["severity"] == "CRITICAL" else "🟡"
            print(f"  {icon} [{a['severity']}] {a['label']}")
            print(f"     current={a['value']:.3f}  threshold={a['threshold']}")

    print(f"\n{'═' * WIDTH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Skill Metrics Demo — L1 / L2 / L3 Framework")
    print(f"Skill: rnd-technical-writer  |  Calls: {len(SKILL_INPUTS)}")
    print("=" * WIDTH)

    l3_results: list[L3Metrics] = []
    l2_results: list[L2Metrics] = []
    outputs: dict[str, str] = {}

    # Step 1: invoke Skill, collect L3
    print(f"\n{'─' * WIDTH}")
    print("Step 1: Skill Invocations (collecting L3 metrics)")
    print(f"{'─' * WIDTH}")
    for item in SKILL_INPUTS:
        print(f"  [{item['id']}] {item['input'][:60]}...")
        output, l3 = invoke_skill(item["id"], item["input"])
        l3_results.append(l3)
        outputs[item["id"]] = output
        status_str = f"✓ {l3.latency_s:.1f}s  ~{l3.token_estimate} tokens" if l3.success else f"✗ {l3.error[:40]}"
        print(f"       {status_str}")

    # Step 2: score outputs, collect L2
    print(f"\n{'─' * WIDTH}")
    print("Step 2: Output Scoring (L2 — format check + LLM-as-Judge)")
    print(f"{'─' * WIDTH}")
    for item in SKILL_INPUTS:
        output = outputs.get(item["id"], "")
        if not output:
            continue
        l2 = score_l2(item["id"], output)
        l2_results.append(l2)
        fmt_str = "✓ format ok" if l2.format_compliant else f"✗ {l2.format_issues}"
        s = l2.judge_scores
        print(f"  [{item['id']}] {fmt_str}  |  "
              f"acc={s.get('technical_accuracy','?')} "
              f"dep={s.get('depth','?')} "
              f"cla={s.get('clarity','?')} "
              f"pra={s.get('practical_value','?')}  "
              f"→ {l2.quality_score:.2f}/5")

    # Step 3: aggregate all layers
    report = compute_metrics(l3_results, l2_results, SIMULATED_L1)
    alerts = check_alerts(report)

    # Step 4: render dashboard
    render_dashboard(report, alerts)

    # Per-call detail
    print("\n  Per-call L2 detail:")
    print(f"  {'ID':<6} {'Format':<14} {'Quality':>8}  Notes")
    print(f"  {'─'*6} {'─'*14} {'─'*8}  {'─'*30}")
    for m in l2_results:
        fmt = "✓" if m.format_compliant else "✗"
        note = "; ".join(m.format_issues) if m.format_issues else "—"
        print(f"  {m.call_id:<6} {fmt:<14} {m.quality_score:>7.2f}  {note}")

    print()


if __name__ == "__main__":
    main()
