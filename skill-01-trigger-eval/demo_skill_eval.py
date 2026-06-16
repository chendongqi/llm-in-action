"""
Skill Evaluation Demo

Demonstrates two evaluation pillars for AI Skills:
  1. Trigger Evaluation   — does the Skill activate at the right time?
     Metrics: Recall / Precision / F1  (TP / TN / FP / FN)
  2. Task Completion Eval — does the Skill do the job well?
     Level 2: structural checks  (format, length, required sections)
     Level 3: LLM-as-Judge       (quality scoring on multiple criteria)

Subject Skill: rnd-technical-writer
  A Skill that writes technical blog articles in Chinese or English.
  Trigger: user asks to write / draft / create a technical article or blog post.

Run:
    conda activate dev_base
    python demo_skill_eval.py
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.0,  # deterministic for eval
)

WIDTH = 70

# ── Skill definition (simulated) ──────────────────────────────────────────────

SKILL_NAME = "rnd-technical-writer"

SKILL_DESCRIPTION = """
Name: rnd-technical-writer
Purpose: Write complete, well-structured technical blog articles or tutorials.
         Output is a full Markdown article with frontmatter, sections, code examples.

Trigger conditions — use this Skill when the user:
  - Asks to write / draft / create a technical article or blog post
  - Requests a tutorial on a technical topic
  - Asks for a technical deep-dive or explainer document
  - Wants to document a technical concept, architecture, or tool

Do NOT trigger when:
  - User is only asking a question about a technology (answer directly)
  - User wants to plan a multi-article SERIES (use rnd-blog-series-planner instead)
  - User wants to improve or edit an existing article (edit in-place)
  - Request is non-technical (marketing copy, story, etc.)
""".strip()

SKILL_SYSTEM_PROMPT = """You are an expert technical writer.
Write a complete, well-structured Markdown article based on the user's request.
Include: frontmatter (title, description, tags), H2 sections, at least one code block.
Target length: 800-1500 words."""

# ── Test cases ────────────────────────────────────────────────────────────────

# label: TP / TN / EDGE
# expect: "trigger" / "no_trigger"
TRIGGER_TEST_CASES = [
    # ── True Positives (TP) ──────────────────────────────────────────────────
    {"label": "TP_write_article",      "expect": "trigger",    "input": "帮我写一篇关于 RAG 技术的技术博客"},
    {"label": "TP_write_tutorial",     "expect": "trigger",    "input": "写一篇 LangChain 的入门教程"},
    {"label": "TP_draft_post",         "expect": "trigger",    "input": "Draft a technical blog post about Kubernetes networking"},
    {"label": "TP_write_deepdive",     "expect": "trigger",    "input": "帮我写一篇深入分析 Transformer 自注意力机制的文章"},
    {"label": "TP_write_explainer",    "expect": "trigger",    "input": "Write an explainer article about vector databases for developers"},
    {"label": "TP_write_guide",        "expect": "trigger",    "input": "给我写一篇 Docker 容器安全配置指南"},
    {"label": "TP_write_post_en",      "expect": "trigger",    "input": "Create a technical article on how Redis handles persistence"},
    {"label": "TP_write_arch",         "expect": "trigger",    "input": "写一篇介绍微服务架构设计原则的技术文章"},

    # ── True Negatives (TN) ─────────────────────────────────────────────────
    {"label": "TN_question",           "expect": "no_trigger", "input": "RAG 技术是什么？"},
    {"label": "TN_question_en",        "expect": "no_trigger", "input": "What is the difference between BERT and GPT?"},
    {"label": "TN_series_plan",        "expect": "no_trigger", "input": "帮我规划一个关于 LangGraph 的系列博客，要 6 篇"},
    {"label": "TN_series_plan_en",     "expect": "no_trigger", "input": "Plan a series of articles about Kubernetes, at least 5 posts"},
    {"label": "TN_edit_existing",      "expect": "no_trigger", "input": "这篇文章的第二段写得不够清晰，帮我改一下"},
    {"label": "TN_non_technical",      "expect": "no_trigger", "input": "帮我写一段产品介绍的营销文案"},
    {"label": "TN_code_help",          "expect": "no_trigger", "input": "帮我写一个解析 JSON 的 Python 函数"},
    {"label": "TN_summarize",          "expect": "no_trigger", "input": "总结一下这篇论文的核心观点"},

    # ── Edge Cases (EDGE) ────────────────────────────────────────────────────
    {"label": "EDGE_vague_write",      "expect": "trigger",    "input": "写点关于 Kafka 的内容"},           # "写" implied article
    {"label": "EDGE_single_vs_series", "expect": "trigger",    "input": "写几篇 Redis 相关的文章"},          # "几篇" = one batch, still articles
    {"label": "EDGE_long_single",      "expect": "trigger",    "input": "给我写一篇很详细的 CI/CD 实践总结"},  # long but single
    {"label": "EDGE_note_vs_article",  "expect": "no_trigger", "input": "帮我整理一下今天会议的技术要点"},    # notes ≠ article
]


# ── Trigger evaluation ────────────────────────────────────────────────────────

@dataclass
class TriggerResult:
    label: str
    expect: str
    predicted: str
    correct: bool
    reasoning: str


TRIGGER_EVAL_PROMPT = """You are evaluating whether a user message would trigger a specific AI Skill.

Skill specification:
{skill_description}

User message: "{user_input}"

Task: Decide whether this user message should trigger the Skill above.
Answer in valid JSON only (no markdown fences):
{{
  "prediction": "trigger" or "no_trigger",
  "reasoning": "one sentence explanation"
}}"""


def eval_trigger(case: dict) -> TriggerResult:
    prompt = TRIGGER_EVAL_PROMPT.format(
        skill_description=SKILL_DESCRIPTION,
        user_input=case["input"],
    )
    raw = str(llm.invoke([HumanMessage(content=prompt)]).content)
    # strip possible markdown fences
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        parsed = json.loads(raw)
        predicted = parsed.get("prediction", "no_trigger")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        # fallback: look for keyword
        predicted = "trigger" if "trigger" in raw.lower() and "no_trigger" not in raw.lower() else "no_trigger"
        reasoning = raw[:120]

    return TriggerResult(
        label=case["label"],
        expect=case["expect"],
        predicted=predicted,
        correct=(predicted == case["expect"]),
        reasoning=reasoning,
    )


def run_trigger_eval() -> dict:
    print(f"\n{'─' * WIDTH}")
    print("Part 1: Trigger Evaluation")
    print(f"Skill: {SKILL_NAME}  |  Test cases: {len(TRIGGER_TEST_CASES)}")
    print(f"{'─' * WIDTH}")

    results: list[TriggerResult] = []
    tp = tn = fp = fn = 0

    for i, case in enumerate(TRIGGER_TEST_CASES, 1):
        r = eval_trigger(case)
        results.append(r)

        if r.expect == "trigger" and r.predicted == "trigger":
            tp += 1; outcome = "✓ TP"
        elif r.expect == "no_trigger" and r.predicted == "no_trigger":
            tn += 1; outcome = "✓ TN"
        elif r.expect == "no_trigger" and r.predicted == "trigger":
            fp += 1; outcome = "✗ FP"
        else:
            fn += 1; outcome = "✗ FN"

        tag = r.label.split("_")[0]
        print(f"  [{i:2d}] {tag:<5} expect={r.expect:<10} got={r.predicted:<10} {outcome}")

    total = len(results)
    correct = tp + tn
    accuracy = correct / total

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    print(f"\n  Confusion matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy:  {accuracy:.0%}  ({correct}/{total})")
    print(f"  Recall:    {recall:.0%}")
    print(f"  Precision: {precision:.0%}")
    print(f"  F1:        {f1:.2f}")

    # show failures
    failures = [r for r in results if not r.correct]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(f"    {r.label}: expected={r.expect}, got={r.predicted}")
            print(f"      Reason: {r.reasoning}")

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


# ── Task completion evaluation ────────────────────────────────────────────────

TASK_INPUTS = [
    {
        "id": "T001",
        "input": "Write a technical article about Redis TTL configuration best practices",
        "expected_keywords": ["TTL", "expire", "eviction"],
        "min_words": 400,
        "required_sections": ["##"],   # at least one H2
        "required_code": True,
    },
    {
        "id": "T002",
        "input": "写一篇关于 Python 类型注解的入门文章",
        "expected_keywords": ["TypedDict", "Optional", "Annotated"],
        "min_words": 300,
        "required_sections": ["##"],
        "required_code": True,
    },
]

JUDGE_PROMPT = """You are an expert technical content reviewer.

Evaluate the following AI-generated technical article.

Article:
{article}

Scoring dimensions (1–5 each, where 5 = excellent):
1. Technical accuracy: Is the content factually correct and precise?
2. Depth: Does it go beyond surface-level explanations?
3. Clarity: Is it well-organized and easy to follow?
4. Practical value: Does it include actionable examples or code?

Respond in valid JSON only (no markdown fences):
{{
  "technical_accuracy": <1-5>,
  "depth": <1-5>,
  "clarity": <1-5>,
  "practical_value": <1-5>,
  "summary": "<one sentence overall assessment>"
}}"""


@dataclass
class TaskResult:
    task_id: str
    level2_pass: bool
    level2_notes: list[str]
    scores: dict
    weighted_score: float
    article_preview: str


def run_skill(task_input: str) -> str:
    """Simulate Skill execution: invoke LLM with Skill system prompt."""
    resp = llm.invoke([
        SystemMessage(content=SKILL_SYSTEM_PROMPT),
        HumanMessage(content=task_input),
    ])
    return str(resp.content)


def level2_check(article: str, spec: dict) -> tuple[bool, list[str]]:
    """Structural checks — no LLM needed."""
    notes = []
    word_count = len(article.split())
    if word_count < spec["min_words"]:
        notes.append(f"Too short: {word_count} words (min {spec['min_words']})")

    for section in spec["required_sections"]:
        if section not in article:
            notes.append(f"Missing required section marker: {section}")

    if spec["required_code"] and "```" not in article:
        notes.append("No code block found")

    return (len(notes) == 0), notes


def level3_judge(article: str) -> dict:
    """LLM-as-Judge scoring."""
    prompt = JUDGE_PROMPT.format(article=article[:3000])  # cap to avoid token blowout
    raw = str(llm.invoke([HumanMessage(content=prompt)]).content)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"technical_accuracy": 3, "depth": 3, "clarity": 3, "practical_value": 3,
                  "summary": "parse error — fallback scores"}
    return scores


WEIGHTS = {"technical_accuracy": 0.35, "depth": 0.25, "clarity": 0.20, "practical_value": 0.20}


def eval_task(spec: dict) -> TaskResult:
    article = run_skill(spec["input"])
    l2_pass, l2_notes = level2_check(article, spec)
    scores = level3_judge(article)

    weighted = sum(
        scores.get(dim, 3) * w
        for dim, w in WEIGHTS.items()
    )

    return TaskResult(
        task_id=spec["id"],
        level2_pass=l2_pass,
        level2_notes=l2_notes,
        scores=scores,
        weighted_score=weighted,
        article_preview=article[:200].replace("\n", " "),
    )


def run_task_eval() -> None:
    print(f"\n{'─' * WIDTH}")
    print("Part 2: Task Completion Evaluation")
    print(f"Tasks: {len(TASK_INPUTS)}  |  Level 2 (structural) + Level 3 (LLM-as-Judge)")
    print(f"{'─' * WIDTH}")

    for spec in TASK_INPUTS:
        print(f"\n  [{spec['id']}] {spec['input'][:60]}")
        t0 = time.time()
        result = eval_task(spec)
        elapsed = time.time() - t0

        # Level 2
        l2_icon = "✓" if result.level2_pass else "✗"
        print(f"    Level 2 (structural): {l2_icon}", end="")
        if result.level2_notes:
            print(f"  Issues: {result.level2_notes}")
        else:
            print("  All checks passed")

        # Level 3
        s = result.scores
        print(f"    Level 3 (LLM-as-Judge):")
        print(f"      Technical accuracy: {s.get('technical_accuracy', '?')}/5")
        print(f"      Depth:              {s.get('depth', '?')}/5")
        print(f"      Clarity:            {s.get('clarity', '?')}/5")
        print(f"      Practical value:    {s.get('practical_value', '?')}/5")
        print(f"      Weighted score:     {result.weighted_score:.2f}/5")
        print(f"      Summary:            {s.get('summary', '')}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Article preview: {result.article_preview[:100]}...")


# ── A/B comparison ────────────────────────────────────────────────────────────

SKILL_SYSTEM_PROMPT_V2 = """You are an expert technical writer specializing in developer education.
Write a complete Markdown article. Requirements:
- Start with a concrete problem or pain point the reader faces
- Include at least 2 code examples with inline comments
- End with a practical checklist or next steps
- Target length: 800-1500 words
- Frontmatter: title, description, tags"""

AB_INPUT = "Write a technical article about Python type hints for API development"


def run_ab_comparison() -> None:
    print(f"\n{'─' * WIDTH}")
    print("Part 3: A/B Prompt Comparison")
    print(f"Input: {AB_INPUT}")
    print(f"{'─' * WIDTH}")

    # Version A: original prompt
    print("\n  [Version A] Original system prompt")
    t0 = time.time()
    article_a = run_skill(AB_INPUT)
    scores_a = level3_judge(article_a)
    wa = sum(scores_a.get(d, 3) * w for d, w in WEIGHTS.items())
    print(f"    Weighted: {wa:.2f}/5  ({time.time()-t0:.1f}s)")
    for dim in WEIGHTS:
        print(f"    {dim:<22}: {scores_a.get(dim, '?')}/5")

    # Version B: improved prompt
    print("\n  [Version B] Improved system prompt (pain-point hook + checklist)")
    t0 = time.time()
    resp_b = llm.invoke([
        SystemMessage(content=SKILL_SYSTEM_PROMPT_V2),
        HumanMessage(content=AB_INPUT),
    ])
    article_b = str(resp_b.content)
    scores_b = level3_judge(article_b)
    wb = sum(scores_b.get(d, 3) * w for d, w in WEIGHTS.items())
    print(f"    Weighted: {wb:.2f}/5  ({time.time()-t0:.1f}s)")
    for dim in WEIGHTS:
        print(f"    {dim:<22}: {scores_b.get(dim, '?')}/5")

    # verdict
    print(f"\n  A/B Verdict:")
    print(f"    Version A score: {wa:.2f}")
    print(f"    Version B score: {wb:.2f}")
    delta = wb - wa
    if abs(delta) < 0.1:
        verdict = "No significant difference (<0.1 delta)"
    elif delta > 0:
        verdict = f"Version B wins  (+{delta:.2f})"
    else:
        verdict = f"Version A wins  ({delta:.2f})"
    print(f"    Result: {verdict}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Skill Evaluation Demo")
    print(f"Model: glm-4-flash  |  Skill: {SKILL_NAME}")
    print(f"{'=' * WIDTH}")

    t_total = time.time()

    # Part 1: Trigger evaluation (TP/TN/FP/FN + F1)
    trigger_metrics = run_trigger_eval()

    # Part 2: Task completion (structural + LLM-as-Judge)
    run_task_eval()

    # Part 3: A/B prompt comparison
    run_ab_comparison()

    elapsed = time.time() - t_total
    print(f"\n{'=' * WIDTH}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Trigger F1: {trigger_metrics['f1']:.2f}  |  "
          f"Accuracy: {trigger_metrics['accuracy']:.0%}")
    print(f"{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
