"""
Skill Design Patterns Demo

Compares two versions of the same Skill across four test scenarios:

  anti_pattern_skill  — violates 5 common anti-patterns:
    ❌ Vague trigger ("when user needs help")
    ❌ No output contract (format undefined)
    ❌ No exclusions declared
    ❌ No degradation strategy for partial input
    ❌ All-in-one (covers too many responsibilities)

  pattern_skill — applies 5 core design patterns:
    ✅ Single Responsibility
    ✅ Contract-Driven (defined input/output structure)
    ✅ Progressive Enhancement (graceful degradation)
    ✅ Observable Design (shows reasoning steps)
    ✅ Defensive Output (uncertainty labels, source notes)

Test scenarios:
  S1: Complete input → both should work; observe output structure
  S2: Partial input (missing dimensions) → pattern_skill degrades gracefully
  S3: Out-of-scope request → pattern_skill redirects; anti-pattern might comply
  S4: Request requiring uncertain/outdated data → pattern_skill labels uncertainty

After runtime tests, a Prompt design quality audit scores both Skill prompts
on the 5 pattern dimensions (1–5 each).

Subject Skill: competitor-analyzer
  Purpose: analyze a competitor company from product/tech/market dimensions.

Run:
    conda activate dev_base
    python demo_skill_design.py
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

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

WIDTH = 70

# ── Skill definitions ─────────────────────────────────────────────────────────

ANTI_PATTERN_PROMPT = """You are a helpful marketing assistant.
When the user needs help with competitors or marketing, assist them.
Provide useful information about the topic they ask about.
Be comprehensive and cover all relevant aspects."""

PATTERN_PROMPT = """# competitor-analyzer

## Role (Who)
You are a competitive intelligence analyst. You produce structured,
evidence-based competitor assessments grounded in publicly available information.

## Trigger (When)
Use this Skill when the user asks to:
- Analyze a specific competitor company
- Compare a company against competitors
- Understand a company's market position, product, or technology

**Not applicable when:**
- User asks to write a marketing plan or campaign (use campaign-planner)
- User asks for internal company analysis (use business-analyst)
- User asks a general industry question without a named company

## Task (What)
Produce a structured competitor analysis covering the dimensions specified.
Default dimensions if not specified: Product, Technology, Market Position.
Default time range if not specified: recent 6 months of publicly available data.

## Execution Steps (How)
1. **Parse request**: identify company name, analysis dimensions, time range
2. **State assumptions**: explicitly list any dimensions or time range you are assuming
3. **Analyze each dimension**: one section per dimension, cite publicly known facts
4. **Flag uncertainty**: mark any claim that cannot be verified with "[unverified]"
5. **Summarize**: 2–3 sentence strategic summary

## Output Contract
Required structure:
```
## Competitor Analysis: [Company Name]
**Assumptions:** [list any inferred inputs]

### [Dimension 1]
[Analysis, 3–5 sentences, cite public sources where possible]

### [Dimension 2]
...

## Strategic Summary
[2–3 sentences]
```

Maximum length: 600 words.
Do not include: financial projections, internal data, or fabricated statistics.

## Degradation Strategy
- Company name only → use default 3 dimensions, state this assumption
- Company + some dimensions → use provided dimensions, fill defaults for rest
- Ambiguous company name → ask: "Did you mean [X] or [Y]?"
- No company name at all → ask: "Which company would you like me to analyze?"

## Constraints
- Label unverifiable claims with [unverified]
- Do not write marketing copy or strategic recommendations
- Do not claim to have real-time data; note information currency limitations
"""


def call_skill(system_prompt: str, user_input: str) -> str:
    return str(llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]).content)


# ── Test scenarios ────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    id: str
    name: str
    input: str
    # What to look for in the output
    check_for: list[str]       # these should appear in pattern_skill output
    check_absent: list[str]    # these should NOT appear in pattern_skill output


SCENARIOS = [
    Scenario(
        id="S1",
        name="Complete input — full analysis request",
        input=(
            "Analyze Notion as a competitor. Focus on: Product features, "
            "Market positioning. Time range: last 12 months."
        ),
        check_for=["## Competitor Analysis", "### Product", "### Market", "Strategic Summary"],
        check_absent=[],
    ),
    Scenario(
        id="S2",
        name="Partial input — company name only",
        input="Analyze Figma as a competitor.",
        check_for=["Assumption", "dimension"],
        check_absent=[],
    ),
    Scenario(
        id="S3",
        name="Out-of-scope — asks for marketing plan",
        input=(
            "Analyze Slack as a competitor and then write us a marketing "
            "campaign to win back their users."
        ),
        check_for=["competitor-analyzer", "campaign"],   # pattern should redirect
        check_absent=[],
    ),
    Scenario(
        id="S4",
        name="Uncertain data — asks for real-time financial data",
        input=(
            "Analyze Linear's latest funding round and current ARR. "
            "What is their exact revenue this quarter?"
        ),
        check_for=["unverified", "public"],
        check_absent=["exact revenue is", "$"],
    ),
]


# ── Output scoring ────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    anti_output: str
    pattern_output: str
    anti_checks: dict   # {check: found?}
    pattern_checks: dict


def score_output(output: str, check_for: list[str], check_absent: list[str]) -> dict:
    result = {}
    for term in check_for:
        result[f"has:{term[:20]}"] = term.lower() in output.lower()
    for term in check_absent:
        result[f"absent:{term[:20]}"] = term.lower() not in output.lower()
    return result


def run_scenario(scenario: Scenario) -> ScenarioResult:
    anti_out = call_skill(ANTI_PATTERN_PROMPT, scenario.input)
    pattern_out = call_skill(PATTERN_PROMPT, scenario.input)

    return ScenarioResult(
        scenario_id=scenario.id,
        name=scenario.name,
        anti_output=anti_out,
        pattern_output=pattern_out,
        anti_checks=score_output(anti_out, scenario.check_for, scenario.check_absent),
        pattern_checks=score_output(pattern_out, scenario.check_for, scenario.check_absent),
    )


# ── Design quality audit ──────────────────────────────────────────────────────

AUDIT_PROMPT = """You are a Skill design reviewer. Evaluate the following AI Skill prompt
on 5 design dimensions. Score each 1–5 (5 = excellent).

Skill prompt to evaluate:
---
{skill_prompt}
---

Dimensions:
1. Single Responsibility — does it do exactly one thing, clearly scoped?
2. Contract-Driven — are input/output formats explicitly defined?
3. Progressive Enhancement — is there a degradation strategy for partial/missing input?
4. Observable Design — does it require showing reasoning steps or intermediate state?
5. Defensive Output — does it require uncertainty labels and avoid unsafe outputs?

Respond in valid JSON only (no markdown fences):
{{
  "single_responsibility": <1-5>,
  "contract_driven": <1-5>,
  "progressive_enhancement": <1-5>,
  "observable_design": <1-5>,
  "defensive_output": <1-5>,
  "total": <sum>,
  "summary": "<one sentence overall assessment>"
}}"""


def audit_skill(prompt_text: str) -> dict:
    raw = str(llm.invoke([
        HumanMessage(content=AUDIT_PROMPT.format(skill_prompt=prompt_text[:2000]))
    ]).content)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"total": 0, "summary": "parse error"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Skill Design Patterns Demo")
    print("Subject: competitor-analyzer  |  Scenarios: 4  |  Audit: 2 Skill prompts")
    print("=" * WIDTH)

    results: list[ScenarioResult] = []

    # Part 1: Runtime comparison
    print(f"\n{'─' * WIDTH}")
    print("Part 1: Runtime Behavior Comparison")
    print(f"{'─' * WIDTH}")

    for scenario in SCENARIOS:
        print(f"\n  [{scenario.id}] {scenario.name}")
        t0 = time.time()
        result = run_scenario(scenario)
        results.append(result)
        elapsed = time.time() - t0

        # Pattern skill checks
        p_pass = sum(1 for v in result.pattern_checks.values() if v)
        p_total = len(result.pattern_checks)
        a_pass = sum(1 for v in result.anti_checks.values() if v)
        a_total = len(result.anti_checks)

        print(f"    Anti-pattern skill:  {a_pass}/{a_total} checks passed")
        print(f"    Pattern skill:       {p_pass}/{p_total} checks passed")

        # Show key outputs (preview)
        print(f"    Anti-pattern preview: {result.anti_output[:120].replace(chr(10),' ')}...")
        print(f"    Pattern preview:      {result.pattern_output[:120].replace(chr(10),' ')}...")
        print(f"    ({elapsed:.1f}s)")

    # Part 2: Design quality audit
    print(f"\n{'─' * WIDTH}")
    print("Part 2: Prompt Design Quality Audit (LLM-as-Judge)")
    print(f"{'─' * WIDTH}")

    print("\n  Auditing anti-pattern Skill prompt...")
    t0 = time.time()
    anti_audit = audit_skill(ANTI_PATTERN_PROMPT)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\n  Auditing pattern Skill prompt...")
    t0 = time.time()
    pattern_audit = audit_skill(PATTERN_PROMPT)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Print audit table
    print(f"\n  {'Dimension':<28} {'Anti-pattern':>12}  {'Pattern':>7}")
    print(f"  {'─'*28} {'─'*12}  {'─'*7}")
    dims = [
        ("Single Responsibility",   "single_responsibility"),
        ("Contract-Driven",         "contract_driven"),
        ("Progressive Enhancement", "progressive_enhancement"),
        ("Observable Design",       "observable_design"),
        ("Defensive Output",        "defensive_output"),
    ]
    for label, key in dims:
        a_score = anti_audit.get(key, "?")
        p_score = pattern_audit.get(key, "?")
        delta = ""
        if isinstance(a_score, int) and isinstance(p_score, int):
            d = p_score - a_score
            delta = f"  (+{d})" if d > 0 else (f"  ({d})" if d < 0 else "  (=)")
        print(f"  {label:<28} {a_score:>12}  {p_score:>7}{delta}")

    a_total = anti_audit.get("total", "?")
    p_total = pattern_audit.get("total", "?")
    print(f"  {'─'*28} {'─'*12}  {'─'*7}")
    print(f"  {'TOTAL (max 25)':<28} {a_total:>12}  {p_total:>7}")
    print(f"\n  Anti-pattern: {anti_audit.get('summary','')}")
    print(f"  Pattern:      {pattern_audit.get('summary','')}")

    # Summary
    print(f"\n{'=' * WIDTH}")
    print("Summary")
    total_checks_p = sum(
        sum(1 for v in r.pattern_checks.values() if v) for r in results
    )
    total_checks_total = sum(len(r.pattern_checks) for r in results)
    total_checks_a = sum(
        sum(1 for v in r.anti_checks.values() if v) for r in results
    )
    print(f"  Runtime checks — Anti-pattern: {total_checks_a}/{total_checks_total}")
    print(f"  Runtime checks — Pattern:      {total_checks_p}/{total_checks_total}")
    if isinstance(a_total, int) and isinstance(p_total, int):
        print(f"  Design audit   — Anti-pattern: {a_total}/25")
        print(f"  Design audit   — Pattern:      {p_total}/25")
    print(f"{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
