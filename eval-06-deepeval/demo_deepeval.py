"""
AI Eval Article 06 — DeepEval: Enterprise Agent Evaluation Suite

DeepEval is a pytest-based testing framework for LLM applications.
Unlike RAGAS (metric-first, DataFrames), DeepEval uses a test-case-first
approach that integrates naturally with CI pipelines.

This demo evaluates the same customer support Agent from Article 05
using DeepEval metrics, then compares the two frameworks:

DeepEval metrics used:
  AnswerRelevancyMetric   → does the answer address the question?
  FaithfulnessMetric      → does the answer stay grounded in context?
  ToolCorrectnessMetric   → were the right tools called?

Key differences from RAGAS:
  - DeepEval uses test cases + assert_test() (pytest-style)
  - RAGAS uses Dataset + evaluate() (batch, DataFrame-style)
  - DeepEval has built-in threshold assertions (pass/fail per test)
  - DeepEval is better for CI gates; RAGAS is better for batch analysis

Note on LLM setup:
  DeepEval defaults to OpenAI. We configure it to use glm-4-flash
  via the DEEPEVAL_* environment variables and a custom LLM class.

Run:
    conda activate llm_base
    pip install deepeval
    python demo_deepeval.py
"""

from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()

# ── Configure DeepEval to use custom LLM ─────────────────────────────────────

from deepeval.models.base_model import DeepEvalBaseLLM  # type: ignore[import-untyped]

class GlmFlashEval(DeepEvalBaseLLM):
    """Wraps glm-4-flash for use as DeepEval's judge LLM."""

    def __init__(self):
        self._llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
            base_url="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.0,
        )

    def load_model(self):
        return self._llm

    def generate(self, prompt: str, *args, **kwargs) -> str:
        result = self._llm.invoke([HumanMessage(content=prompt)])
        return str(result.content)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "glm-4-flash"


judge_llm = GlmFlashEval()

# ── Agent (same as Article 05) ────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.0,
)

FAQ_DB = {
    "退款政策": "购买后 7 天内可全额退款，7-30 天退 50%，30 天后不退款。",
    "配送时间": "普通配送 3-5 个工作日，快递配送 1-2 个工作日。",
    "质保期": "所有产品提供 1 年质保，人为损坏不在质保范围内。",
    "支付方式": "支持微信支付、支付宝、银行卡、信用卡。",
}

ORDER_DB = {
    "ORD-001": {"status": "已发货", "item": "无线耳机", "amount": 299.0, "days_since": 3},
    "ORD-004": {"status": "已完成", "item": "充电宝", "amount": 99.0, "days_since": 15},
}


@tool
def search_faq(query: str) -> str:
    """搜索产品常见问题（FAQ）。"""
    for key, value in FAQ_DB.items():
        if key in query or any(w in query for w in key.split()):
            return f"{key}：{value}"
    return "未找到相关 FAQ。"


@tool
def get_order_status(order_id: str) -> str:
    """查询订单状态。需要提供订单号（格式：ORD-XXX）。"""
    if order_id in ORDER_DB:
        o = ORDER_DB[order_id]
        return (f"订单 {order_id}：{o['item']}，状态：{o['status']}，"
                f"金额：¥{o['amount']}，下单 {o['days_since']} 天前。")
    return f"未找到订单 {order_id}。"


@tool
def calculate_refund(amount: float, days_since_purchase: int) -> str:
    """计算退款金额。"""
    if days_since_purchase <= 7:
        refund = amount
        policy = "7天内全额退款"
    elif days_since_purchase <= 30:
        refund = amount * 0.5
        policy = "7-30天退50%"
    else:
        refund = 0.0
        policy = "30天后不退款"
    return f"退款金额：¥{refund:.2f}（{policy}，原价 ¥{amount}）"


TOOLS = [search_faq, get_order_status, calculate_refund]
TOOL_MAP = {t.name: t for t in TOOLS}
bound_llm = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """你是一名客服助手，帮助用户解答问题和处理订单。
必须使用工具回答问题：
- 关于政策/服务的问题 → search_faq
- 关于订单状态 → get_order_status
- 计算退款 → calculate_refund"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


def agent_node(state: AgentState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = bound_llm.invoke(msgs)
    return {"messages": [response], "steps": state["steps"] + 1}


def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:  # type: ignore[union-attr]
        output = TOOL_MAP[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return {"messages": results}


def router(state: AgentState) -> str:
    if state["steps"] >= 6:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


g = StateGraph(AgentState)
g.add_node("agent", agent_node)
g.add_node("tools", tools_node)
g.set_entry_point("agent")
g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
g.add_edge("tools", "agent")
agent_graph = g.compile()


def run_agent(question: str) -> tuple[str, list[str], list[str]]:
    """Returns (answer, tools_called, retrieval_context)."""
    state = agent_graph.invoke(
        {"messages": [HumanMessage(content=question)], "steps": 0},
        config={"recursion_limit": 20},
    )
    tools_called = []
    context = []
    answer = ""
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
        if isinstance(msg, ToolMessage):
            context.append(str(msg.content))
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            answer = str(msg.content)
    return answer, tools_called, context


# ── DeepEval test cases ───────────────────────────────────────────────────────

from deepeval.test_case import LLMTestCase, ToolCall  # type: ignore[import-untyped]
from deepeval.metrics import (  # type: ignore[import-untyped]
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToolCorrectnessMetric,
)

WIDTH = 70

TEST_CASES_SPEC = [
    {
        "input": "你们的退款政策是什么？",
        "expected_tools": ["search_faq"],
    },
    {
        "input": "我的订单 ORD-001 发货了吗？",
        "expected_tools": ["get_order_status"],
    },
    {
        "input": "ORD-004 这个订单能退款多少钱？",
        "expected_tools": ["get_order_status", "calculate_refund"],
    },
    {
        "input": "支持哪些支付方式？",
        "expected_tools": ["search_faq"],
    },
    {
        "input": "我买了 299 元的商品，3 天前买的，可以退款吗？",
        "expected_tools": ["calculate_refund"],
    },
]


def build_test_cases() -> list[LLMTestCase]:
    """Run agent and build DeepEval test cases."""
    print("  Running agent to collect outputs...")
    cases = []
    for spec in TEST_CASES_SPEC:
        answer, tools_called, context = run_agent(spec["input"])
        # DeepEval requires ToolCall objects, not plain strings
        tools_called_objs = [ToolCall(name=t) for t in tools_called]
        expected_tools_objs = [ToolCall(name=t) for t in spec["expected_tools"]]
        case = LLMTestCase(
            input=spec["input"],
            actual_output=answer,
            expected_tools=expected_tools_objs,
            tools_called=tools_called_objs,
            retrieval_context=context if context else ["No context retrieved"],
        )
        cases.append(case)
        print(f"    Q: {spec['input'][:40]}...  tools={tools_called}")
        time.sleep(0.3)
    return cases


def run_deepeval_metrics(cases: list[LLMTestCase]) -> list[dict]:
    """Evaluate each test case on three metrics, return results."""
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge_llm, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=judge_llm, include_reason=True),
        ToolCorrectnessMetric(model=judge_llm),
    ]

    results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case.input[:45]}...")
        tools_names = [t.name for t in (case.tools_called or [])]
        case_results = {"input": case.input, "tools_called": tools_names}

        for metric in metrics:
            metric.measure(case)
            name = metric.__class__.__name__.replace("Metric", "")
            case_results[name] = {
                "score": round(metric.score, 3) if hasattr(metric, "score") and metric.score is not None else None,
                "passed": metric.is_successful(),
                "reason": getattr(metric, "reason", "") or "",
            }
            time.sleep(0.5)

        results.append(case_results)

    return results


def print_results(results: list[dict]) -> None:
    """Print evaluation results in a readable format."""
    print(f"\n{'─' * WIDTH}")
    print("DeepEval Results")
    print(f"{'─' * WIDTH}")

    metric_names = ["AnswerRelevancy", "Faithfulness", "ToolCorrectness"]

    print(f"\n  {'Question':<38} {'AnsRel':>7}  {'Faith':>6}  {'ToolOK':>7}")
    print(f"  {'─'*38} {'─'*7}  {'─'*6}  {'─'*7}")

    for r in results:
        ar = r.get("AnswerRelevancy", {})
        fa = r.get("Faithfulness", {})
        tc = r.get("ToolCorrectness", {})

        ar_str = f"{ar.get('score', '?'):.2f} {'✓' if ar.get('passed') else '✗'}" if ar.get('score') is not None else "  N/A"
        fa_str = f"{fa.get('score', '?'):.2f} {'✓' if fa.get('passed') else '✗'}" if fa.get('score') is not None else " N/A"
        tc_str = "✓" if tc.get("passed") else "✗"

        print(f"  {r['input'][:38]:<38} {ar_str:>7}  {fa_str:>6}  {tc_str:>7}")

    # Aggregate
    print(f"\n  Aggregate:")
    for name in metric_names:
        scores = [r[name]["score"] for r in results if r.get(name, {}).get("score") is not None]
        passes = [r[name]["passed"] for r in results if name in r]
        if scores:
            import statistics
            avg = statistics.mean(scores)
            pass_rate = sum(passes) / len(passes)
            print(f"    {name:<22} avg={avg:.3f}  pass_rate={pass_rate:.0%}")
        elif passes:
            pass_rate = sum(passes) / len(passes)
            print(f"    {name:<22} pass_rate={pass_rate:.0%}")


def print_ragas_vs_deepeval_comparison() -> None:
    print(f"\n{'─' * WIDTH}")
    print("RAGAS vs DeepEval — Framework Comparison")
    print(f"{'─' * WIDTH}")
    comparison = """
  Dimension          RAGAS                      DeepEval
  ─────────────────────────────────────────────────────────────────
  Paradigm           Metric-first (batch)       Test-case-first (pytest)
  Input format       Dataset (DataFrame)        LLMTestCase objects
  Output             Score per metric           Pass/Fail + score + reason
  CI integration     Needs wrapper              Native pytest, assert_test()
  Tool eval          No built-in                ToolCorrectnessMetric
  Custom metrics     Via custom scorers         Via BaseMetric subclass
  Best for           Batch analysis, trending   CI gates, regression tests
  ─────────────────────────────────────────────────────────────────
  Use RAGAS when:    Analyzing trends over time, comparing system versions
  Use DeepEval when: CI quality gates, per-test pass/fail reporting
"""
    print(comparison)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("DeepEval: Enterprise Agent Evaluation Suite")
    print(f"Test cases: {len(TEST_CASES_SPEC)}  |  Metrics: AnswerRelevancy, Faithfulness, ToolCorrectness")
    print("=" * WIDTH)

    print(f"\n{'─' * WIDTH}")
    print("Step 1: Build test cases (run agent)")
    print(f"{'─' * WIDTH}")
    cases = build_test_cases()

    print(f"\n{'─' * WIDTH}")
    print("Step 2: Evaluate with DeepEval metrics")
    print(f"{'─' * WIDTH}")
    results = run_deepeval_metrics(cases)

    print_results(results)
    print_ragas_vs_deepeval_comparison()

    print(f"{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
