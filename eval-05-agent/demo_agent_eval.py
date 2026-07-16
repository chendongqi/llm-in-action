"""
AI Eval Article 05 — Agent Evaluation: Tool Call Accuracy and Trajectory Quality

Evaluates a ReAct-style Agent on two dimensions:

  Dimension 1: Tool Call Accuracy (automatic, no LLM needed)
    - Tool name accuracy: did the Agent call the right tool?
    - Parameter accuracy: were the arguments correct?
    - Step efficiency: how many steps did the Agent take vs minimum needed?

  Dimension 2: Trajectory Quality (LLM-as-Judge)
    - Reasoning soundness: was each step logically justified?
    - No unnecessary calls: did the Agent avoid redundant tool invocations?
    - Error recovery: when a tool returned nothing, did the Agent adapt?

The Agent under test is a customer support assistant with 3 tools:
    search_faq(query)        → search product FAQ
    get_order_status(order_id) → fetch order status
    calculate_refund(amount, days_since_purchase) → compute refund amount

15 test cases with expected tool call sequences (ground truth).

Run:
    conda activate llm_base
    python demo_agent_eval.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()

WIDTH = 70

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.0,
)

# ── Tools under test ─────────────────────────────────────────────────────────

FAQ_DB = {
    "退款政策": "购买后 7 天内可全额退款，7-30 天退 50%，30 天后不退款。",
    "配送时间": "普通配送 3-5 个工作日，快递配送 1-2 个工作日。",
    "质保期": "所有产品提供 1 年质保，人为损坏不在质保范围内。",
    "支付方式": "支持微信支付、支付宝、银行卡、信用卡。",
    "发票": "购买后可在订单详情页申请电子发票，纸质发票需额外申请。",
    "会员权益": "黄金会员享受 9 折优惠，铂金会员享受 8.5 折，并有专属客服。",
}

ORDER_DB = {
    "ORD-001": {"status": "已发货", "item": "无线耳机", "amount": 299.0, "days_since": 3},
    "ORD-002": {"status": "待发货", "item": "智能手表", "amount": 899.0, "days_since": 1},
    "ORD-003": {"status": "已完成", "item": "蓝牙音箱", "amount": 199.0, "days_since": 45},
    "ORD-004": {"status": "已完成", "item": "充电宝", "amount": 99.0, "days_since": 15},
    "ORD-005": {"status": "已取消", "item": "数据线", "amount": 29.0, "days_since": 2},
}


@tool
def search_faq(query: str) -> str:
    """搜索产品常见问题（FAQ）。用于回答关于政策、服务、流程的一般性问题。"""
    query_lower = query.lower()
    for key, value in FAQ_DB.items():
        if key in query or any(w in query_lower for w in key.lower().split()):
            return f"{key}：{value}"
    return "未找到相关 FAQ，请联系客服。"


@tool
def get_order_status(order_id: str) -> str:
    """查询订单状态。需要提供订单号（格式：ORD-XXX）。"""
    if order_id in ORDER_DB:
        o = ORDER_DB[order_id]
        return (f"订单 {order_id}：{o['item']}，状态：{o['status']}，"
                f"金额：¥{o['amount']}，下单 {o['days_since']} 天前。")
    return f"未找到订单 {order_id}，请检查订单号是否正确。"


@tool
def calculate_refund(amount: float, days_since_purchase: int) -> str:
    """计算退款金额。根据退款政策：7天内全额，7-30天50%，30天后不退。"""
    if days_since_purchase <= 7:
        refund = amount
        policy = "7天内全额退款"
    elif days_since_purchase <= 30:
        refund = amount * 0.5
        policy = "7-30天退50%"
    else:
        refund = 0
        policy = "30天后不退款"
    return f"退款金额：¥{refund:.2f}（{policy}，原价 ¥{amount}，购买 {days_since_purchase} 天前）"


TOOLS = [search_faq, get_order_status, calculate_refund]
TOOL_MAP = {t.name: t for t in TOOLS}
bound_llm = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """你是一名客服助手，帮助用户解答问题和处理订单。

重要：你必须使用工具来回答用户问题，不要依靠自己的知识直接回答。
- 关于退款政策、配送时间、质保期、支付方式、发票、会员权益等问题 → 必须先调用 search_faq
- 关于订单状态的问题 → 必须先调用 get_order_status（需要订单号）
- 计算退款金额 → 必须调用 calculate_refund（需要金额和购买天数）

规则：不重复调用相同参数的工具。闲聊和问候可以直接回答，无需调用工具。"""


# ── Agent graph ───────────────────────────────────────────────────────────────

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


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


agent = build_agent()


def run_agent(user_input: str) -> tuple[str, list[dict]]:
    """Run agent and return (final_answer, tool_calls_made)."""
    state = agent.invoke(
        {"messages": [HumanMessage(content=user_input)], "steps": 0},
        config={"recursion_limit": 20},
    )
    tool_calls = []
    final_answer = ""
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
            elif msg.content:
                final_answer = str(msg.content)
    return final_answer, tool_calls


# ── Test cases with expected tool calls ──────────────────────────────────────

@dataclass
class TestCase:
    id: str
    question: str
    expected_tools: list[str]      # expected tool names in order
    expected_args: list[dict]      # expected args for each tool call
    expected_min_steps: int = 1    # minimum steps needed


TEST_CASES = [
    TestCase("T01", "你们的退款政策是什么？",
             ["search_faq"], [{"query": "退款"}], 1),
    TestCase("T02", "我的订单 ORD-001 发货了吗？",
             ["get_order_status"], [{"order_id": "ORD-001"}], 1),
    TestCase("T03", "配送需要多少天？",
             ["search_faq"], [{"query": "配送"}], 1),
    TestCase("T04", "ORD-004 这个订单能退款多少钱？",
             ["get_order_status", "calculate_refund"],
             [{"order_id": "ORD-004"}, {"amount": 99.0, "days_since_purchase": 15}], 2),
    TestCase("T05", "我买了 299 元的商品，3 天前购买，能全额退款吗？",
             ["calculate_refund"],
             [{"amount": 299.0, "days_since_purchase": 3}], 1),
    TestCase("T06", "ORD-003 这个订单还能退款吗？",
             ["get_order_status", "calculate_refund"],
             [{"order_id": "ORD-003"}, {"amount": 199.0, "days_since_purchase": 45}], 2),
    TestCase("T07", "支持哪些支付方式？",
             ["search_faq"], [{"query": "支付"}], 1),
    TestCase("T08", "质保期是多长时间？",
             ["search_faq"], [{"query": "质保"}], 1),
    TestCase("T09", "ORD-002 订单的状态是什么？",
             ["get_order_status"], [{"order_id": "ORD-002"}], 1),
    TestCase("T10", "我是会员，有什么优惠？",
             ["search_faq"], [{"query": "会员"}], 1),
    # Edge cases
    TestCase("T11", "我想查询一个不存在的订单 ORD-999",
             ["get_order_status"], [{"order_id": "ORD-999"}], 1),
    TestCase("T12", "ORD-001 能退款吗？", # needs order lookup first
             ["get_order_status", "calculate_refund"],
             [{"order_id": "ORD-001"}, {"amount": 299.0, "days_since_purchase": 3}], 2),
    TestCase("T13", "你好，请问有什么可以帮你的？",
             [], [], 0),  # no tools needed
    TestCase("T14", "ORD-005 这个已取消的订单还能退款吗？",
             ["get_order_status"],
             [{"order_id": "ORD-005"}], 1),
    TestCase("T15", "发票怎么申请？",
             ["search_faq"], [{"query": "发票"}], 1),
]


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    question: str
    actual_tools: list[str]
    expected_tools: list[str]
    tool_name_correct: bool
    param_match_score: float      # 0.0-1.0
    step_efficiency: float        # actual_steps / min_steps (lower = better)
    trajectory_score: float = 0.0
    answer: str = ""


def tool_name_accuracy(actual: list[str], expected: list[str]) -> bool:
    """Check if the sequence of tool names matches."""
    if len(actual) != len(expected):
        return False
    return all(a == e for a, e in zip(actual, expected))


def param_match_score(actual_calls: list[dict], expected_calls: list[dict]) -> float:
    """Score parameter accuracy: fraction of expected params correctly provided."""
    if not expected_calls:
        return 1.0 if not actual_calls else 0.0

    if len(actual_calls) != len(expected_calls):
        # Partial credit for correct calls
        min_len = min(len(actual_calls), len(expected_calls))
        actual_calls = actual_calls[:min_len]
        expected_calls = expected_calls[:min_len]

    total_params = 0
    correct_params = 0
    for actual, expected in zip(actual_calls, expected_calls):
        # expected_calls contains only args dicts (no "name" key)
        # actual_calls contains {"name": ..., "args": {...}}
        expected_args = expected if "name" not in expected else expected.get("args", {})
        actual_args = actual.get("args", {}) if isinstance(actual, dict) else {}
        for key, expected_val in expected_args.items():
            total_params += 1
            actual_val = actual_args.get(key)
            # Flexible matching: normalize strings, allow numeric tolerance
            if actual_val is None:
                continue
            if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                if abs(float(actual_val) - float(expected_val)) < 0.01:
                    correct_params += 1
            elif str(actual_val).strip() == str(expected_val).strip():
                correct_params += 1
            elif isinstance(expected_val, str) and expected_val.lower() in str(actual_val).lower():
                correct_params += 1

    return correct_params / total_params if total_params > 0 else 1.0


TRAJECTORY_JUDGE_PROMPT = """评估以下 Agent 执行轨迹的质量（1-5分）：

用户问题：{question}
Agent 调用的工具：{tool_calls}
Agent 的最终回答：{answer}

评分标准：
5分：工具调用完全合理，每步都有必要，回答准确完整
4分：工具调用基本合理，有1个小问题
3分：工具调用方向正确，但有冗余或遗漏
2分：工具调用部分错误，回答质量差
1分：工具调用完全错误或完全无关

只返回 1-5 的整数分数，不要其他内容。"""


def judge_trajectory(question: str, tool_calls: list[dict], answer: str) -> float:
    tools_str = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "无工具调用"
    prompt = TRAJECTORY_JUDGE_PROMPT.format(
        question=question, tool_calls=tools_str, answer=answer[:200]
    )
    result = str(llm.invoke([HumanMessage(content=prompt)]).content).strip()
    match = re.search(r'[1-5]', result)
    return float(match.group()) if match else 3.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Agent Evaluation: Tool Call Accuracy + Trajectory Quality")
    print(f"Agent: Customer Support  |  Test cases: {len(TEST_CASES)}")
    print("=" * WIDTH)

    results: list[EvalResult] = []

    print(f"\n{'─' * WIDTH}")
    print("Running agent on test cases...")
    print(f"{'─' * WIDTH}")

    for tc in TEST_CASES:
        answer, actual_calls = run_agent(tc.question)
        actual_tool_names = [c["name"] for c in actual_calls]

        name_correct = tool_name_accuracy(actual_tool_names, tc.expected_tools)
        param_score = param_match_score(actual_calls, tc.expected_args)
        actual_steps = len(actual_calls)
        efficiency = actual_steps / tc.expected_min_steps if tc.expected_min_steps > 0 else (1.0 if actual_steps == 0 else 2.0)

        result = EvalResult(
            case_id=tc.id,
            question=tc.question,
            actual_tools=actual_tool_names,
            expected_tools=tc.expected_tools,
            tool_name_correct=name_correct,
            param_match_score=param_score,
            step_efficiency=efficiency,
            answer=answer,
        )
        results.append(result)

        status = "✓" if name_correct and param_score >= 0.8 else "✗"
        print(f"  [{tc.id}] {status} tools={actual_tool_names} "
              f"name={'✓' if name_correct else '✗'} "
              f"params={param_score:.0%}")
        time.sleep(0.3)

    # Trajectory quality evaluation
    print(f"\n{'─' * WIDTH}")
    print("Evaluating trajectory quality (LLM-as-Judge)...")
    print(f"{'─' * WIDTH}")

    for result in results:
        score = judge_trajectory(result.question,
                                  [{"name": n} for n in result.actual_tools],
                                  result.answer)
        result.trajectory_score = score
        print(f"  [{result.case_id}] trajectory={score:.0f}/5")
        time.sleep(0.3)

    # Summary metrics
    print(f"\n{'=' * WIDTH}")
    print("Evaluation Summary")
    print(f"{'─' * WIDTH}")

    tool_name_acc = sum(1 for r in results if r.tool_name_correct) / len(results)
    avg_param_score = statistics.mean(r.param_match_score for r in results)
    avg_efficiency = statistics.mean(r.step_efficiency for r in results)
    avg_trajectory = statistics.mean(r.trajectory_score for r in results)

    print(f"\n  Tool Name Accuracy:    {tool_name_acc:.0%}  "
          f"({sum(1 for r in results if r.tool_name_correct)}/{len(results)} correct sequences)")
    print(f"  Parameter Accuracy:    {avg_param_score:.0%}  (avg across all tool calls)")
    print(f"  Step Efficiency:       {avg_efficiency:.2f}x  (1.0 = optimal, >1 = extra steps)")
    print(f"  Trajectory Quality:    {avg_trajectory:.2f}/5  (LLM-as-Judge)")

    # Failure analysis
    failures = [r for r in results if not r.tool_name_correct]
    if failures:
        print(f"\n  Failed cases ({len(failures)}):")
        for r in failures:
            print(f"    [{r.case_id}] expected={r.expected_tools} actual={r.actual_tools}")

    # Per-category breakdown
    print(f"\n  Per-category breakdown:")
    simple = [r for r in results if len(r.expected_tools) <= 1]
    multi  = [r for r in results if len(r.expected_tools) >= 2]
    edge   = [r for r in results if r.case_id in ("T11", "T13", "T14")]

    for label, group in [("Simple (1 tool)", simple), ("Multi-step (2 tools)", multi), ("Edge cases", edge)]:
        if group:
            acc = sum(1 for r in group if r.tool_name_correct) / len(group)
            traj = statistics.mean(r.trajectory_score for r in group)
            print(f"    {label:<24} tool_acc={acc:.0%}  trajectory={traj:.2f}/5")

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
