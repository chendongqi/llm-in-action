"""
Agent 系列第三篇：Plan-and-Solve — 先规划，再执行

演示内容：
  1. ReAct 贪心策略的局限：同一个复杂任务，ReAct 会走弯路
  2. Plan-and-Solve 两阶段架构：
       - Plan 阶段：LLM 生成完整的步骤列表（不执行）
       - Execute 阶段：按计划逐步执行，每步可用工具
  3. Replan 机制：某步骤失败或结果不符合预期时，重新规划剩余步骤
  4. 四个 Demo 场景（从简单到复杂）
  5. 对比实验：相同任务下 ReAct vs Plan-and-Solve 的执行轨迹差异

运行前提：
  - conda activate llm_base
  - 在项目根目录创建 .env 文件，包含 LLM_API_KEY
    （使用智谱 GLM-4-Flash，申请地址：https://open.bigmodel.cn）

结构：
  plan_and_solve_agent.py   ← 本文件，所有逻辑在此
  requirements.txt
  .env.example
"""

import ast
import json
import operator
import os
import textwrap
from typing import Any, Literal, TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from urllib.parse import quote

load_dotenv()

# ─── LLM 配置 ─────────────────────────────────────────────────────────────────

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

if not LLM_API_KEY:
    raise EnvironmentError(
        "LLM_API_KEY 未设置。请在项目根目录创建 .env 文件并写入：\n"
        "  LLM_API_KEY=your_api_key_here\n"
        "申请地址：https://open.bigmodel.cn"
    )

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,  # type: ignore[arg-type]
    model=LLM_MODEL,
    temperature=0,
)

# ─── 工具（复用 agent-01 的设计）────────────────────────────────────────────────

_SAFE_OPS: dict[type, Any] = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量类型：{type(node.value).__name__}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        return op_fn(_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op_fn(_eval_ast(node.operand))
    raise ValueError(f"不支持的 AST 节点：{type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持 +  -  *  /  **  %  以及括号。

    示例：
      "25 * 47 + 1000"  → "2175"
      "(3.14 * 100) / 2" → "157"
      "2 ** 10"         → "1024"

    不支持函数（sin/cos/sqrt 等）和变量。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree.body)
        if result == int(result):
            return str(int(result))
        return f"{result:.6g}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"计算错误：{e}"


_BING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@tool
def web_search(query: str) -> str:
    """搜索网络，返回最相关的 3 条摘要。

    适合获取：最新数据、事实性信息、新闻、定义等。
    """
    try:
        url = f"https://www.bing.com/search?q={quote(query)}&setlang=zh-CN"
        resp = requests.get(url, headers=_BING_HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = []
        for li in soup.find_all("li", class_="b_algo")[:4]:
            h2 = li.find("h2")
            title = h2.get_text(strip=True) if h2 else ""
            p = li.find("p")
            body = p.get_text(strip=True) if p else ""
            if title or body:
                snippets.append(f"• {title}: {body}"[:200])

        if not snippets:
            return "未找到相关结果，请换个关键词再试。"
        return "\n".join(snippets[:3])

    except requests.RequestException as e:
        return f"搜索请求失败：{e}"
    except Exception as e:
        return f"搜索解析错误：{e}"


TOOLS = [calculator, web_search]

# ─── Plan-and-Solve：状态定义 ──────────────────────────────────────────────────

class PlanSolveState(TypedDict):
    """Agent 的完整状态，贯穿整个图的执行过程。"""
    task: str                    # 用户原始任务
    plan: list[str]              # 当前计划（步骤列表）
    completed_steps: list[str]   # 已完成的步骤（含结果摘要）
    current_step_index: int      # 当前执行到第几步（0-based）
    step_result: str             # 本步骤的执行结果
    replan_count: int            # 已重新规划的次数
    final_answer: str            # 最终答案


# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """你是一个任务规划专家。用户会给你一个需要多步骤完成的任务。

你的工作：分析任务，制定一个清晰的、可执行的步骤计划。

规则：
1. 将任务分解为 3-7 个独立的步骤
2. 每个步骤必须是具体的、可操作的（可以用工具完成，或者是整合前面结果的总结步骤）
3. 步骤之间有明确的依赖关系（后面的步骤可以用前面步骤的结果）
4. 最后一步通常是"整合所有信息，给出最终答案"

输出格式（严格按照此格式，只输出步骤列表，不要其他内容）：
1. [步骤描述]
2. [步骤描述]
3. [步骤描述]
...
"""

EXECUTOR_SYSTEM = """你是一个任务执行专家。你需要执行计划中的某一个具体步骤。

你有以下工具可用：
- calculator：计算数学表达式
- web_search：搜索网络获取信息

执行规则：
1. 只执行当前指定的步骤，不要超前执行
2. 如果步骤需要用工具，就调用工具
3. 如果步骤是总结/整合步骤，根据已完成的步骤结果直接给出答案，不需要调用工具
4. 执行完毕后，给出简洁明确的结果说明

已完成的步骤（供参考）：
{completed_steps}

当前需要执行的步骤：
{current_step}
"""

REPLANNER_SYSTEM = """你是一个任务重新规划专家。

原始任务：{task}
原计划（已完成 {completed_count} 步）：
{original_plan}

已完成的步骤结果：
{completed_steps}

问题：{issue}

请根据已有的进展，为剩余工作制定新的计划。

规则：
1. 不要重复已完成的步骤
2. 直接从下一步开始规划
3. 步骤数量 2-5 个
4. 最后一步是给出最终答案

输出格式（只输出步骤列表）：
1. [步骤描述]
2. [步骤描述]
...
"""


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def parse_plan(text: str) -> list[str]:
    """从 LLM 输出中解析步骤列表。

    支持：
      "1. 步骤描述"
      "- 步骤描述"
      "步骤描述"（纯文本行）
    """
    steps = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉序号前缀：1. / 1) / - / * 等
        import re
        line = re.sub(r"^(\d+[\.\)]\s*|[-*]\s*)", "", line)
        if line:
            steps.append(line)
    return steps


def format_completed_steps(completed: list[str]) -> str:
    if not completed:
        return "（还没有已完成的步骤）"
    return "\n".join(f"  步骤 {i+1}：{s}" for i, s in enumerate(completed))


# ─── 图节点：Plan ──────────────────────────────────────────────────────────────

def plan_node(state: PlanSolveState) -> dict:
    """生成任务的完整执行计划。"""
    print(f"\n{'─'*50}")
    print("  [PLAN] 正在制定计划...")

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"任务：{state['task']}"),
    ]
    response = llm.invoke(messages)
    plan_text = response.content if isinstance(response.content, str) else ""
    plan = parse_plan(plan_text)

    print(f"  制定了 {len(plan)} 步计划：")
    for i, step in enumerate(plan):
        print(f"    {i+1}. {step}")

    return {
        "plan": plan,
        "current_step_index": 0,
        "completed_steps": [],
        "replan_count": 0,
        "final_answer": "",
    }


# ─── 图节点：Execute ───────────────────────────────────────────────────────────

def execute_node(state: PlanSolveState) -> dict:
    """执行当前步骤（可调用工具）。"""
    idx = state["current_step_index"]
    current_step = state["plan"][idx]
    total = len(state["plan"])

    print(f"\n{'─'*50}")
    print(f"  [EXECUTE] 步骤 {idx+1}/{total}：{current_step}")

    # 构建执行上下文
    completed_summary = format_completed_steps(state["completed_steps"])
    system_prompt = EXECUTOR_SYSTEM.format(
        completed_steps=completed_summary,
        current_step=current_step,
    )

    # 用 ReAct 子 Agent 执行单个步骤（可能需要工具）
    sub_agent = create_react_agent(model=llm, tools=TOOLS)
    result = sub_agent.invoke(
        {"messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请执行这个步骤：{current_step}"),
        ]},
        config={"recursion_limit": 8},
    )

    # 提取最终答案
    last_msg = result["messages"][-1]
    step_result = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
    # 清理 GLM 可能泄漏的 JSON
    if step_result.strip() and step_result.strip()[0] in ("{", "["):
        try:
            json.loads(step_result.strip())
            step_result = f"（步骤 {idx+1} 已执行，工具调用完成）"
        except json.JSONDecodeError:
            pass

    # 打印工具调用摘要
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                print(f"    → 调用工具：{tc['name']}({args_str})")
        elif hasattr(msg, "content") and not isinstance(msg, (HumanMessage, SystemMessage)):
            from langchain_core.messages import ToolMessage
            if isinstance(msg, ToolMessage):
                obs = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"    → 工具返回：{obs.strip()[:150]}")

    print(f"    → 步骤结果：{step_result.strip()[:200]}")

    # 更新已完成步骤
    new_completed = state["completed_steps"] + [
        f"{current_step} → {step_result.strip()[:100]}"
    ]

    return {
        "step_result": step_result,
        "completed_steps": new_completed,
        "current_step_index": idx + 1,
    }


# ─── 图节点：Replan ────────────────────────────────────────────────────────────

def replan_node(state: PlanSolveState) -> dict:
    """当步骤执行失败或结果异常时，重新规划剩余步骤。"""
    print(f"\n{'─'*50}")
    print(f"  [REPLAN] 第 {state['replan_count']+1} 次重新规划...")

    completed_count = state["current_step_index"]
    original_plan = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(state["plan"])
    )
    completed_summary = format_completed_steps(state["completed_steps"])

    messages = [
        SystemMessage(content=REPLANNER_SYSTEM.format(
            task=state["task"],
            completed_count=completed_count,
            original_plan=original_plan,
            completed_steps=completed_summary,
            issue=f"步骤 {completed_count} 的结果：{state['step_result'][:200]}",
        )),
        HumanMessage(content="请为剩余工作制定新计划。"),
    ]
    response = llm.invoke(messages)
    new_remaining = parse_plan(
        response.content if isinstance(response.content, str) else ""
    )

    # 重建完整计划：已完成的步骤 + 新的剩余步骤
    completed_steps_desc = [
        state["plan"][i] for i in range(completed_count)
    ]
    new_plan = completed_steps_desc + new_remaining

    print(f"  新计划（从步骤 {completed_count+1} 开始）：")
    for i, step in enumerate(new_remaining):
        print(f"    {completed_count + i + 1}. {step}")

    return {
        "plan": new_plan,
        "replan_count": state["replan_count"] + 1,
    }


# ─── 图节点：Finalize ─────────────────────────────────────────────────────────

def finalize_node(state: PlanSolveState) -> dict:
    """所有步骤执行完毕，提取最终答案。"""
    print(f"\n{'─'*50}")
    print("  [FINALIZE] 整合最终答案...")

    # 最后一步的结果就是最终答案
    final = state["step_result"] if state["step_result"] else "任务已完成"
    return {"final_answer": final}


# ─── 路由函数 ──────────────────────────────────────────────────────────────────

MAX_REPLAN = 2   # 最多重新规划 2 次


def should_continue(state: PlanSolveState) -> Literal["execute", "replan", "finalize"]:
    """决定执行完一步后的下一个动作。"""
    idx = state["current_step_index"]
    total = len(state["plan"])

    # 所有步骤执行完毕
    if idx >= total:
        return "finalize"

    # 检测步骤失败（工具报错）
    result = state.get("step_result", "")
    failed = any(kw in result for kw in ["计算错误", "搜索请求失败", "搜索解析错误", "Error"])

    if failed and state["replan_count"] < MAX_REPLAN:
        return "replan"

    return "execute"


def after_replan(state: PlanSolveState) -> Literal["execute", "finalize"]:
    """重新规划后，继续执行（或步骤数为0时结束）。"""
    idx = state["current_step_index"]
    total = len(state["plan"])
    if idx >= total:
        return "finalize"
    return "execute"


# ─── 构建图 ────────────────────────────────────────────────────────────────────

def build_plan_and_solve_agent():
    """构建 Plan-and-Solve Agent 图。"""
    graph = StateGraph(PlanSolveState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        should_continue,
        {"execute": "execute", "replan": "replan", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "replan",
        after_replan,
        {"execute": "execute", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ─── Demo 场景 ─────────────────────────────────────────────────────────────────

DEMOS = [
    # Demo 1：多步搜索 + 计算
    {
        "title": "Demo 1 ▸ 多国数据收集与对比计算",
        "task": (
            "搜索中国、美国、印度三个国家的人口数量，"
            "然后计算三国人口总和，以及中国人口占三国总和的百分比。"
        ),
    },

    # Demo 2：有明确依赖关系的多步任务
    {
        "title": "Demo 2 ▸ 依赖链任务（后步骤用前步骤结果）",
        "task": (
            "搜索目前最新的 iPhone 旗舰机型和它的起售价（美元），"
            "再搜索当前美元对人民币的汇率，"
            "最后计算这款 iPhone 折合人民币大约多少钱。"
        ),
    },

    # Demo 3：简单任务（验证计划不会过度复杂化）
    {
        "title": "Demo 3 ▸ 简单任务（验证计划简洁性）",
        "task": "计算 2 的 10 次方，再加上 3 的 5 次方，结果是多少？",
    },

    # Demo 4：开放式研究任务
    {
        "title": "Demo 4 ▸ 开放式研究任务",
        "task": (
            "我想了解 LangGraph 这个框架：它是什么、主要用途是什么、"
            "和 LangChain 有什么关系？请搜索后给我一个简洁的总结。"
        ),
    },
]


def run_plan_and_solve_demo(demo: dict) -> dict:
    """运行单个 Plan-and-Solve Demo。"""
    print(f"\n\n{'#'*60}")
    print(f"# {demo['title']}")
    print(f"{'#'*60}")
    print(f"任务：{demo['task']}")

    agent = build_plan_and_solve_agent()

    initial_state: PlanSolveState = {
        "task": demo["task"],
        "plan": [],
        "completed_steps": [],
        "current_step_index": 0,
        "step_result": "",
        "replan_count": 0,
        "final_answer": "",
    }

    result = agent.invoke(initial_state)

    print(f"\n{'═'*60}")
    print("  最终答案")
    print(f"{'═'*60}")
    answer = result.get("final_answer", "")
    for line in textwrap.wrap(answer, width=56):
        print(f"  {line}")
    print(f"{'═'*60}")

    return result


# ─── 对比实验：ReAct vs Plan-and-Solve ────────────────────────────────────────

def run_comparison_demo():
    """
    对比实验：相同任务，分别用 ReAct 和 Plan-and-Solve 执行。

    观察点：
    - ReAct 直接从第一步开始行动，可能走弯路
    - Plan-and-Solve 先制定完整计划，再按计划执行
    """
    task = (
        "搜索 Python、Java、Go 三种编程语言各自的首次发布年份，"
        "然后按从早到晚排序，并计算 Python 和 Go 相差多少年。"
    )

    print(f"\n\n{'#'*60}")
    print("# Demo 5 ▸ 对比实验：ReAct vs Plan-and-Solve")
    print(f"{'#'*60}")
    print(f"任务：{task}")

    # ── ReAct ──
    print(f"\n{'─'*60}")
    print("  方式 A：ReAct（贪心执行）")
    print(f"{'─'*60}")

    react_agent = create_react_agent(model=llm, tools=TOOLS)
    react_result = react_agent.invoke(
        {"messages": [("user", task)]},
        config={"recursion_limit": 20},
    )

    step_count = sum(
        1 for m in react_result["messages"]
        if isinstance(m, AIMessage) and m.tool_calls
    )
    final_react = react_result["messages"][-1].content
    if isinstance(final_react, list):
        final_react = ""
    print(f"  工具调用步数：{step_count}")
    print(f"  最终答案：{str(final_react).strip()[:300]}")

    # ── Plan-and-Solve ──
    print(f"\n{'─'*60}")
    print("  方式 B：Plan-and-Solve（先规划，再执行）")
    print(f"{'─'*60}")

    ps_agent = build_plan_and_solve_agent()
    ps_result = ps_agent.invoke({
        "task": task,
        "plan": [],
        "completed_steps": [],
        "current_step_index": 0,
        "step_result": "",
        "replan_count": 0,
        "final_answer": "",
    })

    print(f"\n  Plan-and-Solve 最终答案：")
    print(f"  {ps_result.get('final_answer', '').strip()[:300]}")

    print(f"\n{'─'*60}")
    print("  对比总结：")
    print(f"    ReAct 工具调用步数：{step_count} 步（贪心，无预先规划）")
    print(f"    Plan-and-Solve：先规划 {len(ps_result.get('plan', []))} 步，再按计划执行")
    print("    → 简单任务：ReAct 更快；复杂多步任务：Plan-and-Solve 路径更清晰")


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Plan-and-Solve Agent Demo")
    print("  模型：GLM-4-Flash（智谱 AI）")
    print("  工具：calculator + web_search (Bing)")
    print("  架构：Plan → Execute → (Replan) → Finalize")
    print("=" * 60)

    # 运行 4 个 Demo
    for demo in DEMOS:
        run_plan_and_solve_demo(demo)

    # 运行对比实验
    run_comparison_demo()

    print("\n✓ 所有 Demo 运行完毕")
