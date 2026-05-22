"""
Agent 系列第二篇：ReAct — Reasoning + Acting Loop

演示内容：
  1. Thought → Action → Observation 三元组循环
  2. 两个工具：calculator（纯数学）+ web_search（Bing 搜索）
  3. 四个 Demo 场景（从简单到复杂）
  4. max_iterations / recursion_limit 如何防止失控
  5. 完整的 Trace 可视化

运行前提：
  - conda activate llm_base
  - 在项目根目录创建 .env 文件，包含 LLM_API_KEY
    （使用智谱 GLM-4-Flash，申请地址：https://open.bigmodel.cn）

结构：
  react_agent.py   ← 本文件，所有逻辑在此
  requirements.txt
  .env.example
"""

import ast
import operator
import os
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ─── LLM 配置 ─────────────────────────────────────────────────────────────────
# 使用智谱 GLM-4-Flash（免费额度充足，速度快）
# 支持 OpenAI 兼容接口，切换其他 LLM 只需修改下面三行

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

# ─── 工具 1：安全计算器 ────────────────────────────────────────────────────────
# 用 AST 解析替代 eval()，只允许四则运算和幂次，杜绝代码注入

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
    """递归计算 AST 节点，只支持安全的数学运算。"""
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
      "25 * 47 + 1000"        → "2175.0"
      "(3.14 * 100) / 2"      → "157.0"
      "2 ** 10"               → "1024.0"
      "(1024 * 768) + (1920 * 1080)"  → "2860032"

    不支持函数（sin/cos/sqrt 等）和变量。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree.body)
        # 整数结果去掉小数点
        if result == int(result):
            return str(int(result))
        return f"{result:.6g}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"计算错误：{e}"


# ─── 工具 2：网络搜索（Bing） ──────────────────────────────────────────────────
# 使用 Bing 搜索结果页面，解析前 3 条摘要
# 注意：scraping 方式可能受 Bing 规则变动影响，生产环境建议换用 Bing Search API

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

    适合获取：最新数据、事实性信息、新闻、定义、历史记录等。
    不适合：需要登录才能访问的内容、实时价格（有延迟）。
    """
    try:
        url = f"https://www.bing.com/search?q={quote(query)}&setlang=zh-CN"
        resp = requests.get(url, headers=_BING_HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = []
        for li in soup.find_all("li", class_="b_algo")[:4]:
            # 取标题
            h2 = li.find("h2")
            title = h2.get_text(strip=True) if h2 else ""
            # 取摘要段落
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


# ─── Agent 构建 ────────────────────────────────────────────────────────────────

def build_agent():
    """返回一个带 calculator + web_search 两个工具的 ReAct Agent。"""
    return create_react_agent(
        model=llm,
        tools=[calculator, web_search],
    )


# ─── Trace 可视化 ──────────────────────────────────────────────────────────────

def print_trace(result: dict, title: str = "") -> None:
    """把 Agent 的消息序列打印成可读的 Thought/Action/Observation 格式。"""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    if title:
        print(f"  {title}")
    print(f"{'═' * 60}")

    step = 0
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n[用户提问]\n  {msg.content}")
            print(sep)

        elif isinstance(msg, AIMessage):
            # msg.content 在新版 langchain 中可能是 str 或 list[...]
            content_text = msg.content if isinstance(msg.content, str) else ""
            if msg.tool_calls:
                # 模型决定调用工具（Thought + Action）
                step += 1
                print(f"\n[步骤 {step}] THOUGHT → ACTION")
                if content_text.strip():
                    # 部分模型会把思考过程放在 content 里
                    print(f"  Thought : {content_text.strip()}")
                for tc in msg.tool_calls:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                    print(f"  Action  : {tc['name']}({args_str})")
            else:
                # 模型给出最终答案
                print(f"\n[最终答案]\n  {content_text.strip()}")

        elif isinstance(msg, ToolMessage):
            # 工具执行结果（Observation）
            obs = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"\n  Observation : {obs.strip()[:300]}")
            print(sep)

    print(f"\n{'═' * 60}\n")


# ─── Demo 场景 ─────────────────────────────────────────────────────────────────

DEMOS = [
    # Demo 1：纯计算（单工具，单步）
    # 目的：验证工具调用的基本链路
    {
        "title": "Demo 1 ▸ 纯计算（单工具·单步）",
        "question": "计算 (1024 * 768) + (1920 * 1080) 的结果，并告诉我这个数字",
    },

    # Demo 2：搜索 + 计算（多工具，多步）
    # 目的：看 Agent 如何自己决定先搜索再计算
    {
        "title": "Demo 2 ▸ 搜索 + 计算（多工具·多步）",
        "question": (
            "Python 编程语言和 JavaScript 各是哪年首次发布的？"
            "搜索一下，然后计算两者相差多少年。"
        ),
    },

    # Demo 3：多轮搜索（同一工具多次调用）
    # 目的：展示 Agent 能根据第一次结果决定第二次查什么
    {
        "title": "Demo 3 ▸ 多轮搜索（同一工具多次调用）",
        "question": (
            "先搜索一下北京的面积，再搜索上海的面积，"
            "最后计算北京比上海大多少平方公里。"
        ),
    },

    # Demo 4：不需要工具的问题
    # 目的：展示 Agent 知道什么时候不需要调工具
    {
        "title": "Demo 4 ▸ 无需工具（直接回答）",
        "question": "用一句话解释什么是 ReAct 范式",
    },
]


def run_demos():
    agent = build_agent()

    for demo in DEMOS:
        print(f"\n\n{'#' * 60}")
        print(f"# {demo['title']}")
        print(f"{'#' * 60}")
        print(f"问题：{demo['question']}")

        result = agent.invoke(
            {"messages": [("user", demo["question"])]},
            config={"recursion_limit": 20},  # 防止意外无限循环
        )
        print_trace(result, title=demo["title"])


# ─── recursion_limit 防护演示 ──────────────────────────────────────────────────

def run_limit_demo():
    """
    演示 recursion_limit 如何防止 Agent 失控。

    把 recursion_limit 设得极低（5），Agent 来不及完成任务就会被强制终止。
    这模拟了生产环境中的"安全网"——即使 Agent 因 Bug 陷入循环，也不会永远跑下去。
    """
    print(f"\n\n{'#' * 60}")
    print("# Demo 5 ▸ recursion_limit 防护（故意触发限制）")
    print(f"{'#' * 60}")

    agent = build_agent()
    question = (
        "先搜索 Python 发布年份，再搜索 Java 发布年份，"
        "然后搜索 C 语言发布年份，最后计算三者的年份之和"
    )
    print(f"问题：{question}")
    print(f"recursion_limit 设为 5（正常需要 ~10 步，必然触发限制）\n")

    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": 5},
        )
        # 如果没触发限制，打印最终结果
        print_trace(result, title="未触发限制（步骤数恰好在限制内）")

    except Exception as e:
        print(f"[已触发 recursion_limit]")
        print(f"  异常类型：{type(e).__name__}")
        print(f"  异常信息：{str(e)[:200]}")
        print()
        print("→ 结论：生产环境务必设置合理的 recursion_limit（建议 15~25）")
        print("→ 过低：合法任务被截断；过高：失控 Agent 消耗大量 Token")


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ReAct Agent Demo")
    print("  模型：GLM-4-Flash（智谱 AI）")
    print("  工具：calculator + web_search (Bing)")
    print("=" * 60)

    run_demos()
    run_limit_demo()

    print("\n✓ 所有 Demo 运行完毕")
