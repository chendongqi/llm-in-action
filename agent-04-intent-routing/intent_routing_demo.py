"""
Agent 系列第五篇：意图识别与路由

演示内容：
  Demo 1 ▸ 关键词匹配 vs LLM 分类对比
          — 为什么关键词方案在生产中脆弱
  Demo 2 ▸ LangGraph 意图路由图
          — 4 个专项 Agent（搜索/代码/计算/问答）+ 澄清节点
  Demo 3 ▸ 置信度阈值与澄清机制
          — 置信度低时主动问用户，而不是猜
  Demo 4 ▸ 多轮对话意图跟踪
          — 历史上下文如何影响当前轮意图解读

运行前提：
  - conda activate llm_base
  - 在本目录创建 .env 文件，包含 LLM_API_KEY（见 .env.example）

代码结构：
  intent_routing_demo.py   ← 本文件
  requirements.txt
  .env.example
"""

import ast
import json
import operator
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

if not LLM_API_KEY:
    raise EnvironmentError("LLM_API_KEY 未设置，请创建 .env 文件")

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,   # type: ignore[arg-type]
    model=LLM_MODEL,
    temperature=0,
)

SEP  = "─" * 60
DSEP = "═" * 60


# ══════════════════════════════════════════════════════════════
# 意图类型与识别结果模型
# ══════════════════════════════════════════════════════════════

IntentType = Literal["search", "code", "calculate", "qa", "clarify"]


class IntentResult(BaseModel):
    """意图识别结果（Pydantic 结构化输出）。"""
    intent: IntentType = Field(description="用户意图：search/code/calculate/qa/clarify")
    confidence: float  = Field(ge=0.0, le=1.0, description="置信度，0-1 之间的浮点数")
    reasoning: str     = Field(description="识别理由，一句话说明为什么是这个意图")


# ══════════════════════════════════════════════════════════════
# Part 1：两种分类方案
# ══════════════════════════════════════════════════════════════

# 关键词规则表（生产中维护成本极高）
_KEYWORD_RULES: dict[str, list[str]] = {
    "search":    ["搜索", "查一下", "找找", "最新", "新闻", "资讯"],
    "code":      ["代码", "写一个", "函数", "实现", "bug", "报错", "报警"],
    "calculate": ["计算", "多少钱", "等于", "加", "减", "乘", "除", "换算"],
    "qa":        ["是什么", "为什么", "怎么", "介绍", "解释", "原理"],
}


def keyword_classify(text: str) -> str:
    """基于关键词的意图分类（简单快速，但适应性差）。"""
    scores: dict[str, int] = {k: 0 for k in _KEYWORD_RULES}
    for intent, keywords in _KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                scores[intent] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


def llm_classify(
    text: str,
    history: list[str] | None = None,
    confidence_threshold: float = 0.6,
) -> IntentResult:
    """基于 LLM 的意图分类，手动解析 JSON 输出，支持多轮历史上下文。

    GLM-4-Flash 的 with_structured_output JSON schema 模式有时只返回部分字段，
    改用手动 JSON 解析并加兜底逻辑，兼容性更好。
    """
    history_section = ""
    if history:
        recent = history[-4:]
        history_section = "\n\n最近对话历史：\n" + "\n".join(f"  {h}" for h in recent)

    system_prompt = f"""你是意图分类器，将用户输入分类到以下 5 种意图之一：

  search    — 搜索信息、查询最新动态、了解新闻资讯
  code      — 编写/调试/优化代码、问编程相关问题
  calculate — 数学计算、数值换算、算数题
  qa        — 知识问答、概念解释、原理介绍
  clarify   — 输入不清晰、指代不明，无法判断意图

分类规则：
1. 如果有对话历史，"优化一下""改一下""再来一个"等指令优先结合历史上下文判断指代的是什么
2. 置信度低于 {confidence_threshold} 时，统一返回 clarify，不要猜测{history_section}

必须只返回以下格式的 JSON，不要包含任何其他文字：
{{"intent": "<意图>", "confidence": <0到1的数字>, "reasoning": "<一句话理由>"}}"""

    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(text),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # 从响应中提取 JSON 对象（兼容模型在 JSON 前后附加文字的情况）
    try:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        data = json.loads(match.group() if match else raw)
        intent = data.get("intent", "clarify")
        if intent not in ("search", "code", "calculate", "qa", "clarify"):
            intent = "clarify"
        return IntentResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception:
        # 兜底：尝试从原始文本中提取意图关键词
        for candidate in ("search", "code", "calculate", "qa"):
            if candidate in raw.lower():
                return IntentResult(intent=candidate, confidence=0.5, reasoning="（JSON解析降级）")
        return IntentResult(intent="clarify", confidence=0.3, reasoning="（无法解析LLM输出）")


# ══════════════════════════════════════════════════════════════
# Part 2：专项 Agent 的工具
# ══════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """搜索互联网上的最新信息和新闻资讯。

    参数：
      query：搜索关键词，例如 "Python 3.13 新特性"

    返回：搜索结果摘要字符串。
    """
    _MOCK: dict[str, str] = {
        "python":    "Python 3.13 于 2024 年 10 月发布，主要新特性：free-threaded 模式（GIL 可选）和实验性 JIT 编译器。",
        "langchain": "LangChain 0.3 于 2024 年 9 月发布，重构核心模块，推荐用 LCEL 替代旧版 chain 接口。",
        "langgraph": "LangGraph 0.2 引入 functional API，支持更灵活的 Agent 流程编排。最新版支持 stream_events。",
        "agent":     "2025 年 Agent 工程化趋势：Harness Engineering 成为核心话题，MCP 协议生态快速壮大。",
        "claude":    "Claude 4 系列于 2025 年发布，包含 Haiku 4.5 / Sonnet 4.6 / Opus 4.7，显著提升推理和代码能力。",
    }
    for key, val in _MOCK.items():
        if key in query.lower():
            return val
    return f"关于 '{query}' 的搜索结果：找到 {len(query) * 7} 篇相关文章。主要涉及 {query} 的最新发展和实践案例。"


@tool
def run_code(code: str) -> str:
    """在沙盒中执行 Python 代码片段（模拟）。

    参数：
      code：要执行的 Python 代码

    返回：执行结果或错误信息。

    示例：
      run_code("def add(a, b): return a + b") → "函数 'add' 定义成功"
      run_code("1/0")                          → "执行错误：ZeroDivisionError"
    """
    # 模拟执行结果
    if "def " in code:
        match = re.search(r"def (\w+)\s*\(", code)
        name = match.group(1) if match else "函数"
        return f"函数 '{name}' 定义成功，语法检查通过。示例调用符合预期行为。"
    if "print(" in code:
        match = re.search(r'print\(f?["\'](.+?)["\']', code)
        if match:
            return f"执行成功。输出：{match.group(1)}"
    if "import" in code:
        return "模块导入成功。"
    if "class " in code:
        match = re.search(r"class (\w+)", code)
        name = match.group(1) if match else "类"
        return f"类 '{name}' 定义成功，语法检查通过。"
    return f"代码执行成功（沙盒模拟）。{len(code.splitlines())} 行，无语法错误。"


@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式，支持加减乘除和幂运算。

    参数：
      expression：数学表达式，例如 "(100 + 200) * 3" 或 "2 ** 10"

    返回：计算结果字符串。

    示例：
      calculator("2 ** 10")  → "2 ** 10 = 1024"
      calculator("1 / 0")    → "计算错误：除数为零"
    """
    _OPS: dict[type, Any] = {
        ast.Add:  operator.add,
        ast.Sub:  operator.sub,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
        ast.Pow:  operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.expr) -> float:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"不支持的常量类型：{type(node.value).__name__}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"不支持的表达式节点：{type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        # 整数结果去掉小数点
        formatted = f"{result:,.0f}" if result == int(result) else f"{result:,.4f}".rstrip("0")
        return f"{expression} = {formatted}"
    except ZeroDivisionError:
        return "计算错误：除数为零"
    except Exception as e:
        return f"计算失败：{e}"


@tool
def knowledge_base(question: str) -> str:
    """查询 AI/ML 领域知识库，获取概念解释和原理介绍。

    参数：
      question：问题描述，例如 "什么是 RAG"

    返回：相关知识摘要。
    """
    _KB: dict[str, str] = {
        "agent":       "AI Agent 是具备感知-决策-行动能力的自主程序。四要素：感知(Perception)/记忆(Memory)/决策(Reasoning)/行动(Action)。",
        "react":       "ReAct = Reasoning + Acting。通过 Thought→Action→Observation 循环让 LLM 解决复杂任务，是目前最主流的 Agent 范式。",
        "transformer": "Transformer 是 2017 年 Google 提出的注意力机制神经网络，《Attention is All You Need》，现代 LLM 的基础架构。",
        "rag":         "RAG（检索增强生成）= 从知识库检索相关文档 + 注入 Prompt + LLM 生成。核心价值：减少幻觉、注入实时知识。",
        "langgraph":   "LangGraph 是 LangChain 团队开发的 Agent 编排框架，用有向图表达工作流，原生支持循环、分支、并行和状态持久化。",
        "mcp":         "MCP（Model Context Protocol）是 Anthropic 提出的工具标准化协议，让 AI 助手以统一方式访问各种外部工具和数据源。",
        "plan":        "Plan-and-Solve 是一种 Agent 范式：先用 LLM 生成完整计划，再按计划逐步执行。适合有明确步骤的复杂任务。",
        "意图":        "意图识别（Intent Recognition）是 NLP 的核心任务，目的是从用户输入中提取其真实目的，通常分为分类器方案和路由方案两种实现。",
    }
    q_lower = question.lower()
    for key, val in _KB.items():
        if key in q_lower:
            return val
    return f"知识库中未找到关于 '{question}' 的直接条目。建议参考官方文档或学术论文获取最准确的信息。"


# ══════════════════════════════════════════════════════════════
# Part 3：LangGraph 意图路由图
# ══════════════════════════════════════════════════════════════

class RouterState(TypedDict):
    user_input:           str
    conversation_history: list[str]
    intent:               str
    confidence:           float
    reasoning:            str
    response:             str


def classify_node(state: RouterState) -> dict:
    """意图分类节点：识别用户意图，决定路由目标。"""
    result = llm_classify(
        state["user_input"],
        state.get("conversation_history"),
    )
    print(f"  [分类]  意图={result.intent}  置信度={result.confidence:.0%}")
    print(f"          理由：{result.reasoning}")
    return {
        "intent":     result.intent,
        "confidence": result.confidence,
        "reasoning":  result.reasoning,
    }


def route_by_intent(state: RouterState) -> str:
    """条件边函数：返回路由目标节点名。"""
    return state["intent"]


def _make_specialist_node(node_name: str, tools_list: list, system_text: str):
    """工厂函数：创建一个绑定了特定工具集和系统 Prompt 的专项 Agent 节点。"""
    specialist_llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,   # type: ignore[arg-type]
        model=LLM_MODEL,
        temperature=0,
    )
    specialist = create_react_agent(model=specialist_llm, tools=tools_list)

    def _node(state: RouterState) -> dict:
        print(f"  [路由]  → {node_name}")
        result = specialist.invoke(
            {"messages": [
                ("system", system_text),
                ("user",   state["user_input"]),
            ]},
            config={"recursion_limit": 6},
        )
        last = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        content = (last.content if isinstance(last.content, str) else str(last.content)) if last else "(无回答)"
        return {"response": content.strip()}

    _node.__name__ = node_name
    return _node


def clarify_node(state: RouterState) -> dict:
    """意图不明确时，生成一句友好的澄清问题。"""
    print("  [路由]  → clarify_agent（置信度不足，需澄清）")
    resp = llm.invoke([
        SystemMessage(
            "用户的表达不够清晰，你无法判断他想做什么。"
            "请用一句简洁友好的中文问句请用户说清楚他的需求。不要猜测，直接问。"
        ),
        HumanMessage(state["user_input"]),
    ])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"response": content.strip()}


def build_intent_router() -> Any:
    """构建并编译意图路由图。

    图结构：
      START → classify → [条件边：意图类型]
                            ├── search    → search_agent    → END
                            ├── code      → code_agent      → END
                            ├── calculate → calculator_agent → END
                            ├── qa        → qa_agent        → END
                            └── clarify   → clarify_agent   → END
    """
    graph = StateGraph(RouterState)

    graph.add_node("classify",  classify_node)
    graph.add_node("search",    _make_specialist_node(
        "search_agent",
        [web_search],
        "你是信息搜索专家。使用 web_search 工具查找最新信息，结果要简洁准确，不超过 3 句话。",
    ))
    graph.add_node("code",      _make_specialist_node(
        "code_agent",
        [run_code],
        "你是代码开发专家。帮用户编写、调试或优化代码，必要时用 run_code 工具验证。代码要有注释。",
    ))
    graph.add_node("calculate", _make_specialist_node(
        "calculator_agent",
        [calculator],
        "你是数学计算专家。使用 calculator 工具精确计算，展示计算过程，结果清晰易读。",
    ))
    graph.add_node("qa",        _make_specialist_node(
        "qa_agent",
        [knowledge_base],
        "你是知识问答专家。使用 knowledge_base 工具查询相关知识，深入解释概念，帮助用户真正理解。",
    ))
    graph.add_node("clarify",   clarify_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "search":    "search",
            "code":      "code",
            "calculate": "calculate",
            "qa":        "qa",
            "clarify":   "clarify",
        },
    )
    for node in ["search", "code", "calculate", "qa", "clarify"]:
        graph.add_edge(node, END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# Demo 函数
# ══════════════════════════════════════════════════════════════

def demo_keyword_vs_llm() -> None:
    print(f"\n\n{'#'*60}")
    print("# Demo 1 ▸ 关键词匹配 vs LLM 分类对比")
    print(f"{'#'*60}")
    print("  测试不同类型的用户输入，对比两种分类方案的结果差异\n")

    # 涵盖：清晰意图、模糊指代、话题切换、英文混合
    cases = [
        ("清晰-搜索",   "LangChain 最近发布了什么新版本？"),
        ("清晰-代码",   "帮我写个冒泡排序算法"),
        ("清晰-计算",   "100 美元换成人民币是多少"),
        ("清晰-问答",   "Transformer 模型的原理是什么"),
        ("模糊-歧义",   "优化一下"),          # 关键词：unknown；LLM：clarify
        ("模糊-指代",   "帮我处理一下它"),     # 指代完全不明
        ("自然表达",    "我想知道 RAG 到底解决了什么问题"),  # 无关键词命中
        ("口语化",      "2 加 3 乘以 4 等于多少"),           # 关键词：calculate / qa
    ]

    for label, text in cases:
        kw_result  = keyword_classify(text)
        llm_result = llm_classify(text)
        match_mark = " " if kw_result == llm_result.intent else "△"
        print(f"  {match_mark} [{label}]")
        print(f"    输入：{text}")
        print(f"    关键词 → {kw_result}")
        print(f"    LLM   → {llm_result.intent} ({llm_result.confidence:.0%})  {llm_result.reasoning}")
        print()

    print("  注：△ 表示关键词结果与 LLM 结果不一致")


def demo_intent_router() -> None:
    print(f"\n\n{'#'*60}")
    print("# Demo 2 ▸ LangGraph 意图路由实战")
    print(f"{'#'*60}")

    router = build_intent_router()

    cases = [
        ("信息搜索", "LangGraph 最近发布了什么新版本？"),
        ("代码帮助", "帮我写一个 Python 函数，计算列表的平均值，要能处理空列表"),
        ("数学计算", "计算 2 的 10 次方，再加上 (100 - 37) * 3 的结果"),
        ("知识问答", "RAG 是什么，它解决了 LLM 的哪些问题"),
        ("模糊输入", "帮我弄一下"),   # → clarify
    ]

    for label, user_input in cases:
        print(f"\n{DSEP}")
        print(f"  [{label}]  用户：{user_input}")
        print(SEP)

        result = router.invoke({
            "user_input":           user_input,
            "conversation_history": [],
            "intent":               "",
            "confidence":           0.0,
            "reasoning":            "",
            "response":             "",
        })

        print(f"\n  [最终回答]")
        for line in result["response"][:300].split("\n"):
            if line.strip():
                print(f"  {line.strip()}")


def demo_confidence_and_clarify() -> None:
    print(f"\n\n{'#'*60}")
    print("# Demo 3 ▸ 置信度阈值与澄清机制")
    print(f"{'#'*60}")
    print("  对比高/低置信度输入的路由行为：置信度低时主动问用户\n")

    router = build_intent_router()

    cases = [
        ("高置信-明确", "2 的 10 次方是多少"),
        ("高置信-明确", "Python 最新版本有什么新特性"),
        ("低置信-歧义", "帮我改一下"),
        ("低置信-指代", "再来一个"),
        ("低置信-残缺", "那个怎么弄"),
    ]

    for label, text in cases:
        result_classify = llm_classify(text)
        print(f"  [{label}]  输入：「{text}」")
        print(f"    置信度：{result_classify.confidence:.0%}  意图：{result_classify.intent}")
        print(f"    理由：{result_classify.reasoning}")

        if result_classify.intent == "clarify":
            # 触发澄清
            result_route = router.invoke({
                "user_input":           text,
                "conversation_history": [],
                "intent":               "",
                "confidence":           0.0,
                "reasoning":            "",
                "response":             "",
            })
            print(f"    澄清回答：{result_route['response']}")
        print()


def demo_multiturn_context() -> None:
    print(f"\n\n{'#'*60}")
    print("# Demo 4 ▸ 多轮对话意图跟踪")
    print(f"{'#'*60}")

    # ── 场景 A：代码对话历史 ────────────────────────────────────
    print("\n  === 场景 A：代码对话中的模糊后续指令 ===\n")

    code_history = [
        "用户：帮我写一个 Python 函数，计算列表的平均值",
        "助手：def average(lst): return sum(lst) / len(lst) if lst else 0.0",
        "用户：这个函数如果列表里有非数字怎么办",
        "助手：可以用 try/except TypeError 捕获，或者提前过滤非数值元素",
    ]

    print("  对话历史：")
    for h in code_history:
        print(f"    {h[:70]}")

    followups = [
        "优化一下",
        "加个类型注解",
        "换成英文注释",
    ]

    print(f"\n  {'─'*55}")
    for text in followups:
        r_no_hist   = llm_classify(text)
        r_with_hist = llm_classify(text, code_history)
        print(f"  输入：「{text}」")
        print(f"    ✗ 无历史 → {r_no_hist.intent} ({r_no_hist.confidence:.0%})  {r_no_hist.reasoning}")
        print(f"    ✓ 有历史 → {r_with_hist.intent} ({r_with_hist.confidence:.0%})  {r_with_hist.reasoning}")
        print()

    # ── 场景 B：计算对话历史 ────────────────────────────────────
    print("\n  === 场景 B：计算话题延续 ===\n")

    calc_history = [
        "用户：2 的 10 次方是多少",
        "助手：2 ** 10 = 1024",
    ]

    print("  对话历史：")
    for h in calc_history:
        print(f"    {h}")

    calc_followups = [
        "再乘以 3",       # 延续计算
        "加上 100 再除以 2",
    ]

    print(f"\n  {'─'*55}")
    for text in calc_followups:
        r_no_hist   = llm_classify(text)
        r_with_hist = llm_classify(text, calc_history)
        print(f"  输入：「{text}」")
        print(f"    ✗ 无历史 → {r_no_hist.intent} ({r_no_hist.confidence:.0%})")
        print(f"    ✓ 有历史 → {r_with_hist.intent} ({r_with_hist.confidence:.0%})  {r_with_hist.reasoning}")
        print()


def demo_full_session() -> None:
    print(f"\n\n{'#'*60}")
    print("# Demo 5 ▸ 完整多轮 Session（意图随对话动态变化）")
    print(f"{'#'*60}")

    router = build_intent_router()
    session_history: list[str] = []

    turns = [
        "LangGraph 是什么？",                              # qa
        "帮我用它写一个最简单的 Hello World Agent",        # code
        "优化一下，加上错误处理",                          # code（延续上一轮代码）
        "2 的 8 次方是多少",                               # calculate（话题切换）
        "再乘以 100",                                      # calculate（延续计算）
        "现在帮我查一下 LangGraph 最新的版本号",           # search（新话题）
    ]

    for i, user_input in enumerate(turns, 1):
        print(f"\n{SEP}")
        print(f"  Turn {i}  用户：{user_input}")

        result = router.invoke({
            "user_input":           user_input,
            "conversation_history": session_history.copy(),
            "intent":               "",
            "confidence":           0.0,
            "reasoning":            "",
            "response":             "",
        })

        response_short = result["response"][:120].replace("\n", " ")
        print(f"  → 意图：{result['intent']}  回答（节选）：{response_short}")

        # 更新 session 历史
        session_history.append(f"用户：{user_input}")
        session_history.append(f"助手：{result['response'][:80]}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(DSEP)
    print("  意图识别与路由 Demo")
    print("  模型：GLM-4-Flash（智谱 AI）")
    print(DSEP)

    demo_keyword_vs_llm()
    demo_intent_router()
    demo_confidence_and_clarify()
    demo_multiturn_context()
    demo_full_session()

    print(f"\n✓ 所有 Demo 运行完毕")
