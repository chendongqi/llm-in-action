"""
Agent 系列第四篇：Tool Calling 深度解析

演示内容：
  1. 工具设计三要素：接口（docstring）、验证（Pydantic）、安全（边界检查）
  2. 好工具 vs 坏工具：对比实验 —— 相同功能，不同设计，Agent 行为截然不同
  3. Pydantic 参数校验：让错误在工具入口拦截，而不是在执行中崩溃
  4. 并行工具调用：LangGraph 的内置支持，多工具同时执行
  5. 工具安全三大威胁：路径遍历、命令注入、提示词注入
  6. 工具错误分类：可重试 vs 不可重试，如何影响 Agent 行为

运行前提：
  - conda activate llm_base
  - 在项目根目录创建 .env 文件，包含 LLM_API_KEY

结构：
  tool_calling_demo.py   ← 本文件
  requirements.txt
  .env.example
"""

import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from pydantic import BaseModel, Field, field_validator

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
# Part 1：坏工具 vs 好工具  （相同功能，对比设计质量）
# ══════════════════════════════════════════════════════════════

# ── 1A：坏工具示例 ─────────────────────────────────────────────
# 三个问题：文档不清晰、不验证输入、崩溃时直接抛异常

@tool
def bad_stock_tool(x: str) -> str:
    """Get stock info."""   # ← 文档过于简陋：参数含义、返回格式、示例全无
    # 无输入验证，直接用
    _MOCK_STOCKS = {"AAPL": 189.5, "GOOGL": 175.2, "MSFT": 420.3}
    price = _MOCK_STOCKS[x]      # ← KeyError 直接崩溃，不捕获
    return f"{price}"            # ← 只返回数字，没有单位、没有货币、没有上下文


# ── 1B：好工具示例 ─────────────────────────────────────────────
# 三个改进：清晰文档、输入验证、结构化错误返回

_MOCK_STOCKS: dict[str, dict[str, Any]] = {
    "AAPL":  {"price": 189.50, "currency": "USD", "name": "Apple Inc.",     "change_pct": +1.23},
    "GOOGL": {"price": 175.20, "currency": "USD", "name": "Alphabet Inc.",  "change_pct": -0.45},
    "MSFT":  {"price": 420.30, "currency": "USD", "name": "Microsoft Corp.","change_pct": +0.87},
    "TSLA":  {"price": 248.10, "currency": "USD", "name": "Tesla Inc.",      "change_pct": -2.10},
    "BABA":  {"price": 78.60,  "currency": "USD", "name": "Alibaba Group",  "change_pct": +0.32},
}

@tool
def get_stock_price(symbol: str) -> str:
    """查询股票的当前价格和涨跌幅。

    参数：
      symbol：股票代码，大写字母，例如 "AAPL"、"GOOGL"、"MSFT"、"TSLA"、"BABA"

    返回：
      包含股票名称、当前价格（美元）、今日涨跌幅的字符串。
      如果代码不存在，返回错误说明。

    示例：
      get_stock_price("AAPL")  → "Apple Inc. (AAPL): $189.50 USD, 今日 +1.23%"
      get_stock_price("UNKNOWN") → "未找到股票代码 UNKNOWN，支持的代码：..."
    """
    symbol = symbol.strip().upper()
    if not re.match(r"^[A-Z]{1,5}$", symbol):
        return f"无效的股票代码格式：{symbol!r}。代码应为 1-5 个大写字母。"

    info = _MOCK_STOCKS.get(symbol)
    if info is None:
        supported = "、".join(_MOCK_STOCKS.keys())
        return f"未找到股票代码 {symbol}。当前支持：{supported}"

    sign = "+" if info["change_pct"] >= 0 else ""
    return (
        f"{info['name']} ({symbol}): "
        f"${info['price']:.2f} {info['currency']}，"
        f"今日 {sign}{info['change_pct']:.2f}%"
    )


# ══════════════════════════════════════════════════════════════
# Part 2：Pydantic 参数校验
# ══════════════════════════════════════════════════════════════

# 当工具需要多个参数或复杂约束时，用 Pydantic 的 BaseModel 做输入模型

_EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "CNY": 7.25,
    "EUR": 0.92,
    "JPY": 155.0,
    "GBP": 0.79,
    "HKD": 7.83,
}

SUPPORTED_CURRENCIES = list(_EXCHANGE_RATES.keys())


class CurrencyConvertInput(BaseModel):
    """货币换算的输入参数（Pydantic 自动校验）。"""
    amount: float = Field(
        ...,
        gt=0,
        le=1_000_000_000,
        description="要换算的金额，必须是正数，上限 10 亿",
    )
    from_currency: str = Field(
        ...,
        description=f"源货币代码，支持：{SUPPORTED_CURRENCIES}",
    )
    to_currency: str = Field(
        ...,
        description=f"目标货币代码，支持：{SUPPORTED_CURRENCIES}",
    )

    @field_validator("from_currency", "to_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        code = v.strip().upper()
        if code not in _EXCHANGE_RATES:
            raise ValueError(
                f"不支持的货币代码：{code!r}。"
                f"支持的货币：{SUPPORTED_CURRENCIES}"
            )
        return code


@tool(args_schema=CurrencyConvertInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """将指定金额从一种货币换算为另一种货币。

    参数：
      amount：金额（正数，最大 10 亿）
      from_currency：源货币代码，如 "USD"、"CNY"、"EUR"
      to_currency：目标货币代码

    返回：换算结果和汇率信息。

    示例：
      convert_currency(100, "USD", "CNY")   → "100.00 USD = 725.00 CNY（汇率：7.25）"
      convert_currency(-5, "USD", "CNY")    → Pydantic 自动拦截，返回验证错误
      convert_currency(100, "USD", "BTC")   → Pydantic 自动拦截，返回验证错误
    """
    rate = _EXCHANGE_RATES[to_currency] / _EXCHANGE_RATES[from_currency]
    result = amount * rate
    return (
        f"{amount:,.2f} {from_currency} = {result:,.2f} {to_currency}"
        f"（参考汇率：1 {from_currency} ≈ {rate:.4f} {to_currency}）"
    )


# ══════════════════════════════════════════════════════════════
# Part 3：并行工具调用
# ══════════════════════════════════════════════════════════════

# LangGraph 支持 LLM 在一次响应中调用多个工具（parallel tool calls）
# 这些调用会同时执行，显著减少等待时间

_MOCK_WEATHER: dict[str, dict[str, Any]] = {
    "北京":    {"temp": 28, "condition": "晴", "humidity": 45, "wind": "东北风 3级"},
    "上海":    {"temp": 32, "condition": "多云", "humidity": 78, "wind": "东南风 2级"},
    "广州":    {"temp": 35, "condition": "雷阵雨", "humidity": 90, "wind": "南风 4级"},
    "深圳":    {"temp": 34, "condition": "阴", "humidity": 85, "wind": "东风 2级"},
    "成都":    {"temp": 26, "condition": "小雨", "humidity": 80, "wind": "西南风 1级"},
    "beijing": {"temp": 28, "condition": "Sunny", "humidity": 45, "wind": "NE 3"},
    "shanghai":{"temp": 32, "condition": "Cloudy","humidity": 78, "wind": "SE 2"},
}

@tool
def get_weather(city: str) -> str:
    """查询城市的当前天气状况。

    参数：
      city：城市名称，支持中文或英文，例如 "北京"、"上海"、"成都"

    返回：包含温度、天气状况、湿度和风力的字符串。
    """
    key = city.strip().lower()
    # 模糊匹配
    for k, v in _MOCK_WEATHER.items():
        if k.lower() in key or key in k.lower():
            return (
                f"{city} 当前天气：{v['condition']}，"
                f"气温 {v['temp']}°C，湿度 {v['humidity']}%，{v['wind']}"
            )
    return f"暂无 {city} 的天气数据。支持城市：{list(_MOCK_WEATHER.keys())[:5]}"

@tool
def get_air_quality(city: str) -> str:
    """查询城市的空气质量指数（AQI）。

    参数：
      city：城市名称

    返回：AQI 数值和空气质量等级。
    """
    _MOCK_AQI = {
        "北京": {"aqi": 85,  "level": "良"},
        "上海": {"aqi": 62,  "level": "良"},
        "广州": {"aqi": 45,  "level": "优"},
        "深圳": {"aqi": 55,  "level": "良"},
        "成都": {"aqi": 110, "level": "轻度污染"},
    }
    for k, v in _MOCK_AQI.items():
        if k in city or city in k:
            return f"{city} 空气质量：AQI {v['aqi']}，{v['level']}"
    return f"暂无 {city} 的空气质量数据"


# ══════════════════════════════════════════════════════════════
# Part 4：工具安全实战
# ══════════════════════════════════════════════════════════════

# ── 4A：路径遍历防护 ───────────────────────────────────────────

# 演示用的沙盒目录（只允许读取这个目录下的文件）
_SANDBOX_DIR = Path("/tmp/agent_sandbox")
_SANDBOX_DIR.mkdir(exist_ok=True)
# 预置几个演示文件
(_SANDBOX_DIR / "report.txt").write_text("Q1 销售报告：总营收 1200 万元，同比增长 15%。")
(_SANDBOX_DIR / "config.json").write_text('{"version": "1.0", "env": "demo"}')
(_SANDBOX_DIR / "notes.md").write_text("# 会议记录\n- 讨论了 Q2 目标\n- 确认了发布时间表")


@tool
def read_file(filename: str) -> str:
    """读取沙盒目录中的文件内容。

    参数：
      filename：文件名（仅文件名，不含路径，如 "report.txt"）

    安全说明：
      只能读取预设沙盒目录（/tmp/agent_sandbox/）中的文件。
      禁止使用 ../ 等路径遍历字符。

    示例：
      read_file("report.txt")   → 返回文件内容
      read_file("../etc/passwd") → 返回安全错误
    """
    # 安全检查 1：拒绝路径遍历字符
    if any(char in filename for char in ["../", "..", "/", "\\"]):
        return f"安全拒绝：文件名不允许包含路径字符（{filename!r}）"

    # 安全检查 2：只允许字母、数字、点、下划线、连字符
    if not re.match(r"^[\w.\-]+$", filename):
        return f"安全拒绝：无效的文件名格式（{filename!r}）"

    target = _SANDBOX_DIR / filename

    # 安全检查 3：最终路径必须在沙盒目录内（防止符号链接攻击）
    try:
        target.resolve().relative_to(_SANDBOX_DIR.resolve())
    except ValueError:
        return f"安全拒绝：文件路径超出沙盒范围"

    if not target.exists():
        available = [f.name for f in _SANDBOX_DIR.iterdir()]
        return f"文件不存在：{filename}。沙盒中的文件：{available}"

    return target.read_text(encoding="utf-8")


# ── 4B：命令注入防护 ───────────────────────────────────────────

@tool
def lookup_user(user_id: str) -> str:
    """在用户数据库中查询用户信息（模拟）。

    参数：
      user_id：用户 ID，格式为纯数字，例如 "12345"

    安全说明：
      只接受纯数字 ID，防止 SQL 注入和命令注入。
    """
    # 严格验证：只允许纯数字，拒绝任何特殊字符
    if not re.match(r"^\d{1,10}$", user_id):
        return (
            f"安全拒绝：user_id 必须是 1-10 位纯数字，"
            f"收到的输入：{user_id!r}"
        )

    # 模拟数据库查询（绝不拼接用户输入到 SQL 字符串）
    _MOCK_USERS = {
        "10001": {"name": "张三", "role": "admin", "dept": "工程"},
        "10002": {"name": "李四", "role": "user",  "dept": "产品"},
        "10003": {"name": "王五", "role": "user",  "dept": "设计"},
    }

    user = _MOCK_USERS.get(user_id)
    if not user:
        return f"未找到用户 ID：{user_id}"

    return f"用户 {user_id}：{user['name']}，角色：{user['role']}，部门：{user['dept']}"


# ── 4C：工具调用频率限制（Rate Limiting）──────────────────────

class _RateLimiter:
    """简单的令牌桶限流器（每分钟最多调用 N 次）。"""
    def __init__(self, max_calls: int, window_seconds: int = 60):
        self._max = max_calls
        self._window = window_seconds
        self._calls: list[float] = []

    def allow(self) -> bool:
        now = time.time()
        self._calls = [t for t in self._calls if now - t < self._window]
        if len(self._calls) >= self._max:
            return False
        self._calls.append(now)
        return True

    def wait_seconds(self) -> float:
        if not self._calls:
            return 0.0
        oldest = min(self._calls)
        return max(0.0, self._window - (time.time() - oldest))


_search_limiter = _RateLimiter(max_calls=10, window_seconds=60)


@tool
def rate_limited_search(query: str) -> str:
    """带频率限制的搜索工具（每分钟最多调用 10 次）。

    参数：
      query：搜索关键词

    频率限制：每 60 秒最多调用 10 次，超出后返回限流提示。
    """
    if not _search_limiter.allow():
        wait = _search_limiter.wait_seconds()
        return f"调用频率超限（每分钟最多 10 次）。请等待约 {wait:.0f} 秒后重试。"

    # 模拟搜索结果
    return f"搜索 '{query}' 的结果（模拟）：关于 {query} 的最新信息..."


# ══════════════════════════════════════════════════════════════
# Part 5：工具错误分类
# ══════════════════════════════════════════════════════════════

# 工具错误分两大类：
#   - 可重试：网络超时、临时服务不可用 → 返回 "RETRY: ..." 前缀
#   - 不可重试：参数错误、权限拒绝  → 返回普通错误信息

@tool
def fetch_report(report_id: str, retry_simulation: bool = False) -> str:
    """获取指定 ID 的报告（演示错误分类）。

    参数：
      report_id：报告 ID，格式 RPT-NNNN
      retry_simulation：设为 True 时模拟可重试的网络错误（用于 Demo）

    返回：
      正常：报告内容
      可重试错误：以 "RETRY:" 开头
      不可重试错误：以 "ERROR:" 开头
    """
    if not re.match(r"^RPT-\d{4}$", report_id):
        return f"ERROR: 报告 ID 格式无效（{report_id!r}），应为 RPT-XXXX 格式"

    if retry_simulation:
        return "RETRY: 服务暂时不可用（HTTP 503），请稍后重试"

    _MOCK_REPORTS = {
        "RPT-0001": "2024年Q4财务报告：营收 8.2亿，净利润 1.1亿，同比增长 23%",
        "RPT-0002": "2025年产品路线图：Q1发布 v2.0，Q3发布移动端，Q4国际化",
        "RPT-0003": "技术债务评估：高优先级问题 12 个，中优先级 38 个，预计修复周期 2 季度",
    }

    report = _MOCK_REPORTS.get(report_id)
    if not report:
        return f"ERROR: 未找到报告 {report_id}，有效 ID：{list(_MOCK_REPORTS.keys())}"

    return f"报告 {report_id}：{report}"


# ══════════════════════════════════════════════════════════════
# Trace 打印工具
# ══════════════════════════════════════════════════════════════

def print_trace(result: dict, title: str = "") -> None:
    print(f"\n{DSEP}")
    if title:
        print(f"  {title}")
    print(DSEP)
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n[用户]\n  {msg.content}")
            print(SEP)
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    print(f"\n[工具调用]  {tc['name']}({args_str})")
            else:
                if content.strip():
                    print(f"\n[最终答案]\n  {content.strip()}")
        elif isinstance(msg, ToolMessage):
            obs = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"[工具返回]  {obs.strip()[:200]}")
    print(f"\n{DSEP}\n")


# ══════════════════════════════════════════════════════════════
# Demo 1：好工具 vs 坏工具
# ══════════════════════════════════════════════════════════════

def demo_good_vs_bad():
    print(f"\n\n{'#'*60}")
    print("# Demo 1 ▸ 好工具 vs 坏工具（相同任务，不同设计）")
    print(f"{'#'*60}")
    question = "帮我查一下 AAPL 和一个不存在的股票 XYZ999 的价格"

    print(f"\n── 方式 A：坏工具（文档差 + 无错误处理）──")
    print(f"提问：{question}")
    try:
        agent_bad = create_react_agent(model=llm, tools=[bad_stock_tool])
        result = agent_bad.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": 10},
        )
        print_trace(result, "坏工具执行结果")
    except Exception as e:
        print(f"\n  ⚠ Agent 崩溃：{type(e).__name__}: {e}")
        print("  → 坏工具直接抛 KeyError，Agent 无法继续")

    print(f"\n── 方式 B：好工具（清晰文档 + 优雅错误处理）──")
    agent_good = create_react_agent(model=llm, tools=[get_stock_price])
    result = agent_good.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 10},
    )
    print_trace(result, "好工具执行结果")


# ══════════════════════════════════════════════════════════════
# Demo 2：Pydantic 参数校验拦截无效输入
# ══════════════════════════════════════════════════════════════

def demo_pydantic_validation():
    print(f"\n\n{'#'*60}")
    print("# Demo 2 ▸ Pydantic 校验：在工具入口拦截错误")
    print(f"{'#'*60}")

    agent = create_react_agent(model=llm, tools=[convert_currency])

    cases = [
        ("正常请求", "帮我把 1000 美元换算成人民币"),
        ("负数金额", "帮我把 -500 USD 换算成 CNY"),
        ("不支持的货币", "帮我把 100 USD 换成 BTC（比特币）"),
    ]

    for label, q in cases:
        print(f"\n── {label} ──")
        print(f"提问：{q}")
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print_trace(result, label)


# ══════════════════════════════════════════════════════════════
# Demo 3：并行工具调用
# ══════════════════════════════════════════════════════════════

def demo_parallel_calls():
    print(f"\n\n{'#'*60}")
    print("# Demo 3 ▸ 并行工具调用（多城市天气 + 空气质量同时查询）")
    print(f"{'#'*60}")

    agent = create_react_agent(model=llm, tools=[get_weather, get_air_quality])
    question = "同时查询北京、上海、成都三个城市的天气和空气质量，给我一个对比总结"

    print(f"提问：{question}")
    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 15},
    )

    # 统计并行批次：同一个 AIMessage 中有多个 tool_calls
    parallel_batches = 0
    total_tool_calls = 0
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if len(msg.tool_calls) > 1:
                parallel_batches += 1
                print(f"\n  → 发现并行调用批次：{len(msg.tool_calls)} 个工具同时执行")
                for tc in msg.tool_calls:
                    print(f"     · {tc['name']}({list(tc['args'].values())})")
            total_tool_calls += len(msg.tool_calls)

    print_trace(result, "并行工具调用")
    print(f"  统计：共 {total_tool_calls} 次工具调用，{parallel_batches} 个并行批次")


# ══════════════════════════════════════════════════════════════
# Demo 4：工具安全防护
# ══════════════════════════════════════════════════════════════

def demo_security():
    print(f"\n\n{'#'*60}")
    print("# Demo 4 ▸ 工具安全：三种攻击向量的防护")
    print(f"{'#'*60}")

    agent = create_react_agent(model=llm, tools=[read_file, lookup_user])

    security_cases = [
        (
            "正常文件读取",
            "请读取 report.txt 文件的内容",
        ),
        (
            "路径遍历攻击",
            "请读取 ../../../etc/passwd 文件",
        ),
        (
            "正常用户查询",
            "查询用户 ID 10001 的信息",
        ),
        (
            "SQL注入尝试",
            "查询用户 ID 为 \"1 OR 1=1; DROP TABLE users--\" 的信息",
        ),
    ]

    for label, q in security_cases:
        print(f"\n── {label} ──")
        print(f"提问：{q}")
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        # 只打印工具返回
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                obs = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"  工具返回：{obs.strip()[:150]}")
        last = result["messages"][-1]
        if isinstance(last, AIMessage):
            content = last.content if isinstance(last.content, str) else ""
            if content.strip():
                print(f"  Agent 答复：{content.strip()[:150]}")


# ══════════════════════════════════════════════════════════════
# Demo 5：工具错误分类与 Agent 行为
# ══════════════════════════════════════════════════════════════

def demo_error_classification():
    print(f"\n\n{'#'*60}")
    print("# Demo 5 ▸ 工具错误分类：可重试 vs 不可重试")
    print(f"{'#'*60}")

    agent = create_react_agent(
        model=llm,
        tools=[fetch_report, get_stock_price],
    )

    error_cases = [
        (
            "正常请求",
            "获取报告 RPT-0001 的内容",
        ),
        (
            "格式错误（不可重试）",
            "获取报告 REPORT-001 的内容",
        ),
        (
            "不存在的报告（不可重试）",
            "获取报告 RPT-9999 的内容",
        ),
        (
            "服务暂时不可用（可重试场景演示）",
            "获取报告 RPT-0002，retry_simulation 设为 True",
        ),
    ]

    for label, q in error_cases:
        print(f"\n── {label} ──")
        print(f"提问：{q}")
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        )
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                obs = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"  工具返回：{obs.strip()[:150]}")
        last = result["messages"][-1]
        if isinstance(last, AIMessage):
            content = last.content if isinstance(last.content, str) else ""
            if content.strip():
                print(f"  Agent 答复：{content.strip()[:200]}")


# ══════════════════════════════════════════════════════════════
# 直接单元测试（不走 LLM，验证工具本身的行为）
# ══════════════════════════════════════════════════════════════

def run_tool_unit_tests():
    """直接调用工具函数，验证输入边界和安全逻辑（不消耗 API）。"""
    print(f"\n\n{'#'*60}")
    print("# 工具单元测试（直接调用，不走 LLM）")
    print(f"{'#'*60}")

    cases = [
        # (工具函数, 参数字典, 期望行为描述)
        (get_stock_price, {"symbol": "AAPL"},       "✓ 正常查询"),
        (get_stock_price, {"symbol": "XYZ999"},     "✓ 不存在 → 返回错误提示"),
        (get_stock_price, {"symbol": "aapl"},       "✓ 小写 → 自动转大写"),
        (get_stock_price, {"symbol": "A1!@#"},      "✓ 非法格式 → 拒绝"),
        (read_file,       {"filename": "report.txt"},       "✓ 正常读取"),
        (read_file,       {"filename": "../etc/passwd"},    "✓ 路径遍历 → 安全拒绝"),
        (read_file,       {"filename": "nonexist.txt"},     "✓ 不存在 → 返回可用列表"),
        (lookup_user,     {"user_id": "10001"},             "✓ 正常查询"),
        (lookup_user,     {"user_id": "1 OR 1=1"},         "✓ 注入攻击 → 安全拒绝"),
        (lookup_user,     {"user_id": "99999999999"},       "✓ 超长 ID → 安全拒绝"),
    ]

    print()
    for fn, kwargs, desc in cases:
        # 直接调用工具的底层函数（.invoke 方法）
        result = fn.invoke(kwargs)
        result_str = str(result)[:80].replace("\n", " ")
        print(f"  {desc}")
        print(f"    输入：{kwargs}  →  返回：{result_str}")
        print()


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(DSEP)
    print("  Tool Calling 深度解析 Demo")
    print("  模型：GLM-4-Flash（智谱 AI）")
    print(DSEP)

    # 先运行单元测试（不消耗 API，快速验证工具逻辑）
    run_tool_unit_tests()

    # 再运行 Agent Demo（消耗 API）
    demo_good_vs_bad()
    demo_pydantic_validation()
    demo_parallel_calls()
    demo_security()
    demo_error_classification()

    print(f"\n✓ 所有 Demo 运行完毕")
