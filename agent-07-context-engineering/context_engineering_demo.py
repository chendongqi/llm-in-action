"""
Agent系列 Demo 07: 上下文工程——让每个 Token 都用在刀刃上

三个核心演示：
  Demo 1: 上下文五来源 + Token 成本剖析
          - System Prompt / 工具定义 / 对话历史 / 检索内容 / 当前输入
          - 量化每个来源的 Token 消耗，展示 128K 窗口下的预算分配

  Demo 2: 预算约束下的动态上下文组装
          - 固定 Token 预算，按优先级将内容塞进上下文
          - 演示预算告急时的优先级裁剪策略

  Demo 3: 上下文溢出三种应对策略对比
          - 截断（Truncation）：保留最近 K 轮，丢弃早期历史
          - 摘要（Summarization）：LLM 压缩历史，摘要替代原文
          - 检索（Retrieval）：只拉取与当前问题语义相关的历史
          - 相同问题，三种策略下答案质量对比

运行要求：
  pip install -r requirements.txt
  cp .env.example .env && 填入 LLM_API_KEY 和 EMBEDDING_API_KEY
"""

import os
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ── LLM + Embeddings ──────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    api_key=os.environ["EMBEDDING_API_KEY"],  # type: ignore[arg-type]
    base_url=os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1"),
)

# ── Token 计数工具 ────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """统计文本 Token 数（cl100k_base 编码，对中文是近似值）"""
    return len(_enc.encode(text))


def msg_tokens(msg) -> int:
    """计算单条消息的 Token 数"""
    content = msg.content if hasattr(msg, "content") else str(msg)
    return count_tokens(content) + 4   # 4 个 overhead tokens per message


def _ask(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(system), HumanMessage(user)])
    c = resp.content
    return c if isinstance(c, str) else str(c)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1: 上下文五来源 + Token 成本剖析
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Demo 1: 上下文五来源 + Token 成本剖析")
print("=" * 65)

# ── Source 1: System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """你是 WonderBot 智能助手，专门帮助用户解答产品使用、技术集成和账号管理问题。

你具备以下能力：
1. 查询产品文档和 FAQ 知识库
2. 调用工单系统查询用户反馈
3. 执行代码并返回运行结果

回答要求：
- 优先基于知识库内容回答
- 如不确定，明确告知用户并建议联系人工支持
- 代码示例使用 Python 3.10+
- 回答简洁清晰，避免冗余"""

# ── Source 2: Tool Definitions ────────────────────────────────────────────────

TOOL_DEFINITIONS = """可用工具：

search_knowledge_base(query: str, kb_name: str = "default") -> str
  从知识库检索相关内容。kb_name 可选：product / ops / faq
  示例：search_knowledge_base("API 调用限额", "product")

create_ticket(user_id: str, subject: str, description: str) -> dict
  创建用户工单。返回：{ticket_id, status, estimated_response_time}

execute_code(code: str, language: str = "python") -> dict
  在沙箱中执行代码。返回：{stdout, stderr, exit_code, execution_time_ms}

get_user_info(user_id: str) -> dict
  查询用户账号信息。返回：{plan, api_usage, billing_cycle, status}"""

# ── Source 3: Conversation History ────────────────────────────────────────────

CONVERSATION_HISTORY = [
    HumanMessage("你好，我想了解一下 WonderBot Pro 的 API 限额"),
    AIMessage("您好！WonderBot Pro 基础版每月 10K 次，专业版 100K 次，超出部分 ¥0.01/次。"),
    HumanMessage("我现在是专业版，本月已经用了多少次？"),
    AIMessage("根据您的账号数据，本月已使用 67,432 次，剩余 32,568 次，当前计费周期还有 8 天。"),
    HumanMessage("如果超出了会自动续费吗？"),
    AIMessage("会自动按量计费，¥0.01/次，从您绑定的支付方式扣款，月底出账单。"),
    HumanMessage("好的，另外我在用 Python SDK 时遇到了 401 错误"),
    AIMessage("401 通常是 API Key 无效或过期。请检查：1) Key 是否正确复制 2) 是否已在控制台激活。"),
]

# ── Source 4: Retrieved Content ───────────────────────────────────────────────

RETRIEVED_CONTENT = """[检索结果 - 来自 product_kb]

文档1（相关度 0.92）：
Python SDK 认证示例：
  from wonderbot import WonderBot
  client = WonderBot(api_key="your_key_here")
  response = client.chat.completions.create(model="glm-4", messages=[...])

文档2（相关度 0.87）：
常见 401 错误原因：
  - API Key 包含多余空格（复制时常见）
  - Key 已被撤销（在控制台'开发者设置'中查看状态）
  - 使用了已过期的测试 Key（测试 Key 有效期 7 天）

文档3（相关度 0.81）：
SDK 版本兼容性：wonderbot-python >= 2.0.0 才支持流式输出"""

# ── Source 5: Current User Input ─────────────────────────────────────────────

CURRENT_INPUT = "我的 API Key 应该没问题，但还是 401，代码怎么写的？"

# ── Token 成本统计 ─────────────────────────────────────────────────────────────

sources = {
    "① System Prompt":     SYSTEM_PROMPT,
    "② Tool Definitions":  TOOL_DEFINITIONS,
    "③ Conv. History (8轮)": "\n".join(
        f"{type(m).__name__}: {m.content}" for m in CONVERSATION_HISTORY
    ),
    "④ Retrieved Content": RETRIEVED_CONTENT,
    "⑤ Current Input":     CURRENT_INPUT,
}

total_budget = 128_000
output_reserve = 4_000

print(f"\n  模型上下文窗口：{total_budget:,} tokens（GLM-4 / Claude Sonnet 量级）")
print(f"  输出预留：{output_reserve:,} tokens")
print(f"  可用于上下文：{total_budget - output_reserve:,} tokens\n")

print(f"  {'来源':<22} {'Token数':>8} {'占预算%':>8}  {'用途'}")
print("  " + "─" * 60)

total_used = 0
for name, text in sources.items():
    t = count_tokens(text)
    total_used += t
    pct = t / (total_budget - output_reserve) * 100
    print(f"  {name:<22} {t:>8,} {pct:>7.1f}%  ", end="")
    # Usage annotation
    if "System" in name:
        print("固定载入")
    elif "Tool" in name:
        print("按需加载（当前任务相关工具）")
    elif "History" in name:
        print("最近 N 轮（可截断/摘要）")
    elif "Retrieved" in name:
        print("动态（相关度过滤）")
    else:
        print("当前 Turn")

print("  " + "─" * 60)
pct_total = total_used / (total_budget - output_reserve) * 100
print(f"  {'合计':<22} {total_used:>8,} {pct_total:>7.1f}%")
print(f"  {'剩余 Buffer':<22} {total_budget - output_reserve - total_used:>8,}")

print(f"""
  关键洞察：
  • 即使加上 8 轮历史 + 检索内容，总 Token 消耗仍远低于 128K
  • 但如果对话持续到 100 轮，对话历史单项就会达到 ~{count_tokens(sources['③ Conv. History (8轮)']) * 12:,} tokens
  • 工具定义在工具多的 Agent 中会膨胀：20 个工具 × 平均 {count_tokens(TOOL_DEFINITIONS)//4} tokens ≈ {count_tokens(TOOL_DEFINITIONS)//4*20:,} tokens
  → 这就是为什么需要 Context Budget 管理""")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2: 预算约束下的动态上下文组装
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Demo 2: 预算约束下的动态上下文组装")
print("=" * 65)


class ContextBudgetManager:
    """
    按优先级分配 Token 预算，构建最优上下文。

    优先级（高→低）：
      P0 System Prompt  — 永远载入
      P1 Current Input  — 当前问题，必须载入
      P2 Recent History — 最近 K 轮（可压缩）
      P3 Retrieved Docs — 相关内容（按相关度裁剪）
      P4 Tool Defs      — 按需（只加载当前任务相关工具）
    """

    def __init__(self, total_budget: int = 128_000, output_reserve: int = 4_000):
        self.available = total_budget - output_reserve
        self.used = 0
        self.items: list[dict] = []

    def _remaining(self) -> int:
        return self.available - self.used

    def add(self, name: str, content: str, priority: int) -> bool:
        t = count_tokens(content)
        if t <= self._remaining():
            self.used += t
            self.items.append({"name": name, "tokens": t, "priority": priority})
            return True
        return False

    def add_with_trim(self, name: str, content: str, priority: int) -> int:
        """尽量多塞内容：超预算时截断到能放下的最大长度"""
        t = count_tokens(content)
        if t <= self._remaining():
            self.used += t
            self.items.append({"name": name, "tokens": t, "priority": priority})
            return t
        # 截断到剩余预算
        budget_left = self._remaining()
        if budget_left <= 0:
            return 0
        # 按比例截断
        ratio = budget_left / t
        trimmed = content[:int(len(content) * ratio)]
        actual_t = count_tokens(trimmed)
        self.used += actual_t
        self.items.append({"name": name, "tokens": actual_t, "priority": priority,
                           "trimmed": True})
        return actual_t

    def report(self) -> str:
        lines = [f"  {'来源':<22} {'Tokens':>8} {'%':>6}  状态"]
        lines.append("  " + "─" * 48)
        for item in self.items:
            pct = item["tokens"] / self.available * 100
            status = "✂ 截断" if item.get("trimmed") else "✓ 完整"
            lines.append(f"  {item['name']:<22} {item['tokens']:>8,} {pct:>5.1f}%  {status}")
        lines.append("  " + "─" * 48)
        pct_used = self.used / self.available * 100
        lines.append(f"  {'已用':<22} {self.used:>8,} {pct_used:>5.1f}%")
        lines.append(f"  {'剩余':<22} {self._remaining():>8,} {100-pct_used:>5.1f}%")
        return "\n".join(lines)


# 场景A：正常对话（预算充足）
print("\n  [场景 A] 正常对话 — 预算 12,000 tokens")
mgr_a = ContextBudgetManager(total_budget=12_000, output_reserve=2_000)
mgr_a.add("System Prompt",     SYSTEM_PROMPT,     priority=0)
mgr_a.add("Current Input",     CURRENT_INPUT,     priority=1)

# 对话历史：从最新到最旧逐轮加入，直到塞不下
history_rev = list(reversed(CONVERSATION_HISTORY))
loaded_turns = 0
for i in range(0, len(history_rev), 2):
    pair = history_rev[i:i+2]
    pair_text = "\n".join(m.content for m in pair)
    if mgr_a.add(f"History Turn -{loaded_turns+1}", pair_text, priority=2):
        loaded_turns += 1
    else:
        break

mgr_a.add_with_trim("Retrieved Docs", RETRIEVED_CONTENT, priority=3)
mgr_a.add_with_trim("Tool Defs",      TOOL_DEFINITIONS,  priority=4)
print(mgr_a.report())

# 场景B：紧张预算（模拟工具密集型 Agent）
print("\n  [场景 B] 预算紧张 — 预算 3,000 tokens（模拟工具密集型 Agent）")
mgr_b = ContextBudgetManager(total_budget=3_000, output_reserve=1_000)
results = {
    "System Prompt":   mgr_b.add("System Prompt",    SYSTEM_PROMPT,     0),
    "Current Input":   mgr_b.add("Current Input",    CURRENT_INPUT,     1),
    "Recent History":  mgr_b.add_with_trim("Recent History (1轮)", CONVERSATION_HISTORY[-1].content, 2),
    "Retrieved Docs":  mgr_b.add_with_trim("Retrieved Docs",       RETRIEVED_CONTENT,               3),
    "Tool Defs":       mgr_b.add_with_trim("Tool Defs",            TOOL_DEFINITIONS,                4),
}
print(mgr_b.report())
print("""
  预算紧张时的优先级体现：
  • System Prompt + 当前输入永远完整载入（P0/P1）
  • 对话历史只保留最近 1 轮（超出预算则截断）
  • 检索内容和工具定义按剩余预算按比例截断
  → 核心信息不丢，辅助信息随预算弹性调整""")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 3: 上下文溢出三种应对策略对比
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Demo 3: 上下文溢出三种应对策略对比")
print("=" * 65)

# ── 构造一个"超长"对话历史（20 轮 Python 学习对话）─────────────────────────────

LONG_HISTORY_TOPICS = [
    ("Python 列表推导式是什么？", "列表推导式是简化 for 循环的语法：[x*2 for x in range(5)] 得到 [0,2,4,6,8]。"),
    ("字典推导式怎么用？", "字典推导式：{k: v for k, v in items.items() if v > 0}，类似列表但生成 dict。"),
    ("生成器和列表有什么区别？", "生成器惰性求值，只在需要时计算，用 () 而非 []，适合大数据集，节省内存。"),
    ("装饰器的原理是什么？", "@decorator 本质是高阶函数，把被装饰函数包装后返回新函数，常用于日志、权限验证。"),
    ("上下文管理器 with 语句怎么实现？", "实现 __enter__ 和 __exit__ 方法，或用 @contextmanager 装饰器配合 yield。"),
    ("Python 的 GIL 是什么？", "GIL（全局解释器锁）保证同一时刻只有一个线程执行 Python 字节码，CPU 密集型用多进程。"),
    ("multiprocessing 和 threading 怎么选？", "IO 密集型用 threading（等待时可切换），CPU 密集型用 multiprocessing（绕过 GIL）。"),
    ("async/await 和线程的区别？", "async/await 是协程，单线程内的并发，适合大量 IO 等待场景，比线程开销更低。"),
    ("dataclass 和普通 class 有什么优势？", "@dataclass 自动生成 __init__、__repr__、__eq__，减少样板代码，支持 frozen 不可变。"),
    ("Pydantic 和 dataclass 怎么选？", "需要数据验证、序列化、类型强制用 Pydantic；纯数据容器用 dataclass，更轻量。"),
]

long_history = []
for q, a in LONG_HISTORY_TOPICS:
    long_history.append(HumanMessage(q))
    long_history.append(AIMessage(a))

OVERFLOW_QUESTION = "我最开始问的 Python 列表推导式是什么？能给我一个实际的使用例子吗？"

history_text = "\n".join(
    f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
    for m in long_history
)
total_history_tokens = count_tokens(history_text)
MAX_HISTORY_TOKENS = 300   # 模拟 token 告急：只允许 300 tokens 的历史

print(f"\n  对话历史：{len(long_history)} 条消息 / {total_history_tokens} tokens")
print(f"  历史 Token 上限：{MAX_HISTORY_TOKENS} tokens（模拟窗口告急）")
print(f"  测试问题：'{OVERFLOW_QUESTION}'")
print(f"  （答案在第 1 轮，需要从 10 轮历史中找到）\n")

# ── Strategy 1: 截断（保留最近 K 轮）────────────────────────────────────────────

def strategy_truncation(history: list, max_tokens: int) -> tuple[list, int]:
    """从最新消息开始，倒序累积直到超预算，丢弃早期消息"""
    kept = []
    used = 0
    for msg in reversed(history):
        t = msg_tokens(msg)
        if used + t > max_tokens:
            break
        kept.insert(0, msg)
        used += t
    return kept, used


kept_msgs, used_t = strategy_truncation(long_history, MAX_HISTORY_TOKENS)
truncation_context = "\n".join(
    f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
    for m in kept_msgs
)

print("  [策略 1] 截断（Truncation）")
print(f"  保留消息数：{len(kept_msgs)}/{len(long_history)} | 使用 tokens：{used_t}/{MAX_HISTORY_TOKENS}")
print(f"  最早可见消息：'{kept_msgs[0].content[:40]}...'")

answer_truncation = _ask(
    f"你是 Python 助手。根据以下对话历史回答用户问题。\n历史：{truncation_context}",
    OVERFLOW_QUESTION,
)
print(f"  回答：{answer_truncation[:180]}...")

# ── Strategy 2: 摘要（LLM 压缩历史）─────────────────────────────────────────────

print("\n  [策略 2] 摘要（Summarization）")
summary = _ask(
    "将以下 Python 学习对话压缩为简洁摘要，保留所有已讨论的技术主题和关键结论（不超过 150 字）：",
    history_text,
)
summary_tokens = count_tokens(summary)
print(f"  原始历史：{total_history_tokens} tokens → 摘要：{summary_tokens} tokens（压缩比 {total_history_tokens/summary_tokens:.1f}x）")
print(f"  摘要内容：{summary[:200]}...")

answer_summary = _ask(
    f"你是 Python 助手。基于以下历史摘要回答用户问题。\n历史摘要：{summary}",
    OVERFLOW_QUESTION,
)
print(f"  回答：{answer_summary[:180]}...")

# ── Strategy 3: 检索（只拉取相关历史）──────────────────────────────────────────────

print("\n  [策略 3] 检索（Retrieval）")

# 把每轮对话变成 Document，建向量索引
history_docs = []
for i, (q, a) in enumerate(LONG_HISTORY_TOPICS):
    history_docs.append(Document(
        page_content=f"Q: {q}\nA: {a}",
        metadata={"turn": i + 1, "topic": q[:20]},
    ))

history_store = Chroma.from_documents(
    history_docs, embeddings, collection_name="history_retrieval_demo"
)
history_retriever = history_store.as_retriever(search_kwargs={"k": 2})

relevant_docs = history_retriever.invoke(OVERFLOW_QUESTION)
retrieval_context = "\n\n".join(
    f"[Turn {d.metadata['turn']}] {d.page_content}" for d in relevant_docs
)
retrieval_tokens = count_tokens(retrieval_context)
print(f"  检索到 {len(relevant_docs)} 条相关历史（{retrieval_tokens} tokens）：")
for d in relevant_docs:
    print(f"    Turn {d.metadata['turn']}: {d.metadata['topic']}...")

answer_retrieval = _ask(
    f"你是 Python 助手。根据以下相关历史记录回答用户问题。\n相关历史：{retrieval_context}",
    OVERFLOW_QUESTION,
)
print(f"  回答：{answer_retrieval[:180]}...")

# ── 三策略汇总对比 ────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  三策略汇总对比")
print("=" * 65)
print(f"""
  {'策略':<12} {'历史Token使用':>14} {'第1轮是否可见':>14}
  {'─'*50}
  {'截断':<12} {used_t:>14} {'✗ 已丢弃':>14}
  {'摘要':<12} {summary_tokens:>14} {'✓ 在摘要中':>14}
  {'检索':<12} {retrieval_tokens:>14} {'✓ 语义命中':>14}

  截断：实现最简单，但丢失了第 1 轮的列表推导式内容
  摘要：压缩了 {total_history_tokens/summary_tokens:.0f}x，第 1 轮内容保留在摘要中，但细节可能被泛化
  检索：精准命中第 1 轮话题，token 消耗最低，但需要向量索引

  生产选择建议：
  → 对话轮数 < 20：截断（简单可靠）
  → 对话轮数 20-100：摘要（平衡效果与成本）
  → 对话轮数 > 100 或有"找早期内容"需求：检索（最优）
""")
