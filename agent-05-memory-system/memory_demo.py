"""Agent 系列第 6 篇：记忆管理 Demo

演示 Agent 四种记忆类型 + LangGraph checkpointer/store + 自动摘要压缩：
  Demo 1: 四种记忆类型对比（感觉/工作/情景/语义）
  Demo 2: 三种上下文管理策略（截断/摘要/检索）
  Demo 3: checkpointer — 会话内多轮对话状态持久化
  Demo 4: InMemoryStore — 跨会话长期用户记忆
  Demo 5: 自动摘要压缩 — 历史超长时触发 RemoveMessage 压缩
"""

import json
import os
import uuid
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

load_dotenv()


def _s(content: object) -> str:
    """AIMessage.content 可能是 str 或 list，统一转为 str。"""
    return content if isinstance(content, str) else str(content)


# ── 模型初始化 ──────────────────────────────────────────────────────────────────
llm = ChatOpenAI(  # type: ignore[call-arg]
    model="glm-4-flash",
    openai_api_key=os.environ["LLM_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)


# ════════════════════════════════════════════════════════════════════════════════
# DEMO 1: 四种记忆类型对比
# ════════════════════════════════════════════════════════════════════════════════

def demo1_four_memory_types():
    print("\n" + "=" * 60)
    print("DEMO 1: 四种记忆类型对比")
    print("=" * 60)

    # ── 感觉记忆：当前 Turn 的输入/输出，响应后丢弃 ──
    print("\n[感觉记忆 - Sensory Memory]")
    print("  特点: 单次 Turn 的消息，生命周期 = 一次 LLM 调用")
    q = "Python 中 len([1, 2, 3]) 等于多少"
    answer = llm.invoke([HumanMessage(q)])
    print(f"  输入: {q}")
    print(f"  输出: {_s(answer.content).strip()}")
    print("  生命周期: 这个 invoke 结束后，内容不会保留到下一次")

    # ── 工作记忆：有限的对话历史 ──
    print("\n[工作记忆 - Working Memory]")
    history = [
        HumanMessage("我叫李雷，是一名 Python 工程师"),
        AIMessage("你好，李雷！很高兴认识你。"),
        HumanMessage("我最近在学习 LangGraph"),
        AIMessage("LangGraph 很强大，特别适合构建有状态的 Agent。"),
    ]
    test_q = "我之前告诉你我叫什么名字？"
    resp_with = llm.invoke(history + [HumanMessage(test_q)])
    resp_without = llm.invoke([HumanMessage(test_q)])
    print(f"  问题: '{test_q}'")
    print(f"  有历史 → {_s(resp_with.content).strip()[:80]}")
    print(f"  无历史 → {_s(resp_without.content).strip()[:80]}")

    # ── 情景记忆：压缩的历史摘要 ──
    print("\n[情景记忆 - Episodic Memory]")
    long_history = history * 4  # 模拟很长的历史
    summary_resp = llm.invoke([
        SystemMessage("将以下对话压缩为 60 字以内的摘要，保留关键信息"),
        HumanMessage(str([m.content for m in long_history])),
    ])
    print(f"  原始: {len(long_history)} 条消息")
    print(f"  摘要: {_s(summary_resp.content).strip()[:120]}")
    print("  特点: 用摘要代替原始消息，大幅节省 Token")

    # ── 语义记忆：跨会话的用户偏好/事实 ──
    print("\n[语义记忆 - Semantic Memory]")
    user_profile = {
        "name": "李雷",
        "role": "Python 工程师",
        "interests": ["LangGraph", "Agent 开发"],
        "level": "中级",
    }
    resp = llm.invoke([
        SystemMessage(f"用户资料: {json.dumps(user_profile, ensure_ascii=False)}"),
        HumanMessage("给我推荐下一步的学习方向"),
    ])
    print(f"  存储的用户信息: {user_profile}")
    print(f"  个性化推荐: {_s(resp.content).strip()[:150]}")
    print("  特点: 跨会话持久化，Agent 重启后依然知道用户信息")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO 2: 三种上下文管理策略对比
# ════════════════════════════════════════════════════════════════════════════════

def demo2_context_strategies():
    print("\n" + "=" * 60)
    print("DEMO 2: 上下文管理策略对比（截断 / 摘要 / 检索）")
    print("=" * 60)

    # 模拟长对话历史
    topics = [
        ("Python 列表", "list=[1,2,3]，有序可变序列，支持增删改查"),
        ("Python 元组", "tuple=(1,2,3)，有序不可变序列，比 list 更省内存"),
        ("Python 字典", "dict={'k':'v'}，哈希表实现的键值对映射"),
        ("Python 集合", "set={1,2,3}，无序唯一元素集合，支持交并差运算"),
        ("Python 函数", "def func():，封装可复用逻辑，支持默认参数和关键字参数"),
        ("Python 类", "class Cls:，面向对象编程，封装/继承/多态三要素"),
        ("Python 装饰器", "@decorator，函数包装模式，不修改函数本身添加功能"),
        ("Python 生成器", "yield 关键字，惰性计算，适合大数据集遍历"),
    ]

    history = []
    for topic, answer in topics:
        history.append(HumanMessage(f"解释一下 {topic}"))
        history.append(AIMessage(answer))

    test_q = "Python 列表是什么？有哪些特点？"
    print(f"\n原始历史: {len(history)} 条消息（{len(topics)} 个主题）")
    print(f"测试问题: '{test_q}'（这个话题在第 1 条）\n")

    # 策略一：截断 — 保留最近 N 条
    truncated = history[-4:]
    resp_trunc = llm.invoke(truncated + [HumanMessage(test_q)])
    print(f"[策略一：截断] 只保留最近 4 条消息（最早可见：'{truncated[0].content[:30]}'）")
    print(f"  回答: {_s(resp_trunc.content).strip()[:120]}")
    print("  ⚠  丢失了早期的'列表'知识，LLM 靠自身知识回答，不知道是我们'学过'的")

    # 策略二：摘要 — LLM 压缩历史
    summary_resp = llm.invoke([
        SystemMessage("将对话历史压缩为一段摘要（不超过 80 字），保留所有已介绍的主题名称"),
        HumanMessage("\n".join([f"{m.type}: {m.content}" for m in history])),
    ])
    summary = _s(summary_resp.content).strip()
    resp_summary = llm.invoke([
        SystemMessage(f"对话历史摘要: {summary}"),
        HumanMessage(test_q),
    ])
    print(f"\n[策略二：摘要] {len(history)} 条 → 1 段摘要（{len(summary)} 字）")
    print(f"  摘要内容: {summary[:100]}")
    print(f"  回答: {_s(resp_summary.content).strip()[:120]}")
    print("  ✓ 保留了所有主题脉络，Token 消耗大幅减少")

    # 策略三：检索 — 只拉取语义相关历史
    relevant = [m for m in history if "列表" in m.content or "list" in m.content.lower()]
    resp_retrieved = llm.invoke(relevant + [HumanMessage(test_q)])
    print(f"\n[策略三：检索] {len(history)} 条 → {len(relevant)} 条相关历史")
    print(f"  回答: {_s(resp_retrieved.content).strip()[:120]}")
    print("  ✓ 精准 + 最省 Token，适合长期知识积累场景（需要向量检索支持）")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO 3: LangGraph checkpointer — 会话内状态持久化
# ════════════════════════════════════════════════════════════════════════════════

def demo3_checkpointer():
    print("\n" + "=" * 60)
    print("DEMO 3: LangGraph checkpointer — 多轮对话状态持久化")
    print("=" * 60)

    @tool
    def get_weather(city: str) -> str:
        """查询城市今日天气（模拟数据）"""
        data = {
            "北京": "晴，25°C，东北风 3 级，空气良好",
            "上海": "多云，22°C，东南风 2 级，轻度雾霾",
            "广州": "小雨，28°C，南风 1 级，湿度 85%",
            "深圳": "阵雨，27°C，西南风 2 级，雷暴预警",
        }
        return data.get(city, f"{city} 暂无天气数据")

    checkpointer = MemorySaver()
    agent = create_react_agent(model=llm, tools=[get_weather], checkpointer=checkpointer)

    print("\n── 会话 A（thread_id: weather_001）──")
    config_a = {"configurable": {"thread_id": "weather_001"}}

    turns = [
        "北京今天天气怎么样？",
        "那上海呢？",                       # "那"需要上文知道在说天气
        "这两个城市哪个更适合今天出行？",    # 需要前两轮的查询结果
    ]

    for i, question in enumerate(turns, 1):
        result = agent.invoke({"messages": [HumanMessage(question)]}, config=config_a)  # type: ignore[arg-type]
        answer = _s(result["messages"][-1].content).strip()
        msg_count = len(result["messages"])
        print(f"\n[Turn {i}] 用户: {question}")
        print(f"         Agent: {answer[:120]}")
        print(f"         (当前状态中消息数: {msg_count})")

    print("\n── 会话 B（不同 thread_id，没有任何历史）──")
    config_b = {"configurable": {"thread_id": "weather_002"}}
    result_new = agent.invoke(
        {"messages": [HumanMessage("我刚才问的是哪个城市？")]},
        config=config_b,  # type: ignore[arg-type]
    )
    print("用户: 我刚才问的是哪个城市？")
    print(f"Agent: {_s(result_new['messages'][-1].content).strip()[:120]}")
    print("\n→ 不同 thread_id 完全隔离，新会话不知道任何历史")

    print("\n── 会话 A 继续（同一 thread_id，记忆保留）──")
    result_cont = agent.invoke(
        {"messages": [HumanMessage("刚才查的两个城市，再查一下深圳对比一下")]},
        config=config_a,  # type: ignore[arg-type]
    )
    print("用户: 刚才查的两个城市，再查一下深圳对比一下")
    print(f"Agent: {_s(result_cont['messages'][-1].content).strip()[:200]}")
    print("\n→ 同一 thread_id 记住了前面查过北京和上海")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO 4: LangGraph InMemoryStore — 跨会话长期记忆
# ════════════════════════════════════════════════════════════════════════════════

def demo4_long_term_store():
    print("\n" + "=" * 60)
    print("DEMO 4: LangGraph InMemoryStore — 跨会话长期记忆")
    print("=" * 60)
    print("""
checkpointer  vs  store 区别：
  checkpointer  → 短期记忆，绑定 thread_id，通常随会话生命周期
  InMemoryStore → 长期记忆，绑定 user_id，跨会话永久存在
  生产中 store 可替换为 PostgresStore / RedisStore
""")

    store = InMemoryStore()
    checkpointer = MemorySaver()

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        user_id: str

    def memory_node(state: AgentState) -> dict:
        user_id = state.get("user_id", "anon")  # type: ignore[call-overload]

        # 从 store 读取该用户的长期记忆
        stored_facts = store.search(("user_facts", user_id))
        system_text = "你是一个有记忆的个人助手。"
        if stored_facts:
            fact_list = [item.value.get("fact", "") for item in stored_facts]
            system_text += "\n\n【已知用户信息】\n" + "\n".join(f"  - {f}" for f in fact_list)

        response = llm.invoke([SystemMessage(system_text)] + state["messages"])

        # 提取并存储用户的新信息（姓名、职业、偏好等事实）
        last_msg = _s(state["messages"][-1].content) if state["messages"] else ""
        extraction = _s(llm.invoke([
            SystemMessage(
                "从用户消息中提取关于用户本人的事实（姓名/职业/技术栈/偏好等）。"
                "只有确实存在明确事实时才输出，否则输出空字符串。只输出事实本身，不要解释。"
            ),
            HumanMessage(last_msg),
        ]).content).strip()

        if extraction and len(extraction) > 3:
            key = f"fact_{uuid.uuid4().hex[:8]}"
            store.put(("user_facts", user_id), key, {"fact": extraction})
            print(f"  [记忆写入 → store] {extraction[:70]}")

        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", memory_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    app = graph.compile(checkpointer=checkpointer)

    user_id = "li_lei"

    # ── 会话 A：用户自我介绍 ──
    print(f"\n── 会话 A（用户 '{user_id}' 自我介绍）──")
    config_a = {"configurable": {"thread_id": "session_a_001"}}
    msgs_a = [
        "我叫李雷，在一家互联网公司做后端工程师",
        "我主要用 Python 和 Go，最近在学 LangGraph 和 Agent 开发",
        "我喜欢通过动手实践来学习，不太喜欢纯看文档",
    ]
    for msg in msgs_a:
        r = app.invoke({"messages": [HumanMessage(msg)], "user_id": user_id}, config=config_a)  # type: ignore[arg-type]
        print(f"\n  用户: {msg}")
        print(f"  Agent: {_s(r['messages'][-1].content).strip()[:100]}")

    # 查看长期记忆中存储的内容
    stored = store.search(("user_facts", user_id))
    print(f"\n── 存储到 store 的长期记忆（{len(stored)} 条）──")
    for item in stored:
        print(f"  • {item.value['fact']}")

    # ── 会话 B：完全不同的会话，但 store 的记忆跨会话存在 ──
    print("\n── 会话 B（全新 thread_id，store 记忆依然存在）──")
    config_b = {"configurable": {"thread_id": "session_b_002"}}
    r_b = app.invoke(
        {"messages": [HumanMessage("你好，你认识我吗？")], "user_id": user_id},
        config=config_b,  # type: ignore[arg-type]
    )
    print("  用户: 你好，你认识我吗？")
    print(f"  Agent: {_s(r_b['messages'][-1].content).strip()[:250]}")
    print("\n  → 虽然是全新 thread_id，但 store 中的用户信息跨会话持久")

    # ── 对比：不同用户，没有记忆 ──
    print("\n── 不同用户（han_meimei）——没有任何记忆 ──")
    r_other = app.invoke(
        {"messages": [HumanMessage("你知道我的职业是什么吗？")], "user_id": "han_meimei"},
        config={"configurable": {"thread_id": "han_001"}},  # type: ignore[arg-type]
    )
    print("  用户 han_meimei: 你知道我的职业是什么吗？")
    print(f"  Agent: {_s(r_other['messages'][-1].content).strip()[:100]}")
    print("  → 不同 user_id 的记忆完全隔离")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO 5: 自动摘要压缩 — RemoveMessage + 摘要轮替
# ════════════════════════════════════════════════════════════════════════════════

def demo5_auto_summarization():
    print("\n" + "=" * 60)
    print("DEMO 5: 自动摘要压缩 — 历史超长时触发 RemoveMessage 压缩")
    print("=" * 60)
    print("""
策略：
  当消息数超过阈值，触发 compress 节点：
    1. 旧消息 → LLM 压缩成摘要
    2. 用 RemoveMessage 删除旧消息（add_messages reducer 处理）
    3. 摘要存入 state["summary"]，下一轮注入 system prompt
  效果：无论对话多长，active 消息数始终 ≤ 阈值 + 2
""")

    THRESHOLD = 8  # 超过 8 条消息触发压缩

    class SummaryState(TypedDict):
        messages: Annotated[list, add_messages]
        summary: Optional[str]

    def chat_node(state: SummaryState) -> dict:
        summary = state.get("summary") or ""  # type: ignore[call-overload]
        system_parts = ["你是一个 Python 教学助手，回答简洁。"]
        if summary:
            system_parts.append(f"\n\n【历史摘要】{summary}")
        resp = llm.invoke([SystemMessage("".join(system_parts))] + state["messages"])
        return {"messages": [resp]}

    def should_compress(state: SummaryState) -> str:
        return "compress" if len(state["messages"]) > THRESHOLD else "end"

    def compress_node(state: SummaryState) -> dict:
        messages = state["messages"]
        to_compress = messages[:-2]   # 保留最新 2 条，其余全部压缩
        keep = messages[-2:]

        # 合并现有摘要 + 旧消息 → 新摘要
        prev_summary = state.get("summary") or ""  # type: ignore[call-overload]
        compress_parts = []
        if prev_summary:
            compress_parts.append(f"现有摘要：{prev_summary}")
        compress_parts.append("新增对话：")
        compress_parts.extend([f"  {m.type}: {m.content}" for m in to_compress])

        new_summary = _s(llm.invoke([
            SystemMessage("将以下内容压缩为 120 字以内的摘要，保留所有学习过的主题名称和关键结论"),
            HumanMessage("\n".join(compress_parts)),
        ]).content).strip()

        print(f"\n  [压缩触发] {len(messages)} 条 → 压缩 {len(to_compress)} 条，保留 {len(keep)} 条")
        print(f"  [新摘要]   {new_summary[:100]}...")

        # RemoveMessage：告诉 add_messages reducer 删除这些消息
        remove_ops = [RemoveMessage(id=m.id) for m in to_compress]
        return {"messages": remove_ops, "summary": new_summary}

    graph = StateGraph(SummaryState)
    graph.add_node("chat", chat_node)
    graph.add_node("compress", compress_node)
    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", should_compress, {"compress": "compress", "end": END})
    graph.add_edge("compress", END)

    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "python_learner_001"}}

    questions = [
        "Python 列表有哪些常用方法？",
        "字典的 get() 和直接用 [] 有什么区别？",
        "Python 的 *args 是什么用法？",
        "**kwargs 呢？",
        "什么是列表推导式？举个简单例子",
        "集合推导式也是类似的语法吗？",
        "帮我总结一下我们讲过的内容",
        "好的，继续讲一下 Python 的 lambda 函数",
        "lambda 和普通 def 函数有什么本质区别？",
        "什么场景下适合用 lambda？",
        "现在汇总：我掌握了哪些 Python 知识点？",
    ]

    print(f"阈值设置：超过 {THRESHOLD} 条消息触发自动压缩\n")

    for i, q in enumerate(questions, 1):
        invoke_input: dict = {"messages": [HumanMessage(q)]}
        if i == 1:
            invoke_input["summary"] = None  # 首轮初始化

        result = app.invoke(invoke_input, config=config)  # type: ignore[arg-type]
        last = _s(result["messages"][-1].content).strip()
        msg_count = len(result["messages"])
        has_summary = bool(result.get("summary"))

        print(f"[Turn {i:2d}] {q}")
        print(f"         回答: {last[:80]}...")
        print(f"         消息数: {msg_count:2d} | 摘要: {'✓ 已压缩' if has_summary else '○ 无'}\n")


# ════════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Agent 系列第 6 篇：记忆管理 Demo")
    print("四种记忆类型 + checkpointer + store + 自动摘要压缩")
    print("=" * 60)

    demo1_four_memory_types()
    demo2_context_strategies()
    demo3_checkpointer()
    demo4_long_term_store()
    demo5_auto_summarization()

    print("\n" + "=" * 60)
    print("全部 Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
