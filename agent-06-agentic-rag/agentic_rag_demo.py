"""
Agent系列 Demo 06: Agentic RAG——知识库集成的正确姿势

三个核心演示：
  Demo 1: Pipeline RAG vs Agentic RAG 核心差异
          - Pipeline：无论什么问题都检索，结果直接注入 Prompt
          - Agentic：Agent 自主决定是否需要检索
  Demo 2: 多知识库路由
          - 三个知识库：产品文档 / 运维手册 / 用户FAQ
          - Agent 根据问题类型路由到正确的知识库
  Demo 3: 质量门控 + 查询重写 Fallback
          - 检索质量不足时，自动重写查询并重试
          - 最大重试 2 次，保证有限资源内给出最优答案

运行要求：
  pip install -r requirements.txt
  cp .env.example .env && 填入 LLM_API_KEY（ZhiPuAI）和 EMBEDDING_API_KEY（SiliconFlow）
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

# ── Embeddings ────────────────────────────────────────────────────────────────

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    api_key=os.environ["EMBEDDING_API_KEY"],  # type: ignore[arg-type]
    base_url=os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1"),
)

# ── 三个知识库的文档 ───────────────────────────────────────────────────────────

PRODUCT_DOCS = [
    Document(
        page_content="WonderBot Pro 订阅价格：基础版 ¥99/月，专业版 ¥299/月，企业版按需报价。",
        metadata={"kb": "product", "topic": "pricing"},
    ),
    Document(
        page_content="API 调用限额：基础版 10K次/月，专业版 100K次/月，超出按 ¥0.01/次计费。",
        metadata={"kb": "product", "topic": "api_limits"},
    ),
    Document(
        page_content="WonderBot Pro 支持 GPT-4、Claude 3、Gemini Pro、GLM-4，可在控制台自由切换。",
        metadata={"kb": "product", "topic": "llm_support"},
    ),
    Document(
        page_content="数据安全：对话数据存储在中国区服务器，符合等保三级认证，支持数据加密导出。",
        metadata={"kb": "product", "topic": "security"},
    ),
]

OPS_DOCS = [
    Document(
        page_content="部署要求：Docker 20+，内存 ≥ 8GB，CPU ≥ 4核，推荐 docker-compose up --build。",
        metadata={"kb": "ops", "topic": "deployment"},
    ),
    Document(
        page_content="故障排查：服务无响应→检查 docker ps；API 超时→检查 LLM 连通性；内存溢出→调高 docker memory limit。",
        metadata={"kb": "ops", "topic": "troubleshooting"},
    ),
    Document(
        page_content="备份策略：每日凌晨 2 点自动备份，保留 30 天，存储在 /data/backups/，用 restore.sh 恢复。",
        metadata={"kb": "ops", "topic": "backup"},
    ),
    Document(
        page_content="监控告警：CPU > 80% 持续 5 分钟告警；内存 > 90% 告警；API 错误率 > 5% 告警，通过企微 Webhook 发送。",
        metadata={"kb": "ops", "topic": "monitoring"},
    ),
]

FAQ_DOCS = [
    Document(
        page_content="重置密码：登录页点击'忘记密码'→输入注册邮箱→查收重置邮件→设置新密码（8位以上，含大小写字母和数字）。",
        metadata={"kb": "faq", "topic": "account"},
    ),
    Document(
        page_content="退款政策：购买后 7 天内可申请全额退款，7-30 天按剩余时间比例退款，30 天后不支持退款。",
        metadata={"kb": "faq", "topic": "refund"},
    ),
    Document(
        page_content="申请发票：在'账单中心'点击'申请发票'，填写企业信息，3-5 工作日开出电子发票并发送邮箱。",
        metadata={"kb": "faq", "topic": "invoice"},
    ),
    Document(
        page_content="API Key 管理：在'开发者设置'中创建/撤销 API Key，每账号最多创建 5 个。",
        metadata={"kb": "faq", "topic": "api_key"},
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 构建向量索引
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  构建向量索引（三个知识库）")
print("=" * 65)

product_store = Chroma.from_documents(
    PRODUCT_DOCS, embeddings, collection_name="product_kb"
)
ops_store = Chroma.from_documents(
    OPS_DOCS, embeddings, collection_name="ops_kb"
)
faq_store = Chroma.from_documents(
    FAQ_DOCS, embeddings, collection_name="faq_kb"
)

all_docs = PRODUCT_DOCS + OPS_DOCS + FAQ_DOCS
unified_store = Chroma.from_documents(
    all_docs, embeddings, collection_name="unified_kb"
)

product_retriever = product_store.as_retriever(search_kwargs={"k": 2})
ops_retriever     = ops_store.as_retriever(search_kwargs={"k": 2})
faq_retriever     = faq_store.as_retriever(search_kwargs={"k": 2})
unified_retriever = unified_store.as_retriever(search_kwargs={"k": 3})

print(f"  product_kb: {len(PRODUCT_DOCS)} 文档")
print(f"  ops_kb:     {len(OPS_DOCS)} 文档")
print(f"  faq_kb:     {len(FAQ_DOCS)} 文档")
print(f"  unified_kb: {len(all_docs)} 文档（合并索引）")

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _ask(system: str, user: str) -> str:
    """单次 LLM 调用，返回文本"""
    resp = llm.invoke([SystemMessage(system), HumanMessage(user)])
    c = resp.content
    return c if isinstance(c, str) else str(c)


def _score_quality(question: str, context: str) -> float:
    """让 LLM 评估检索内容与问题的相关度，返回 0.0-1.0"""
    raw = _ask(
        "评估以下上下文对回答问题的帮助程度，只输出一个 0.0~1.0 的数字：\n"
        "1.0=完全覆盖；0.5=部分相关；0.0=完全不相关",
        f"问题：{question}\n\n上下文：{context[:600]}",
    )
    try:
        return round(max(0.0, min(1.0, float(raw.strip()))), 2)
    except ValueError:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 1: Pipeline RAG vs Agentic RAG
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Demo 1: Pipeline RAG vs Agentic RAG 核心差异")
print("=" * 65)

# ── Pipeline RAG：每次必检索 ──────────────────────────────────────────────────

def pipeline_rag(question: str) -> dict:
    """Pipeline RAG：检索→注入→生成，永远不跳过检索步骤"""
    docs = unified_retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    answer = _ask(
        f"根据以下参考资料回答问题，若资料无关请基于资料内容作答。\n参考：{context}",
        question,
    )
    return {"answer": answer, "retrieved": True, "docs": len(docs)}


# ── Agentic RAG：先判断是否需要检索 ──────────────────────────────────────────

def agentic_rag(question: str) -> dict:
    """Agentic RAG：先决策，再（选择性）检索"""
    # Step 1：Agent 决定是否需要检索
    decision = _ask(
        "判断以下问题是否需要查询知识库才能回答。\n"
        "需要检索的场景：产品定价/功能、运维操作、用户服务政策\n"
        "不需要检索的场景：常识问题、数学计算、通用编程知识\n"
        "只输出 yes 或 no",
        f"问题：{question}",
    ).strip().lower()

    need_retrieval = "yes" in decision

    if not need_retrieval:
        # 直接回答，不检索
        answer = _ask("你是一个知识丰富的助手，请直接回答问题。", question)
        return {"answer": answer, "retrieved": False, "docs": 0}
    else:
        # 检索后生成
        docs = unified_retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        answer = _ask(
            f"根据以下参考资料回答问题。\n参考：{context}",
            question,
        )
        return {"answer": answer, "retrieved": True, "docs": len(docs)}


DEMO1_QUESTIONS = [
    ("产品功能", "WonderBot Pro 基础版每月能调用多少次 API？"),
    ("运维操作", "部署 WonderBot 服务最低需要多少内存？"),
    ("用户服务", "购买 30 天后还能退款吗？"),
    ("通用常识", "Python 中如何计算列表的平均值？"),
    ("数学计算", "1024 除以 32 等于多少？"),
]

print(f"\n{'问题类型':<8} | {'Pipeline 检索':^10} | {'Agentic 检索':^10} | 问题")
print("-" * 65)

for q_type, q in DEMO1_QUESTIONS:
    p_result = pipeline_rag(q)
    a_result = agentic_rag(q)
    p_flag = f"✓ ({p_result['docs']}条)" if p_result["retrieved"] else "✗ 跳过"
    a_flag = f"✓ ({a_result['docs']}条)" if a_result["retrieved"] else "✗ 跳过"
    print(f"{q_type:<8} | {p_flag:^12} | {a_flag:^12} | {q[:30]}...")

print()
print("  结果分析：")
print("  • Pipeline RAG 对所有问题执行检索，包括常识和数学题")
print("  • Agentic RAG 对通用常识/数学题跳过检索，直接用 LLM 知识回答")
print("  → 节省了不必要的检索开销，同时避免知识库内容干扰通用问答")

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 2: 多知识库路由
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Demo 2: 多知识库路由")
print("=" * 65)

# ── LangGraph 实现多知识库路由 ─────────────────────────────────────────────────

class RoutingState(TypedDict):
    question:     str
    kb_choice:    str          # "product" | "ops" | "faq"
    context:      str
    answer:       str
    path:         list


def route_node(state: RoutingState) -> RoutingState:
    """Step 1：LLM 判断应查哪个知识库"""
    decision = _ask(
        "根据问题内容，判断应该查询哪个知识库，只输出知识库名称：\n"
        "product - 涉及产品功能、价格、技术规格、支持的模型\n"
        "ops     - 涉及部署、运维、故障排查、监控告警、备份恢复\n"
        "faq     - 涉及账号密码、退款、发票、API Key 等用户服务",
        f"问题：{state['question']}",
    ).strip().lower()

    kb = "faq"  # default
    for k in ["product", "ops", "faq"]:
        if k in decision:
            kb = k
            break

    return {**state, "kb_choice": kb, "path": [f"route→{kb}"]}


def retrieve_node(state: RoutingState) -> RoutingState:
    """Step 2：从选定的知识库检索"""
    retriever_map = {
        "product": product_retriever,
        "ops":     ops_retriever,
        "faq":     faq_retriever,
    }
    docs = retriever_map[state["kb_choice"]].invoke(state["question"])
    context = "\n".join(d.page_content for d in docs)
    return {**state, "context": context, "path": state["path"] + ["retrieve"]}


def generate_node(state: RoutingState) -> RoutingState:
    """Step 3：基于检索结果生成答案"""
    answer = _ask(
        f"根据以下参考资料简洁回答问题。\n参考：{state['context']}",
        state["question"],
    )
    return {**state, "answer": answer, "path": state["path"] + ["generate"]}


routing_graph = StateGraph(RoutingState)
routing_graph.add_node("route",    route_node)
routing_graph.add_node("retrieve", retrieve_node)
routing_graph.add_node("generate", generate_node)
routing_graph.set_entry_point("route")
routing_graph.add_edge("route",    "retrieve")
routing_graph.add_edge("retrieve", "generate")
routing_graph.add_edge("generate", END)
routing_agent = routing_graph.compile()


DEMO2_QUESTIONS = [
    ("应查 product", "专业版订阅每月多少钱？支持哪些大模型？"),
    ("应查 product", "数据存储在哪里，符合什么安全认证？"),
    ("应查 ops",     "服务 API 超时了怎么排查？"),
    ("应查 ops",     "监控到 CPU 超过 80% 会触发什么告警？"),
    ("应查 faq",     "我买了 15 天，还能退款多少？"),
    ("应查 faq",     "怎么给公司开增值税发票？"),
]

print(f"\n{'预期KB':<12} | {'实际路由':^10} | {'匹配':^4} | 问题")
print("-" * 65)

correct = 0
for expected, q in DEMO2_QUESTIONS:
    result = routing_agent.invoke({
        "question":  q,
        "kb_choice": "",
        "context":   "",
        "answer":    "",
        "path":      [],
    })
    actual = result["kb_choice"]
    expected_kb = expected.split()[-1]  # extract "product"/"ops"/"faq"
    match = "✓" if actual == expected_kb else "✗"
    if actual == expected_kb:
        correct += 1
    print(f"{expected:<12} | {actual:^10} | {match:^4} | {q[:32]}...")

print(f"\n  路由准确率：{correct}/{len(DEMO2_QUESTIONS)} = {correct/len(DEMO2_QUESTIONS)*100:.0f}%")

# 展示一个完整回答
print("\n  [完整示例] 问题：'API 超时了怎么排查？'")
sample = routing_agent.invoke({
    "question":  "API 超时了怎么排查？",
    "kb_choice": "",
    "context":   "",
    "answer":    "",
    "path":      [],
})
print(f"  路由到：{sample['kb_choice']}_kb")
print(f"  回答：{sample['answer'][:200]}...")

# ═══════════════════════════════════════════════════════════════════════════════
# Demo 3: 质量门控 + 查询重写 Fallback
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  Demo 3: 质量门控 + 查询重写 Fallback")
print("=" * 65)

QUALITY_THRESHOLD = 0.6
MAX_RETRIES = 2


class QualityGateState(TypedDict):
    question:       str
    rewritten_q:    str          # 重写后的查询（第一次等于原始问题）
    context:        str
    quality_score:  float
    answer:         str
    attempts:       int
    path:           list


def qg_retrieve_node(state: QualityGateState) -> QualityGateState:
    """检索：使用当前查询（原始或重写后）"""
    query = state["rewritten_q"] or state["question"]
    docs = unified_retriever.invoke(query)
    context = "\n".join(d.page_content for d in docs)
    return {
        **state,
        "context":  context,
        "path":     state["path"] + [f"retrieve(q='{query[:20]}...')"],
    }


def qg_evaluate_node(state: QualityGateState) -> QualityGateState:
    """评估检索质量"""
    score = _score_quality(state["question"], state["context"])
    return {
        **state,
        "quality_score": score,
        "path": state["path"] + [f"evaluate(score={score:.2f})"],
    }


def qg_rewrite_node(state: QualityGateState) -> QualityGateState:
    """查询重写：把模糊的问题改写得更具体"""
    rewritten = _ask(
        "将以下模糊问题改写为更具体的检索查询，保留原意但增加关键词，只输出改写后的问题：",
        state["question"],
    ).strip()
    return {
        **state,
        "rewritten_q": rewritten,
        "attempts":    state["attempts"] + 1,
        "path":        state["path"] + [f"rewrite→'{rewritten[:30]}...'"],
    }


def qg_generate_node(state: QualityGateState) -> QualityGateState:
    """基于当前 context 生成答案"""
    answer = _ask(
        f"根据以下参考资料回答问题，若信息不足请如实说明。\n参考：{state['context']}",
        state["question"],
    )
    return {**state, "answer": answer, "path": state["path"] + ["generate"]}


def should_rewrite(state: QualityGateState) -> str:
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "generate"
    if state["attempts"] >= MAX_RETRIES:
        return "generate"   # 达到重试上限，直接生成
    return "rewrite"


qg_graph = StateGraph(QualityGateState)
qg_graph.add_node("retrieve", qg_retrieve_node)
qg_graph.add_node("evaluate", qg_evaluate_node)
qg_graph.add_node("rewrite",  qg_rewrite_node)
qg_graph.add_node("generate", qg_generate_node)
qg_graph.set_entry_point("retrieve")
qg_graph.add_edge("retrieve", "evaluate")
qg_graph.add_conditional_edges(
    "evaluate",
    should_rewrite,
    {"generate": "generate", "rewrite": "rewrite"},
)
qg_graph.add_edge("rewrite",  "retrieve")
qg_graph.add_edge("generate", END)
qg_agent = qg_graph.compile()


# 测试用例：模糊查询，初次检索质量可能不高
DEMO3_QUESTIONS = [
    "价钱怎么样",        # 太模糊，没有"WonderBot"等关键词
    "出问题了怎么办",    # 模糊的故障排查
    "钱的事",            # 极度模糊
]

print(f"\n{'原始问题':<15} | {'重试次数':^6} | {'最终质量':^8} | 执行路径")
print("-" * 65)

for q in DEMO3_QUESTIONS:
    result = qg_agent.invoke({
        "question":      q,
        "rewritten_q":   q,
        "context":       "",
        "quality_score": 0.0,
        "answer":        "",
        "attempts":      0,
        "path":          [],
    })
    path_str = " → ".join(result["path"])
    retries  = result["attempts"]
    score    = result["quality_score"]
    print(f"{q:<15} | {retries:^6} | {score:^8.2f} | {path_str[:55]}...")

# 展示详细过程
print("\n  [详细过程] 原始问题：'价钱怎么样'")
verbose_result = qg_agent.invoke({
    "question":      "价钱怎么样",
    "rewritten_q":   "价钱怎么样",
    "context":       "",
    "quality_score": 0.0,
    "answer":        "",
    "attempts":      0,
    "path":          [],
})
print(f"  执行路径：{' → '.join(verbose_result['path'])}")
print(f"  最终查询：'{verbose_result['rewritten_q']}'")
print(f"  最终质量分：{verbose_result['quality_score']:.2f}")
print(f"  回答：{verbose_result['answer'][:200]}...")

# ═══════════════════════════════════════════════════════════════════════════════
# 总结对比
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  三种方案对比总结")
print("=" * 65)
print("""
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │              │ 检索决策     │ 知识库选择   │ 质量保障     │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ Pipeline RAG │ 总是检索     │ 固定单库     │ 无           │
  │ Agentic RAG  │ LLM 自主决定 │ LLM 路由选库 │ 质量门控+重写│
  └──────────────┴──────────────┴──────────────┴──────────────┘

  Agentic RAG 相比 Pipeline RAG 的核心优势：
  1. 节省 Token：不相关问题直接回答，不消耗检索资源
  2. 精准匹配：多知识库场景下找到最相关的数据源
  3. 自我修正：低质量检索结果触发查询重写，提升最终答案质量
""")
