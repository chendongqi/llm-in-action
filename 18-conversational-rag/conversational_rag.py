"""
Article 18: Conversational RAG — Multi-Turn QA with Memory

Problem: all previous articles treated each question independently.
In real applications users ask follow-up questions:

  Turn 1: "What is RAGAS?"
  Turn 2: "What are its four metrics?"      ← "its" refers to RAGAS
  Turn 3: "Which one is hardest to improve?" ← refers to the metrics above

Without chat history, Turn 2–3 retrieve garbage because the standalone
query "What are its four metrics?" has no referent.

Solution: History-Aware Retriever
  Before retrieval, feed the full chat history to an LLM and ask it to
  rewrite the current question as a standalone, self-contained question.

  "What are its four metrics?" + [Turn 1 history]
    → "What are the four metrics in the RAGAS framework?"

Two pipelines compared:
  Baseline:  plain RAG, ignores history, retrieves on raw question
  ConvRAG:   history-aware retriever → contextualised retrieval → answer

Evaluation:
  - Qualitative: show retrieved docs & answers for each turn side-by-side
  - Quantitative: RAGAS metrics on the final turn of each conversation
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ─── LLM / Embeddings ─────────────────────────────────────────────────────────

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

EMB_BASE_URL = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMB_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")
EMB_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                 model=LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(base_url=EMB_BASE_URL, api_key=EMB_API_KEY,
                              model=EMB_MODEL)

TOP_K = 4

# ─── Knowledge Base ────────────────────────────────────────────────────────────

DOCUMENTS = [
    Document(page_content="""
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识检索与大语言模型结合的技术。
RAG的核心流程：检索（Retrieval）→ 增强（Augmentation）→ 生成（Generation）。
RAG由Meta AI在2020年提出，解决了LLM的知识截止问题。
RAG适合需要最新信息、私有知识库问答、减少幻觉的场景。
""", metadata={"source": "rag-intro"}),

    Document(page_content="""
RAGAS是专为RAG系统设计的评估框架，由Es等人在2023年提出。
RAGAS的四个核心指标：
1. context_recall（上下文召回率）：检索内容是否覆盖了所有必要信息
2. context_precision（上下文精确率）：检索内容中相关文档的比例
3. faithfulness（忠实度）：答案是否完全基于上下文，衡量幻觉程度
4. answer_relevancy（答案相关性）：答案与问题的相关程度
其中faithfulness最难提升，因为需要LLM严格约束自己只说文档里有的内容。
""", metadata={"source": "ragas"}),

    Document(page_content="""
向量数据库是RAG的核心存储组件，负责存储文档的向量表示并支持相似度搜索。
常见向量数据库：
- Chroma：轻量级，适合本地开发和原型验证
- Pinecone：云托管，适合生产级应用，支持大规模数据
- Milvus：开源高性能，适合企业大规模部署
- Qdrant：Rust实现，内存效率高，适合资源受限环境
选择建议：开发阶段用Chroma，生产阶段根据规模选Pinecone或Milvus。
""", metadata={"source": "vector-db"}),

    Document(page_content="""
Embedding模型将文本转换为向量，决定了语义检索的质量上限。
中文场景推荐BAAI/bge-large-zh-v1.5（北京智源研究院开发，MTEB中文榜前列）。
bge-large-zh-v1.5的向量维度为1024，适合大多数中文RAG场景。
英文场景可选text-embedding-ada-002（OpenAI）或bge-large-en-v1.5。
Embedding模型的选择直接影响context_recall指标。
""", metadata={"source": "embedding"}),

    Document(page_content="""
Rerank（重排序）使用Cross-Encoder对初检结果重新评分，是提升精确率的关键步骤。
Rerank流程：初检top-20 → Cross-Encoder逐一打分 → 取top-4。
常用Rerank模型：BAAI/bge-reranker-v2-m3（中英双语均佳）。
Rerank主要提升context_precision指标，通常提升幅度在+0.15~+0.30。
Rerank与查询优化（HyDE、Multi-Query）正交，可以叠加使用。
""", metadata={"source": "rerank"}),

    Document(page_content="""
Self-RAG（2023）通过反思令牌让模型自主决定是否检索以及评估答案质量。
CRAG（Corrective RAG，2024）在检索后对文档质量打分，不合格时触发网络搜索。
Graph RAG将文档构建为知识图谱，通过图遍历解决多跳关系推理问题。
Agentic RAG整合多种策略，由Agent动态决定检索路径，包含质量评估和重试机制。
这四种高级RAG技术是循序渐进的：从"决定要不要检索"到"动态选择最优策略"。
""", metadata={"source": "advanced-rag"}),

    Document(page_content="""
对话式RAG（Conversational RAG）在多轮对话中保持上下文理解能力。
核心问题：用户追问时往往使用代词（"它"、"这个"）或省略主语，直接检索会失效。
解决方案：History-Aware Retriever——用LLM将当前问题结合历史重写为独立完整问题，
再用改写后的问题进行检索，确保每次检索都是语义完整的查询。
对话记忆管理策略：完整历史（精确但耗token）、摘要记忆（节省token但有损耗）。
""", metadata={"source": "conv-rag"}),

    Document(page_content="""
文档分块策略影响RAG检索质量：固定大小分块（chunk_size=512-1024）适合通用场景。
父子分块：子chunk（200字符）用于精准检索，父chunk（800字符）用于完整生成。
上下文感知分块（Anthropic提出）：为每个chunk添加LLM生成的上下文描述前缀。
多级分块（Multi-granularity）：同一段落建立粗细两级索引，按需选择粒度。
分块策略主要影响context_recall和context_precision两个RAGAS指标。
""", metadata={"source": "chunking"}),
]

# ─── Multi-turn Conversation Test Cases ────────────────────────────────────────

# Three conversations, each with 3 turns
# Turn 3 of each is used for RAGAS evaluation
CONVERSATIONS = [
    {
        "id": "conv_ragas",
        "name": "RAGAS追问对话",
        "turns": [
            {
                "question": "RAGAS是什么？",
                "ground_truth": "RAGAS是专为RAG系统设计的评估框架，由Es等人在2023年提出，提供context_recall、context_precision、faithfulness、answer_relevancy四个核心评估指标。",
            },
            {
                "question": "它有哪四个核心指标？",   # "它" = RAGAS
                "ground_truth": "RAGAS的四个核心指标是：context_recall（上下文召回率）、context_precision（上下文精确率）、faithfulness（忠实度）、answer_relevancy（答案相关性）。",
            },
            {
                "question": "其中哪个最难提升？为什么？",  # "其中" = 这四个指标
                "ground_truth": "faithfulness（忠实度）最难提升，因为需要LLM严格约束自己只基于文档内容生成答案，不引入任何外部知识，这与LLM的训练目标存在天然张力。",
            },
        ],
    },
    {
        "id": "conv_vector_db",
        "name": "向量数据库追问对话",
        "turns": [
            {
                "question": "常见的向量数据库有哪些？",
                "ground_truth": "常见向量数据库包括Chroma（轻量级，适合本地开发）、Pinecone（云托管，生产级）、Milvus（开源高性能，企业部署）、Qdrant（Rust实现，内存效率高）。",
            },
            {
                "question": "其中哪个最适合生产环境？",  # "其中" = 上面的向量数据库
                "ground_truth": "生产环境中Pinecone和Milvus最适合，Pinecone提供云托管服务支持大规模数据，Milvus是开源高性能方案适合企业大规模部署。",
            },
            {
                "question": "如果我的团队刚开始做RAG，应该选哪个？",  # 继续追问
                "ground_truth": "刚开始做RAG建议选Chroma，它轻量易用，适合原型开发和本地验证。等业务规模扩大后再迁移到Pinecone或Milvus。",
            },
        ],
    },
    {
        "id": "conv_advanced",
        "name": "高级RAG技术追问",
        "turns": [
            {
                "question": "Self-RAG和CRAG分别解决什么问题？",
                "ground_truth": "Self-RAG解决'要不要检索'问题，通过反思令牌让模型自主判断是否需要外部知识；CRAG解决'检索结果够不够好'问题，对检索结果质量打分，不合格时触发网络搜索纠偏。",
            },
            {
                "question": "Graph RAG和Agentic RAG又分别解决什么？",  # 接上文
                "ground_truth": "Graph RAG解决多跳关系推理问题，通过知识图谱遍历找到实体间的关系链；Agentic RAG整合多种策略，Agent动态决定使用哪种检索方式并评估结果质量。",
            },
            {
                "question": "这四种技术的演进关系是什么？",  # "这四种" = 上面提到的
                "ground_truth": "四种技术循序渐进：Self-RAG（决定要不要检索）→ CRAG（检索后纠偏）→ Graph RAG（解决关系推理）→ Agentic RAG（动态策略选择+质量评估），是从单点优化到系统性智能化的演进。",
            },
        ],
    },
]

# ─── Build Vector Index ─────────────────────────────────────────────────────────

print("=" * 70)
print("  Building Vector Index")
print("=" * 70)

vectorstore = Chroma.from_documents(
    documents=DOCUMENTS,
    embedding=embeddings,
    collection_name="conv_rag_demo",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print(f"Vector index built: {len(DOCUMENTS)} documents")

# ─── Pipeline 1: Baseline RAG (no history awareness) ───────────────────────────

BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个RAG技术专家。根据以下参考资料回答问题。\n"
     "参考资料：\n{context}"),
    ("human", "{input}"),
])
base_qa_chain = create_stuff_documents_chain(llm, BASE_PROMPT)
base_chain    = create_retrieval_chain(retriever, base_qa_chain)

# ─── Pipeline 2: History-Aware Conversational RAG ───────────────────────────────

# Step 1: Contextualize question — rewrite question given chat history
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史和最新问题，将最新问题改写为一个独立完整的问题。\n"
     "要求：\n"
     "- 替换所有代词（它、这个、这些、其中等）为具体名词\n"
     "- 补全省略的主语或宾语\n"
     "- 只输出改写后的问题，不加任何解释\n"
     "如果问题本身已经完整独立，原样返回。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def _extract_standalone_question(text: str) -> str:
    """Keep only the first non-empty line to prevent verbose LLM output
    from exceeding the embedding model's 512-token limit."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    question = lines[0] if lines else text
    # Hard cap: embedding model (bge-large-zh-v1.5) accepts ≤512 tokens
    return question[:400]

# Build history-aware retriever manually to add the question-extraction step.
# When there is no chat history, pass the raw input directly to the retriever.
# When there is chat history, rewrite → extract first line → retrieve.
_contextualize_chain = (
    CONTEXTUALIZE_PROMPT
    | llm
    | StrOutputParser()
    | RunnableLambda(_extract_standalone_question)
)

history_aware_retriever = RunnableBranch(
    (
        lambda x: not x.get("chat_history"),
        (lambda x: x["input"]) | retriever,
    ),
    _contextualize_chain | retriever,
)

# Step 2: Answer using retrieved docs and history
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个RAG技术专家。根据以下参考资料回答问题。\n"
     "参考资料：\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Wrap with session-based memory
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conv_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ─── Run Conversations ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Running Multi-Turn Conversations")
print("=" * 70)

# Stores for RAGAS evaluation (last turn of each conversation)
baseline_eval_data = []
convrag_eval_data  = []

for conv in CONVERSATIONS:
    print(f"\n{'─' * 60}")
    print(f"  Conversation: {conv['name']}")
    print(f"{'─' * 60}")

    # Reset baseline history manually
    baseline_history: list[dict] = []

    for t_idx, turn in enumerate(conv["turns"]):
        q = turn["question"]
        print(f"\n  Turn {t_idx + 1}: {q}")

        # ── Baseline: retrieve on raw question ──
        # Build messages manually for baseline (just pass raw question, no history rewriting)
        base_result = base_chain.invoke({"input": q})
        base_answer = base_result["answer"]
        base_docs   = base_result["context"]

        # ── Conversational RAG ──
        conv_result = conv_rag.invoke(
            {"input": q},
            config={"configurable": {"session_id": conv["id"]}},
        )
        conv_answer = conv_result["answer"]
        conv_docs   = conv_result["context"]

        # Show retrieved docs comparison for Turn 2 (the tricky pronoun turn)
        if t_idx == 1:
            print(f"\n  [Turn {t_idx + 1} — Retrieval Comparison]")
            print(f"  Baseline retrieved:")
            for i, d in enumerate(base_docs[:2]):
                preview = d.page_content.replace('\n', ' ')[:80]
                print(f"    doc{i+1}: {preview}...")
            print(f"  ConvRAG retrieved:")
            for i, d in enumerate(conv_docs[:2]):
                preview = d.page_content.replace('\n', ' ')[:80]
                print(f"    doc{i+1}: {preview}...")

        print(f"\n  Baseline: {base_answer[:120]}...")
        print(f"  ConvRAG:  {conv_answer[:120]}...")

        # Collect last turn for RAGAS
        if t_idx == len(conv["turns"]) - 1:
            baseline_eval_data.append({
                "question":     q,
                "answer":       base_answer,
                "contexts":     [d.page_content for d in base_docs],
                "ground_truth": turn["ground_truth"],
                "conv_name":    conv["name"],
            })
            convrag_eval_data.append({
                "question":     q,
                "answer":       conv_answer,
                "contexts":     [d.page_content for d in conv_docs],
                "ground_truth": turn["ground_truth"],
                "conv_name":    conv["name"],
            })

# ─── Standalone Question Rewriting Demo ─────────────────────────────────────────

print("\n" + "=" * 70)
print("  Question Rewriting Demo (Turn 2 of each conversation)")
print("=" * 70)

# Show what the contextualization prompt rewrites
DEMO_HISTORY = [
    {
        "conv": "RAGAS追问",
        "history": [("RAGAS是什么？",
                     "RAGAS是专为RAG系统设计的评估框架。")],
        "followup": "它有哪四个核心指标？",
    },
    {
        "conv": "向量数据库追问",
        "history": [("常见的向量数据库有哪些？",
                     "常见向量数据库包括Chroma、Pinecone、Milvus、Qdrant。")],
        "followup": "其中哪个最适合生产环境？",
    },
    {
        "conv": "高级RAG追问",
        "history": [("Self-RAG和CRAG分别解决什么问题？",
                     "Self-RAG解决要不要检索；CRAG解决检索质量纠偏。")],
        "followup": "Graph RAG和Agentic RAG又分别解决什么？",
    },
]

contextualize_chain = CONTEXTUALIZE_PROMPT | llm

for demo in DEMO_HISTORY:
    history_msgs = []
    for h_q, h_a in demo["history"]:
        history_msgs.append(HumanMessage(content=h_q))
        history_msgs.append(AIMessage(content=h_a))

    rewritten = contextualize_chain.invoke({
        "chat_history": history_msgs,
        "input": demo["followup"],
    })
    rewritten_text = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
    print(f"\n  [{demo['conv']}]")
    print(f"  原始问题: {demo['followup']}")
    print(f"  改写后:   {rewritten_text.strip()}")

# ─── RAGAS Evaluation ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  RAGAS Evaluation (last turn of each conversation)")
print("=" * 70)

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import _LangchainEmbeddingsWrapper as LangchainEmbeddingsWrapper
    from ragas.llms import _LangchainLLMWrapper as LangchainLLMWrapper
    from ragas.metrics import answer_relevancy as ragas_answer_relevancy
    from ragas.metrics import context_precision as ragas_context_precision
    from ragas.metrics import context_recall as ragas_context_recall
    from ragas.metrics import faithfulness as ragas_faithfulness

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    metrics   = [ragas_faithfulness, ragas_answer_relevancy,
                 ragas_context_precision, ragas_context_recall]
    for m in metrics:
        m.llm        = ragas_llm
        m.embeddings = ragas_emb

    def make_ds(data):
        return Dataset.from_dict({
            "question":    [r["question"]     for r in data],
            "answer":      [r["answer"]       for r in data],
            "contexts":    [r["contexts"]     for r in data],
            "ground_truth":[r["ground_truth"] for r in data],
        })

    print("Evaluating Baseline RAG (last turns)...")
    baseline_scores = evaluate(make_ds(baseline_eval_data), metrics=metrics)
    print("Evaluating Conversational RAG (last turns)...")
    convrag_scores  = evaluate(make_ds(convrag_eval_data),  metrics=metrics)

    bm = baseline_scores.to_pandas().mean(numeric_only=True)
    cm = convrag_scores.to_pandas().mean(numeric_only=True)

    print("\n" + "=" * 70)
    print("  RAGAS Metrics: Baseline RAG vs Conversational RAG (last turns)")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'Baseline':>12} {'ConvRAG':>12} {'Delta':>10}")
    print("  " + "─" * 63)

    keys = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    deltas = {k: float(cm.get(k, 0)) - float(bm.get(k, 0)) for k in keys}
    best = max(deltas, key=lambda k: abs(deltas[k]))

    for key in keys:
        b = float(bm.get(key, 0))
        c = float(cm.get(key, 0))
        d = c - b
        arrow  = "↑" if d > 0.01 else ("↓" if d < -0.01 else "→")
        marker = "  ◀" if key == best else ""
        print(f"  {key:<25} {b:>12.3f} {c:>12.3f} {arrow}{d:>+9.3f}{marker}")

    print("=" * 70)

    # Per-conversation breakdown
    print("\n  Per-conversation breakdown (last turn):")
    b_df = baseline_scores.to_pandas()
    c_df = convrag_scores.to_pandas()
    for i, conv in enumerate(CONVERSATIONS):
        print(f"\n  [{conv['name']}]")
        for key in ["context_recall", "context_precision"]:
            bv = float(b_df.iloc[i].get(key, 0))
            cv = float(c_df.iloc[i].get(key, 0))
            d  = cv - bv
            print(f"    {key:<25} baseline={bv:.3f}  convrag={cv:.3f}  Δ={d:+.3f}")

    report = {
        "baseline": {k: float(bm.get(k, 0)) for k in keys},
        "conv_rag":  {k: float(cm.get(k, 0)) for k in keys},
        "note": "Evaluated on last turn (turn 3) of each 3-turn conversation",
    }
    with open("conv_rag_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nReport saved: conv_rag_report.json")

except Exception as e:
    print(f"RAGAS error: {e}")
    import traceback; traceback.print_exc()
