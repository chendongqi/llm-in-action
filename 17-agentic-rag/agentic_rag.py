"""
Article 17: Agentic RAG — Dynamic Multi-Strategy Retrieval

An agent that decides at runtime which retrieval strategy to use,
and can switch strategies when the first attempt falls short.

Three retrieval strategies (learned from articles 14-16):
  - vector:  top-k semantic similarity (baseline)
  - graph:   BFS traversal on knowledge graph (relational questions)
  - web:     DuckDuckGo search + LLM refinement (out-of-scope questions)
  - direct:  no retrieval (general knowledge, math, small talk)

Agent flow:
  question
    ↓
  [classify]  → pick initial strategy based on question type
    ↓
  [retrieve]  → run the chosen strategy
    ↓
  [evaluate]  → score context quality (0.0–1.0)
    ↓
  good? ──yes──→ [generate] → answer
    │
    no
    ↓
  [re_route]  → pick a different strategy (max 2 attempts)
    ↓
  [retrieve] again ...
"""

import json
import os
from typing import Any, Literal

import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

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
QUALITY_THRESHOLD = 0.6   # below this → try a different strategy
MAX_ATTEMPTS      = 2     # max re-route cycles

# ─── Knowledge Base ────────────────────────────────────────────────────────────

DOCUMENTS = [
    Document(page_content="""
RAG（Retrieval-Augmented Generation）是将外部知识检索与大语言模型生成相结合的技术。
RAG核心流程：检索（Retrieval）→ 增强（Augmentation）→ 生成（Generation）。
RAG由Meta AI在2020年提出，解决了大语言模型的知识截止问题。
""", metadata={"source": "rag-intro", "topic": "RAG基础"}),

    Document(page_content="""
RAGAS是专为RAG系统设计的评估框架，包含四个核心指标：
context_recall（上下文召回率）：检索内容是否覆盖了答案所需信息。
context_precision（上下文精确率）：检索内容中相关文档的比例。
faithfulness（忠实度）：答案是否完全基于检索到的上下文，衡量幻觉程度。
answer_relevancy（答案相关性）：生成答案与问题的相关程度。
RAGAS由Es等人在2023年提出，支持与LangChain集成。
""", metadata={"source": "ragas", "topic": "RAGAS评估"}),

    Document(page_content="""
向量数据库是RAG系统的核心组件，负责存储和检索文档的向量表示。
常见向量数据库：Chroma（轻量，本地开发）、Pinecone（云托管，生产级）、
Milvus（高性能，大规模部署）、Qdrant（Rust实现，高效）。
向量数据库使用HNSW算法加速近似最近邻搜索。
Chroma和Qdrant适合原型开发，Pinecone和Milvus适合企业级应用。
""", metadata={"source": "vector-db", "topic": "向量数据库"}),

    Document(page_content="""
Embedding模型将文本转换为高维向量，是RAG检索质量的基础。
中文场景推荐：BAAI/bge-large-zh-v1.5（北京智源，综合性能最佳），向量维度1024。
BGE模型由北京智源人工智能研究院（BAAI）开发。
Rerank模型bge-reranker-v2-m3也由BAAI开发，用于检索结果的重排序。
""", metadata={"source": "embedding", "topic": "Embedding模型"}),

    Document(page_content="""
Rerank（重排序）是RAG的后处理步骤，使用Cross-Encoder对初检结果重新评分排序。
Rerank流程：初检top-20 → rerank → 取top-4，在context_precision上典型提升+0.2。
常用Rerank模型：BAAI/bge-reranker-v2-m3（中英文均佳）。
SiliconFlow提供/v1/rerank接口兼容主流Rerank模型。
""", metadata={"source": "rerank", "topic": "Rerank重排序"}),

    Document(page_content="""
混合检索（Hybrid Search）结合向量检索和BM25关键词检索。
RRF（Reciprocal Rank Fusion）融合算法：score = Σ 1/(k+rank_i)，k通常取60。
EnsembleRetriever是LangChain提供的混合检索实现。
混合检索在context_precision上通常提升0.1-0.3。
""", metadata={"source": "hybrid", "topic": "混合检索"}),

    Document(page_content="""
Self-RAG通过反思令牌让模型自主决定是否检索，由Asai等人于2023年提出。
CRAG（Corrective RAG）在检索后评估质量，不合格时触发网络搜索，由Yan等人于2024年提出。
Graph RAG将文档构建为知识图谱，通过图遍历检索，适合多跳关系问题。
Agentic RAG进一步整合多种策略，由Agent动态决定最优检索路径。
""", metadata={"source": "advanced-rag", "topic": "高级RAG技术"}),

    Document(page_content="""
文档分块（Chunking）策略直接影响RAG检索质量。
固定大小分块（chunk_size通常512-1024字符），父子分块，上下文感知分块（Anthropic提出）。
父子分块：子chunk用于检索（精准），父chunk用于生成（完整）。
分块大小选择：问答场景推荐512，长文档摘要推荐1024-2048。
""", metadata={"source": "chunking", "topic": "文档分块"}),
]

# ─── Test Questions ─────────────────────────────────────────────────────────────

QUESTIONS = [
    # Category: factual → expected strategy: vector
    "RAGAS框架包含哪四个核心评估指标？",
    "向量数据库中Chroma和Milvus分别适合什么场景？",

    # Category: relational → expected strategy: graph
    "bge-large-zh-v1.5和bge-reranker-v2-m3都来自哪个机构，各自用于RAG的哪个阶段？",
    "Self-RAG、CRAG、Graph RAG分别解决了什么问题？",

    # Category: current/out-of-scope → expected strategy: web
    "2025年最新发布的RAG相关论文有哪些？",
    "LangChain最新版本是多少？",

    # Category: general knowledge → expected strategy: direct
    "把'检索增强生成'翻译成英文是什么？",
    "Python中如何计算列表的平均值？",
]

GROUND_TRUTHS = [
    "RAGAS包含四个指标：context_recall、context_precision、faithfulness、answer_relevancy。",
    "Chroma适合原型开发和本地开发；Milvus适合高性能大规模部署的企业级应用。",
    "两者都来自北京智源人工智能研究院（BAAI）。bge-large-zh-v1.5是Embedding模型用于检索阶段；bge-reranker-v2-m3是Rerank模型用于重排序阶段。",
    "Self-RAG解决'要不要检索'；CRAG解决'检索结果够不够好'并触发网络搜索；Graph RAG解决多跳关系问题。",
    "2024年发布的GraphRAG（Microsoft）使用层级聚类；CRAG（2024）引入自动纠偏；还有HippoRAG等。",
    "LangChain版本持续更新，需查看官方发布说明。",
    "检索增强生成的英文是Retrieval-Augmented Generation，缩写RAG。",
    "Python计算列表平均值：sum(lst)/len(lst) 或 import statistics; statistics.mean(lst)。",
]

# ─── LangGraph State ────────────────────────────────────────────────────────────

class AgenticRAGState(TypedDict):
    question:        str
    strategy:        str          # "vector" | "graph" | "web" | "direct"
    tried_strategies: list[str]   # strategies already attempted
    retrieved_docs:  list[Document]
    quality_score:   float        # context quality 0.0–1.0
    answer:          str
    path:            list[str]    # execution trace

# ─── Build Vector Index ─────────────────────────────────────────────────────────

print("=" * 70)
print("  Building Vector Index")
print("=" * 70)

vectorstore = Chroma.from_documents(
    documents=DOCUMENTS,
    embedding=embeddings,
    collection_name="agentic_rag_demo",
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print(f"Vector index built: {len(DOCUMENTS)} documents")

# ─── Build Knowledge Graph ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Building Knowledge Graph")
print("=" * 70)

TRIPLE_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "从文本中提取实体和关系，每行一个三元组，格式：实体A | 关系 | 实体B\n"
     "只输出三元组，不加编号或解释，每篇提取6-12个。"),
    ("human", "文本：\n{text}"),
])
triple_chain = TRIPLE_EXTRACT_PROMPT | llm | StrOutputParser()

KG = nx.DiGraph()

for i, doc in enumerate(DOCUMENTS):
    try:
        raw = triple_chain.invoke({"text": doc.page_content})
        count = 0
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                h, r, t = parts
                KG.add_node(h, source=doc.metadata["source"])
                KG.add_node(t, source=doc.metadata["source"])
                KG.add_edge(h, t, relation=r)
                count += 1
        print(f"  [{i+1}/{len(DOCUMENTS)}] {doc.metadata['topic']}: +{count} triples")
    except Exception as e:
        print(f"  [{i+1}/{len(DOCUMENTS)}] {doc.metadata['topic']}: error — {e}")

print(f"\nKnowledge graph: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")

# ─── Node: Classify → pick initial strategy ─────────────────────────────────────

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "判断以下问题最适合哪种检索策略，只输出策略名，不加解释：\n\n"
     "vector  - 需要检索知识库，问题是事实型（定义、参数、步骤、比较）\n"
     "graph   - 需要检索知识库，问题涉及多实体之间的关系（来自哪里、谁开发了什么）\n"
     "web     - 需要最新信息，知识库可能没有（最新版本、今日新闻、近期论文）\n"
     "direct  - 不需要检索（常识、数学计算、翻译、编程语法）\n"),
    ("human", "问题：{question}"),
])
classify_chain = CLASSIFY_PROMPT | llm | StrOutputParser()


def classify_node(state: AgenticRAGState) -> AgenticRAGState:
    raw = classify_chain.invoke({"question": state["question"]}).strip().lower()
    strategy = "vector"  # default
    for s in ["vector", "graph", "web", "direct"]:
        if s in raw:
            strategy = s
            break
    return {
        **state,
        "strategy": strategy,
        "tried_strategies": [strategy],
        "path": state.get("path", []) + [f"classify→{strategy}"],
    }

# ─── Node: Vector Retrieve ───────────────────────────────────────────────────────

def vector_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    docs = vector_retriever.invoke(state["question"])
    return {
        **state,
        "retrieved_docs": docs,
        "path": state["path"] + ["vector_retrieve"],
    }

# ─── Node: Graph Retrieve ────────────────────────────────────────────────────────

ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "从问题中提取关键实体，每行一个，不超过5个，不加解释。"),
    ("human", "问题：{question}"),
])
entity_chain = ENTITY_PROMPT | llm | StrOutputParser()


def graph_retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    raw = entity_chain.invoke({"question": state["question"]})
    entities = [e.strip() for e in raw.strip().splitlines() if e.strip()]

    seed_nodes = []
    for entity in entities:
        ent_lower = entity.lower()
        for node in KG.nodes:
            if ent_lower in node.lower() or node.lower() in ent_lower:
                seed_nodes.append(node)
    seed_nodes = list(dict.fromkeys(seed_nodes))

    if not seed_nodes:
        # Fallback to vector if no graph match
        docs = vector_retriever.invoke(state["question"])
        return {
            **state,
            "retrieved_docs": docs,
            "path": state["path"] + ["graph_retrieve(no_match→vector_fallback)"],
        }

    visited = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(2):  # 2-hop BFS
        nxt = set()
        for node in frontier:
            nxt |= (set(KG.successors(node)) | set(KG.predecessors(node))) - visited
        visited |= nxt
        frontier = nxt

    triples = [
        f"{u} --[{d['relation']}]--> {v}"
        for u, v, d in KG.edges(data=True)
        if u in visited or v in visited
    ]

    graph_doc = Document(
        page_content=(
            f"[图谱实体]: {', '.join(list(visited)[:20])}\n\n"
            f"[图谱关系]:\n" + "\n".join(triples[:40])
        ),
        metadata={"source": "knowledge_graph", "nodes": len(visited)},
    )
    # Supplement with 2 vector docs
    vector_docs = vector_retriever.invoke(state["question"])[:2]

    return {
        **state,
        "retrieved_docs": [graph_doc] + vector_docs,
        "path": state["path"] + [f"graph_retrieve(nodes={len(visited)})"],
    }

# ─── Node: Web Search ────────────────────────────────────────────────────────────

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "从以下网络搜索结果中提取与问题最相关的关键信息，"
     "去除噪声，保留核心事实，整理成简洁参考资料。"),
    ("human", "问题：{question}\n\n搜索结果：\n{results}\n\n提取："),
])
refine_chain = REFINE_PROMPT | llm | StrOutputParser()


def web_search_node(state: AgenticRAGState) -> AgenticRAGState:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        raw = search.invoke(state["question"])
        refined = refine_chain.invoke({
            "question": state["question"],
            "results": raw[:2000],
        })
        web_doc = Document(page_content=refined, metadata={"source": "web_search"})
        docs = [web_doc]
    except Exception as e:
        # Network unavailable — fall back to vector
        print(f"  [web_search] failed ({e}), falling back to vector")
        docs = vector_retriever.invoke(state["question"])

    return {
        **state,
        "retrieved_docs": docs,
        "path": state["path"] + ["web_search"],
    }

# ─── Node: Direct Generate (no retrieval) ────────────────────────────────────────

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识丰富的助手。请直接回答以下问题。"),
    ("human", "{question}"),
])
direct_chain = DIRECT_PROMPT | llm | StrOutputParser()


def direct_generate_node(state: AgenticRAGState) -> AgenticRAGState:
    answer = direct_chain.invoke({"question": state["question"]})
    return {
        **state,
        "answer": answer,
        "quality_score": 1.0,   # direct generation always "passes"
        "path": state["path"] + ["direct_generate"],
    }

# ─── Node: Evaluate context quality ──────────────────────────────────────────────

QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "评估以下检索到的上下文对回答问题的帮助程度，"
     "输出0.0到1.0之间的一个数字，不要任何解释：\n"
     "1.0 = 完全覆盖，可以直接回答\n"
     "0.5 = 部分相关，勉强可以回答\n"
     "0.0 = 完全不相关，无法回答"),
    ("human", "问题：{question}\n\n上下文：{context}"),
])
quality_chain = QUALITY_PROMPT | llm | StrOutputParser()


def evaluate_node(state: AgenticRAGState) -> AgenticRAGState:
    context = "\n\n".join(d.page_content[:300] for d in state["retrieved_docs"])
    raw = quality_chain.invoke({
        "question": state["question"],
        "context": context,
    })
    try:
        score = float(raw.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5  # parse fail → neutral
    return {
        **state,
        "quality_score": score,
        "path": state["path"] + [f"evaluate(score={score:.2f})"],
    }

# ─── Node: Re-route to a different strategy ──────────────────────────────────────

STRATEGY_ORDER = ["vector", "graph", "web"]


def re_route_node(state: AgenticRAGState) -> AgenticRAGState:
    tried = set(state.get("tried_strategies", []))
    # Try strategies in order, skip already-tried ones
    next_strategy = None
    for s in STRATEGY_ORDER:
        if s not in tried:
            next_strategy = s
            break

    if next_strategy is None:
        next_strategy = "vector"  # last resort

    tried.add(next_strategy)
    return {
        **state,
        "strategy": next_strategy,
        "tried_strategies": list(tried),
        "path": state["path"] + [f"re_route→{next_strategy}"],
    }

# ─── Node: Generate Answer ───────────────────────────────────────────────────────

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是RAG技术专家。根据以下参考资料回答问题。\n参考资料：\n{context}"),
    ("human", "{question}"),
])
generate_chain = GENERATE_PROMPT | llm | StrOutputParser()


def generate_node(state: AgenticRAGState) -> AgenticRAGState:
    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])
    answer = generate_chain.invoke({
        "context": context,
        "question": state["question"],
    })
    return {
        **state,
        "answer": answer,
        "path": state["path"] + ["generate"],
    }

# ─── Routing Functions ───────────────────────────────────────────────────────────

def route_after_classify(state: AgenticRAGState) -> str:
    return state["strategy"]  # "vector" | "graph" | "web" | "direct"


def route_after_evaluate(state: AgenticRAGState) -> str:
    score = state["quality_score"]
    attempts = len(state["tried_strategies"])
    if score >= QUALITY_THRESHOLD or attempts >= MAX_ATTEMPTS:
        return "generate"
    return "re_route"


def route_after_reroute(state: AgenticRAGState) -> str:
    return state["strategy"]  # "vector" | "graph" | "web"

# ─── Build LangGraph ────────────────────────────────────────────────────────────

graph = StateGraph(AgenticRAGState)

# Add nodes
graph.add_node("classify",         classify_node)
graph.add_node("vector_retrieve",  vector_retrieve_node)
graph.add_node("graph_retrieve",   graph_retrieve_node)
graph.add_node("web_search",       web_search_node)
graph.add_node("direct_generate",  direct_generate_node)
graph.add_node("evaluate",         evaluate_node)
graph.add_node("re_route",         re_route_node)
graph.add_node("generate",         generate_node)

# Entry
graph.set_entry_point("classify")

# Classify → dispatch by strategy
graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "vector": "vector_retrieve",
        "graph":  "graph_retrieve",
        "web":    "web_search",
        "direct": "direct_generate",
    },
)

# All retrieve paths → evaluate
graph.add_edge("vector_retrieve", "evaluate")
graph.add_edge("graph_retrieve",  "evaluate")
graph.add_edge("web_search",      "evaluate")

# Evaluate → generate or re_route
graph.add_conditional_edges(
    "evaluate",
    route_after_evaluate,
    {"generate": "generate", "re_route": "re_route"},
)

# Re-route → retry retrieve
graph.add_conditional_edges(
    "re_route",
    route_after_reroute,
    {
        "vector": "vector_retrieve",
        "graph":  "graph_retrieve",
        "web":    "web_search",
    },
)

# Terminals
graph.add_edge("generate",        END)
graph.add_edge("direct_generate", END)

agent = graph.compile()

# ─── Run Experiments ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Running Agentic RAG Experiments")
print("=" * 70)

results = []
for i, q in enumerate(QUESTIONS):
    print(f"\nQ{i+1}: {q[:65]}...")
    initial_state: AgenticRAGState = {
        "question":          q,
        "strategy":          "",
        "tried_strategies":  [],
        "retrieved_docs":    [],
        "quality_score":     0.0,
        "answer":            "",
        "path":              [],
    }
    final = agent.invoke(initial_state)
    path_str = " → ".join(final["path"])
    print(f"  Path:  {path_str}")
    print(f"  Score: {final['quality_score']:.2f}  |  "
          f"Strategies tried: {final['tried_strategies']}")
    results.append({
        "question":     q,
        "answer":       final["answer"],
        "contexts":     [d.page_content for d in final.get("retrieved_docs", [])],
        "ground_truth": GROUND_TRUTHS[i],
        "path":         path_str,
        "strategies":   final["tried_strategies"],
        "quality":      final["quality_score"],
    })

# ─── Routing Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Routing Summary")
print("=" * 70)
from collections import Counter
initial_strategies = Counter(r["strategies"][0] for r in results)
rerouted           = sum(1 for r in results if len(r["strategies"]) > 1)
print(f"\n  Initial strategy distribution:")
for s, c in sorted(initial_strategies.items()):
    print(f"    {s:<10} {c} questions")
print(f"\n  Re-routed (quality too low): {rerouted} / {len(results)}")
print(f"\n  Execution paths:")
for i, r in enumerate(results):
    print(f"  Q{i+1}: {r['path']}")

# ─── RAGAS Evaluation ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  RAGAS Evaluation")
print("=" * 70)

# Separate RAG questions (not direct) for fair evaluation
rag_results   = [r for r in results if "direct_generate" not in r["path"]]
direct_count  = len(results) - len(rag_results)
print(f"\n  ({direct_count} direct-generation questions excluded from RAGAS eval)")

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

    ds = Dataset.from_dict({
        "question":    [r["question"]     for r in rag_results],
        "answer":      [r["answer"]       for r in rag_results],
        "contexts":    [r["contexts"]     for r in rag_results],
        "ground_truth":[r["ground_truth"] for r in rag_results],
    })

    # Baseline: always vector retrieval (same questions)
    print("  Evaluating Always-Vector (baseline)...")
    baseline_results = []
    for r in rag_results:
        docs = vector_retriever.invoke(r["question"])
        ctx  = "\n\n".join(d.page_content for d in docs)
        ans  = generate_chain.invoke({"context": ctx, "question": r["question"]})
        baseline_results.append({
            "question":     r["question"],
            "answer":       ans,
            "contexts":     [d.page_content for d in docs],
            "ground_truth": r["ground_truth"],
        })

    baseline_ds = Dataset.from_dict({
        "question":    [r["question"]     for r in baseline_results],
        "answer":      [r["answer"]       for r in baseline_results],
        "contexts":    [r["contexts"]     for r in baseline_results],
        "ground_truth":[r["ground_truth"] for r in baseline_results],
    })

    print("  Evaluating Agentic RAG...")
    agent_scores   = evaluate(ds,          metrics=metrics)
    baseline_scores = evaluate(baseline_ds, metrics=metrics)

    am = agent_scores.to_pandas().mean(numeric_only=True)
    bm = baseline_scores.to_pandas().mean(numeric_only=True)

    print("\n" + "=" * 70)
    print("  RAGAS Metrics: Always-Vector vs Agentic RAG")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'Always-Vector':>14} {'Agentic RAG':>14} {'Delta':>10}")
    print("  " + "─" * 67)

    keys = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    deltas = {k: float(am.get(k, 0)) - float(bm.get(k, 0)) for k in keys}
    max_delta_key = max(deltas, key=lambda k: abs(deltas[k]))

    for key in keys:
        b = float(bm.get(key, 0))
        a = float(am.get(key, 0))
        d = a - b
        arrow  = "↑" if d > 0.01 else ("↓" if d < -0.01 else "→")
        marker = "  ◀" if key == max_delta_key else ""
        print(f"  {key:<25} {b:>14.3f} {a:>14.3f} {arrow}{d:>+9.3f}{marker}")

    print("=" * 70)

    report = {
        "always_vector": {k: float(bm.get(k, 0)) for k in keys},
        "agentic_rag":   {k: float(am.get(k, 0)) for k in keys},
        "routing": {
            "initial_distribution": dict(initial_strategies),
            "rerouted_count": rerouted,
            "total_questions": len(results),
            "direct_count": direct_count,
        },
    }
    with open("agentic_rag_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nReport saved: agentic_rag_report.json")

except Exception as e:
    print(f"RAGAS evaluation error: {e}")
    import traceback; traceback.print_exc()
