"""
Article 16: Graph RAG — Knowledge Graph Enhanced Retrieval

Two pipelines compared on relation-heavy multi-hop questions:
  - Vector RAG:  top-k semantic similarity
  - Graph RAG:   entity extraction → graph build → traversal → context assembly

Architecture:
  Build phase:
    documents → LLMGraphTransformer → (entity, relation, entity) triples → NetworkX graph
    documents → ChromaDB → vector index (baseline)

  Query phase (Graph RAG):
    question → entity extraction → graph entity match → BFS k-hop traversal
             → collect neighbor nodes/edges as context → generate answer

  Query phase (Vector RAG):
    question → embedding → top-k similarity → generate answer
"""

import json
import os
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ─── LLM / Embeddings ────────────────────────────────────────────────────────

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

EMB_BASE_URL = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMB_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")
EMB_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    base_url=EMB_BASE_URL,
    api_key=EMB_API_KEY,
    model=EMB_MODEL,
)

TOP_K = 4

# ─── Knowledge Base Documents ────────────────────────────────────────────────

DOCUMENTS = [
    Document(page_content="""
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识检索与大语言模型生成相结合的技术。
RAG的核心流程包括三个步骤：检索（Retrieval）、增强（Augmentation）和生成（Generation）。
RAG解决了大语言模型知识截止问题，让模型能够获取最新信息。
RAG由Meta AI在2020年提出，最初用于开放域问答任务。
""", metadata={"source": "rag-intro", "topic": "RAG基础"}),

    Document(page_content="""
向量数据库是RAG系统的核心组件，负责存储和检索文档的向量表示。
常见向量数据库包括：Chroma（轻量级，适合本地开发）、Pinecone（云托管，生产级）、
Weaviate（开源，支持混合检索）、Milvus（高性能，大规模部署）、Qdrant（Rust实现，高效）。
Chroma适合原型开发，Pinecone和Milvus适合企业级应用。
向量数据库使用HNSW（Hierarchical Navigable Small World）算法加速近似最近邻搜索。
""", metadata={"source": "vector-db", "topic": "向量数据库"}),

    Document(page_content="""
Embedding模型将文本转换为高维向量，是RAG检索质量的基础。
中文场景推荐模型：BAAI/bge-large-zh-v1.5（北京智源，综合性能最佳）、
text2vec-large-chinese（适合通用中文场景）、m3e-large（多语言场景）。
OpenAI的text-embedding-ada-002适合英文场景，维度1536。
bge-large-zh-v1.5向量维度为1024，在MTEB中文榜单排名靠前。
BGE模型由北京智源人工智能研究院（BAAI）开发。
""", metadata={"source": "embedding", "topic": "Embedding模型"}),

    Document(page_content="""
文档分块（Chunking）策略直接影响RAG的检索质量。
固定大小分块（Fixed-size chunking）：简单高效，chunk_size通常512-1024字符，overlap=50-200。
语义分块（Semantic chunking）：按段落或语义边界切分，保持上下文完整性。
父子分块（Parent-Child chunking）：子chunk用于检索（更精准），父chunk用于生成（更完整）。
上下文感知分块（Contextual chunking，Anthropic提出）：为每个chunk添加LLM生成的上下文描述。
分块大小选择：问答场景推荐512，长文档摘要推荐1024-2048。
""", metadata={"source": "chunking", "topic": "文档分块"}),

    Document(page_content="""
RAGAS是专为RAG系统设计的评估框架，提供四个核心指标。
context_recall（上下文召回率）：检索到的文档是否覆盖了回答所需的全部信息。
context_precision（上下文精确率）：检索到的文档中相关文档的比例，越高表示噪声越少。
faithfulness（忠实度）：生成答案是否完全基于检索到的上下文，衡量幻觉程度。
answer_relevancy（答案相关性）：生成答案与问题的相关程度。
RAGAS由Es等人在2023年提出，支持与LangChain集成。
""", metadata={"source": "ragas", "topic": "RAGAS评估"}),

    Document(page_content="""
混合检索（Hybrid Search）结合向量检索和BM25关键词检索，通常比单一方式效果更好。
RRF（Reciprocal Rank Fusion）是常用的混合检索融合算法，公式：score = Σ 1/(k+rank_i)，k通常取60。
BM25是经典的词频-逆文档频率检索算法，对精确关键词匹配效果好。
EnsembleRetriever是LangChain提供的混合检索实现，支持配置各检索器权重。
混合检索在context_precision上通常提升0.1-0.3，在专有名词和精确匹配场景收益最大。
""", metadata={"source": "hybrid-search", "topic": "混合检索"}),

    Document(page_content="""
Rerank（重排序）是RAG的后处理步骤，对初检结果重新排序提升精确率。
Cross-Encoder是主流Rerank实现，对query和每个文档进行联合编码，比双编码器更精准但更慢。
常用Rerank模型：BAAI/bge-reranker-v2-m3（中英文均佳）、ms-marco-MiniLM系列（英文）。
SiliconFlow提供/v1/rerank接口，兼容主流Rerank模型。
Rerank流程：初检top-20 → rerank → 取top-4，在context_precision上典型提升+0.2以上。
""", metadata={"source": "rerank", "topic": "Rerank重排序"}),

    Document(page_content="""
查询优化技术通过改写用户问题来提升检索效果。
Multi-Query（多查询）：用LLM将原始问题改写为多个视角的问题，分别检索后合并去重。
HyDE（Hypothetical Document Embeddings）：用LLM生成假设性答案，用答案的embedding检索。
查询分解（Query Decomposition）：将复杂问题拆解为多个子问题，分别检索后综合回答。
HyDE在上下文召回率上表现最好（+0.125），查询分解在复杂推理问题上优势明显。
查询优化与Rerank正交，可以叠加使用。
""", metadata={"source": "query-opt", "topic": "查询优化"}),

    Document(page_content="""
Self-RAG通过反思令牌（Reflection Tokens）让模型自主决定是否检索。
Self-RAG的四个反思令牌：[Retrieve]决定是否检索、[IsRel]判断文档相关性、
[IsSup]判断答案是否有文档支撑、[IsUse]评估答案对用户是否有用。
Self-RAG由Asai等人于2023年提出，使用LangGraph实现节点化流程。
Self-RAG对混合意图系统（部分问题需要检索，部分不需要）最有价值。
实测：Self-RAG的token消耗是普通RAG的2.4倍，但context_precision提升+0.104。
""", metadata={"source": "self-rag", "topic": "Self-RAG"}),

    Document(page_content="""
CRAG（Corrective RAG，纠偏RAG）在检索后评估结果质量，不合格时触发网络搜索。
CRAG对每篇检索文档打0-1的相关性分数，综合评分决定三种策略：
CORRECT（≥0.7）直接使用知识库文档，INCORRECT（≤0.3）触发网络搜索替代知识库，
AMBIGUOUS（0.3-0.7）合并知识库文档和网络搜索结果。
CRAG的网络搜索结果通过LLM精炼步骤去噪，提取核心信息。
实测：CRAG使context_precision从0.444提升到0.875（+0.431），是本系列最大单项提升。
CRAG由Yan等人于2024年提出，适合知识库覆盖不全的场景。
""", metadata={"source": "crag", "topic": "CRAG"}),

    Document(page_content="""
Graph RAG（知识图谱增强检索）将文档构建为知识图谱，通过图遍历而非相似度搜索来检索。
知识图谱由实体节点和关系边构成，例如（RAG, 使用, 向量数据库）。
Graph RAG对多跳关系问题（multi-hop reasoning）有天然优势：
通过图遍历可以找到间接相关的实体，而向量检索只能找到语义相似的文档。
LLMGraphTransformer可以从文本中自动提取实体和关系三元组。
Microsoft在2024年发布GraphRAG开源项目，使用层级聚类构建社区摘要。
Graph RAG适合：文档之间有复杂关系网络的场景，问题需要跨多个实体推理的场景。
""", metadata={"source": "graph-rag", "topic": "Graph RAG"}),

    Document(page_content="""
LangGraph是LangChain推出的工作流编排框架，专为LLM应用的有状态多步骤流程设计。
LangGraph核心概念：StateGraph（状态图）、节点（Node）、边（Edge）、条件边（Conditional Edge）。
LangGraph使用TypedDict定义State，节点函数接受State返回更新后的State。
LangGraph支持循环流程（self-RAG的重试机制），而LangChain的LCEL只支持线性流程。
LangGraph编译后的图（compiled graph）可以通过.invoke()同步调用或.stream()流式调用。
LangGraph与LangSmith集成，支持完整的追踪和可观测性。
""", metadata={"source": "langgraph", "topic": "LangGraph"}),
]

# ─── Test Questions (relation-heavy multi-hop) ───────────────────────────────

QUESTIONS = [
    # Single-hop: basic factual (both should handle well)
    "RAGAS框架包含哪四个核心评估指标？",
    "中文场景推荐哪个Embedding模型？为什么？",

    # Multi-hop relational: requires connecting multiple entities
    "CRAG和Self-RAG在解决问题的思路上有什么区别？",
    "Rerank和查询优化技术（如HyDE）各自提升了哪个RAGAS指标？",
    "父子分块和上下文感知分块分别是哪个机构提出的？解决了什么问题？",
    "从RAG到CRAG，检索质量评估经历了哪些演进步骤？",
    "bge-large-zh-v1.5和bge-reranker-v2-m3都来自哪个机构？各自在RAG中扮演什么角色？",
    "向量数据库中哪些适合企业级应用，哪些适合本地开发？各用什么算法？",
]

GROUND_TRUTHS = [
    "RAGAS包含四个核心指标：context_recall（上下文召回率）、context_precision（上下文精确率）、faithfulness（忠实度）和answer_relevancy（答案相关性）。",
    "中文场景推荐BAAI/bge-large-zh-v1.5，由北京智源人工智能研究院开发，在MTEB中文榜单排名靠前，向量维度1024，综合性能最佳。",
    "Self-RAG在检索前决策（要不要检索），通过反思令牌判断是否需要外部知识；CRAG在检索后评估（结果够不够好），对检索文档打分，不合格时触发网络搜索纠偏。",
    "Rerank主要提升context_precision（精确率），通过Cross-Encoder重排序减少噪声文档；HyDE主要提升context_recall（召回率），通过生成假设文档扩大语义覆盖范围。",
    "父子分块是通用策略；上下文感知分块（Contextual chunking）由Anthropic提出。两者都解决了标准固定分块丢失上下文的问题，提升检索精确度和召回率。",
    "从基础RAG（盲目检索）→ Rerank（对检索结果重排序）→ Self-RAG（检索前决策是否检索，检索后评估相关性）→ CRAG（检索后评分，不合格触发网络搜索纠偏）。",
    "两者都来自北京智源人工智能研究院（BAAI）。bge-large-zh-v1.5是Embedding模型，将文本转换为向量用于检索；bge-reranker-v2-m3是Rerank模型，对检索结果重排序提升精确率。",
    "企业级：Pinecone（云托管）和Milvus（高性能大规模），适合生产部署；本地开发：Chroma（轻量级）。向量数据库使用HNSW算法加速近似最近邻搜索。",
]

# ─── Phase 1: Build Vector Index ─────────────────────────────────────────────

print("=" * 70)
print("  Phase 1: Building Vector Index")
print("=" * 70)

vectorstore = Chroma.from_documents(
    documents=DOCUMENTS,
    embedding=embeddings,
    collection_name="graph_rag_demo",
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print(f"Vector index built: {len(DOCUMENTS)} documents")

# ─── Phase 2: Build Knowledge Graph ──────────────────────────────────────────

print("\n" + "=" * 70)
print("  Phase 2: Building Knowledge Graph (custom triple extraction)")
print("=" * 70)

TRIPLE_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "从以下文本中提取实体和关系，输出三元组列表。\n"
     "格式要求：每行一个三元组，格式严格为：实体A | 关系 | 实体B\n"
     "规则：\n"
     "- 实体用名词短语，不加括号或引号\n"
     "- 关系用动词短语，如：使用、包含、由...提出、适用于、优于\n"
     "- 每行只输出三元组，不要编号，不要解释，不要其他内容\n"
     "- 每篇文档提取8-15个三元组\n\n"
     "示例输出（格式参考）：\n"
     "RAG | 使用 | 向量检索\n"
     "RAGAS | 由...提出 | Es等人\n"
     "Chroma | 适用于 | 本地开发"),
    ("human", "文本：\n{text}"),
])

triple_chain = TRIPLE_EXTRACT_PROMPT | llm | StrOutputParser()


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (head, relation, tail) triples from text using LLM."""
    raw = triple_chain.invoke({"text": text})
    triples = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


KG = nx.DiGraph()

print("Extracting entities and relations...")
for i, doc in enumerate(DOCUMENTS):
    try:
        triples = extract_triples(doc.page_content)
        src = doc.metadata.get("source", "")
        for head, rel, tail in triples:
            if not KG.has_node(head):
                KG.add_node(head, source=src)
            if not KG.has_node(tail):
                KG.add_node(tail, source=src)
            KG.add_edge(head, tail, relation=rel, source_doc=src)
        print(f"  [{i+1:2d}/{len(DOCUMENTS)}] {doc.metadata['topic']}: "
              f"+{len(triples)} triples")
    except Exception as e:
        print(f"  [{i+1:2d}/{len(DOCUMENTS)}] {doc.metadata['topic']}: error — {e}")

print(f"\nKnowledge graph: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")

# ─── Phase 3: Graph RAG Retrieval ────────────────────────────────────────────

ENTITY_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "从用户问题中提取关键实体（技术术语、产品名、人名、机构名等），"
     "每行一个，不加编号或解释，不超过6个。"),
    ("human", "问题：{question}"),
])

entity_chain = ENTITY_EXTRACT_PROMPT | llm | StrOutputParser()


def extract_entities(question: str) -> list[str]:
    raw = entity_chain.invoke({"question": question})
    return [e.strip() for e in raw.strip().splitlines() if e.strip()]


def fuzzy_match_nodes(entity: str, graph: nx.DiGraph, threshold: int = 3) -> list[str]:
    """Find graph nodes that contain the entity string (case-insensitive substring match)."""
    entity_lower = entity.lower()
    matched = []
    for node in graph.nodes:
        node_lower = node.lower()
        if entity_lower in node_lower or node_lower in entity_lower:
            matched.append(node)
    return matched


def graph_retrieve(question: str, graph: nx.DiGraph, hops: int = 2) -> list[Document]:
    """
    Entity linking → BFS k-hop traversal → collect context documents.

    Returns a list of Document objects assembled from graph neighborhood.
    """
    entities = extract_entities(question)
    seed_nodes = []
    for entity in entities:
        matched = fuzzy_match_nodes(entity, graph)
        seed_nodes.extend(matched)

    seed_nodes = list(dict.fromkeys(seed_nodes))  # deduplicate, preserve order

    if not seed_nodes:
        return []

    # BFS: collect all nodes within `hops` hops from seed nodes
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            next_frontier |= neighbors - visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    # Build context: collect triples (entity → relation → entity) for all visited nodes
    triples = []
    for u, v, data in graph.edges(data=True):
        if u in visited or v in visited:
            triples.append(f"{u} --[{data.get('relation', 'RELATED_TO')}]--> {v}")

    # Also collect node type info
    node_info = []
    for node in visited:
        ndata = graph.nodes[node]
        node_info.append(f"{node} (type: {ndata.get('type', 'unknown')})")

    context_text = (
        f"[Graph entities found]: {', '.join(node_info[:20])}\n\n"
        f"[Graph relationships]:\n" + "\n".join(triples[:40])
    )

    return [Document(
        page_content=context_text,
        metadata={
            "source": "knowledge_graph",
            "seed_entities": str(entities),
            "visited_nodes": len(visited),
            "triples": len(triples),
        }
    )]


# ─── Phase 4: Generation Prompts ─────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个RAG技术专家。根据提供的参考资料回答问题。\n"
     "参考资料：\n{context}"),
    ("human", "{question}"),
])

rag_chain = RAG_PROMPT | llm | StrOutputParser()


def run_vector_rag(question: str) -> dict[str, Any]:
    docs = vector_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = rag_chain.invoke({"context": context, "question": question})
    return {"answer": answer, "docs": docs}


def run_graph_rag(question: str) -> dict[str, Any]:
    graph_docs = graph_retrieve(question, KG, hops=2)

    # Supplement with vector retrieval for factual grounding
    vector_docs = vector_retriever.invoke(question)[:2]

    all_docs = graph_docs + vector_docs
    context = "\n\n".join(d.page_content for d in all_docs)
    answer = rag_chain.invoke({"context": context, "question": question})
    return {"answer": answer, "docs": all_docs, "graph_docs": graph_docs}


# ─── Phase 5: Run Experiments ────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Phase 5: Running Experiments")
print("=" * 70)

vector_results = []
graph_results  = []

for i, q in enumerate(QUESTIONS):
    print(f"\nQ{i+1}: {q[:60]}...")

    vr = run_vector_rag(q)
    gr = run_graph_rag(q)

    if gr["graph_docs"]:
        gd = gr["graph_docs"][0]
        meta = gd.metadata
        graph_info = f"nodes={meta['visited_nodes']}, triples={meta['triples']}"
    else:
        graph_info = "no graph match"

    print(f"  Vector: {len(vr['docs'])} docs")
    print(f"  Graph:  {graph_info}")

    vector_results.append({
        "question": q,
        "answer": vr["answer"],
        "contexts": [d.page_content for d in vr["docs"]],
        "ground_truth": GROUND_TRUTHS[i],
    })
    graph_results.append({
        "question": q,
        "answer": gr["answer"],
        "contexts": [d.page_content for d in gr["docs"]],
        "ground_truth": GROUND_TRUTHS[i],
    })

# ─── Phase 6: RAGAS Evaluation ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Phase 6: RAGAS Evaluation")
print("=" * 70)

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import _LangchainEmbeddingsWrapper as LangchainEmbeddingsWrapper
    from ragas.llms import _LangchainLLMWrapper as LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy as ragas_answer_relevancy,
    )
    from ragas.metrics import (
        context_precision as ragas_context_precision,
    )
    from ragas.metrics import (
        context_recall as ragas_context_recall,
    )
    from ragas.metrics import (
        faithfulness as ragas_faithfulness,
    )

    ragas_llm   = LangchainLLMWrapper(llm)
    ragas_emb   = LangchainEmbeddingsWrapper(embeddings)
    metrics     = [ragas_faithfulness, ragas_answer_relevancy,
                   ragas_context_precision, ragas_context_recall]

    for m in metrics:
        m.llm       = ragas_llm
        m.embeddings = ragas_emb

    def build_dataset(results):
        return Dataset.from_dict({
            "question":    [r["question"]    for r in results],
            "answer":      [r["answer"]      for r in results],
            "contexts":    [r["contexts"]    for r in results],
            "ground_truth":[r["ground_truth"] for r in results],
        })

    print("Evaluating Vector RAG...")
    vector_scores = evaluate(build_dataset(vector_results), metrics=metrics)
    print("Evaluating Graph RAG...")
    graph_scores  = evaluate(build_dataset(graph_results),  metrics=metrics)

    vm = vector_scores.to_pandas().mean(numeric_only=True)
    gm = graph_scores.to_pandas().mean(numeric_only=True)

    print("\n" + "=" * 70)
    print("  RAGAS Metrics Comparison (Vector RAG vs Graph RAG)")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'Vector RAG':>12} {'Graph RAG':>12} {'Delta':>10}")
    print("  " + "─" * 63)

    metric_keys = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    for key in metric_keys:
        v = float(vm.get(key, 0))
        g = float(gm.get(key, 0))
        d = g - v
        arrow = "↑" if d > 0.01 else ("↓" if d < -0.01 else "→")
        marker = "  ◀" if abs(d) == max(abs(float(gm.get(k, 0)) - float(vm.get(k, 0))) for k in metric_keys) else ""
        print(f"  {key:<25} {v:>12.3f} {g:>12.3f} {arrow}{d:>+9.3f}{marker}")

    print("=" * 70)

    report = {
        "vector_rag": {k: float(vm.get(k, 0)) for k in metric_keys},
        "graph_rag":  {k: float(gm.get(k, 0)) for k in metric_keys},
        "kg_stats": {
            "nodes": KG.number_of_nodes(),
            "edges": KG.number_of_edges(),
        },
    }
    with open("graph_rag_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nReport saved: graph_rag_report.json")

except Exception as e:
    print(f"RAGAS evaluation error: {e}")
    import traceback
    traceback.print_exc()

# ─── Phase 7: Graph Statistics ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Knowledge Graph Statistics")
print("=" * 70)

degree_centrality = nx.degree_centrality(KG)
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"\nTop 10 hub entities (by degree centrality):")
for node, centrality in top_nodes:
    in_deg  = KG.in_degree(node)
    out_deg = KG.out_degree(node)
    print(f"  {node:<35} centrality={centrality:.3f}  in={in_deg}  out={out_deg}")

print(f"\nRelation type distribution:")
relation_counts: dict[str, int] = {}
for _, _, data in KG.edges(data=True):
    rel = data.get("relation", "UNKNOWN")
    relation_counts[rel] = relation_counts.get(rel, 0) + 1

for rel, cnt in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {rel:<35} {cnt}")
