"""
Article 19: Incremental RAG Updates — Keeping the Knowledge Base Fresh

Problem: naive RAG rebuilds the entire index when any document changes.
  - Embeds ALL documents every time (even the 95% that didn't change)
  - Costs money and time proportional to corpus size, not change size
  - Becomes impractical as the knowledge base grows

Solution: LangChain Indexing API (SQLRecordManager + index())
  - RecordManager stores a content hash for each indexed document
  - On re-index: skips unchanged (same hash), embeds only new/modified
  - With cleanup="full": also removes deleted documents automatically
  - Cost proportional to the change, not the total corpus size

This script demonstrates three scenarios:
  1. Initial index (V1 corpus, 6 docs)
  2. Incremental update (V2 corpus: 3 unchanged, 2 modified, 1 deleted, 2 added)
  3. Full rebuild comparison (same V2 corpus, but record manager wiped)
"""

import json
import os
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document
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


# ─── Counting Embeddings Wrapper ──────────────────────────────────────────────
# Wraps OpenAIEmbeddings to count how many document chunks are actually embedded.
# This makes the "full rebuild vs incremental" cost comparison concrete.

class CountingEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings that counts embed_documents() calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embed_count = 0

    def embed_documents(self, texts: list[str], *args, **kwargs) -> list[list[float]]:
        self._embed_count += len(texts)
        return super().embed_documents(texts, *args, **kwargs)

    def reset_count(self) -> None:
        self._embed_count = 0

    @property
    def embed_count(self) -> int:
        return self._embed_count


embeddings = CountingEmbeddings(
    base_url=EMB_BASE_URL,
    api_key=EMB_API_KEY,
    model=EMB_MODEL,
)

# ─── Document Corpus ──────────────────────────────────────────────────────────
# V1: 6 documents — the initial knowledge base
# V2: simulates a real update cycle:
#     - 3 documents UNCHANGED  (rag-intro, vector-db, rerank)
#     - 2 documents MODIFIED   (ragas: added details, chunking: updated)
#     - 1 document  DELETED    (embedding: removed)
#     - 2 documents ADDED      (advanced-rag, conv-rag: new content)

DOCS_V1 = [
    Document(
        page_content=(
            "RAG（Retrieval-Augmented Generation）是一种将外部知识检索与大语言模型结合的技术。"
            "RAG的核心流程：检索（Retrieval）→ 增强（Augmentation）→ 生成（Generation）。"
            "RAG由Meta AI在2020年提出，解决了LLM的知识截止问题和幻觉问题。"
        ),
        metadata={"source": "rag-intro", "version": 1},
    ),
    Document(
        page_content=(
            "RAGAS是专为RAG系统设计的评估框架，由Es等人在2023年提出。"
            "RAGAS的核心指标：context_recall、context_precision、faithfulness、answer_relevancy。"
        ),
        metadata={"source": "ragas", "version": 1},
    ),
    Document(
        page_content=(
            "向量数据库是RAG的核心存储组件，负责存储向量表示并支持相似度搜索。"
            "常见向量数据库：Chroma（开发）、Pinecone（云托管）、Milvus（企业部署）。"
        ),
        metadata={"source": "vector-db", "version": 1},
    ),
    Document(
        page_content=(
            "Embedding模型将文本转换为向量，决定语义检索的质量上限。"
            "中文场景推荐BAAI/bge-large-zh-v1.5，英文可选text-embedding-ada-002。"
        ),
        metadata={"source": "embedding", "version": 1},
    ),
    Document(
        page_content=(
            "Rerank（重排序）使用Cross-Encoder对初检结果重新评分，是提升精确率的关键步骤。"
            "常用Rerank模型：BAAI/bge-reranker-v2-m3（中英双语均佳）。"
        ),
        metadata={"source": "rerank", "version": 1},
    ),
    Document(
        page_content=(
            "文档分块策略影响RAG检索质量：固定大小分块适合通用场景。"
            "父子分块：子chunk用于精准检索，父chunk用于完整生成。"
        ),
        metadata={"source": "chunking", "version": 1},
    ),
]

DOCS_V2 = [
    # ── UNCHANGED (identical content to V1) ─────────────────────────────────
    Document(
        page_content=(
            "RAG（Retrieval-Augmented Generation）是一种将外部知识检索与大语言模型结合的技术。"
            "RAG的核心流程：检索（Retrieval）→ 增强（Augmentation）→ 生成（Generation）。"
            "RAG由Meta AI在2020年提出，解决了LLM的知识截止问题和幻觉问题。"
        ),
        metadata={"source": "rag-intro", "version": 1},    # same version
    ),
    Document(
        page_content=(
            "向量数据库是RAG的核心存储组件，负责存储向量表示并支持相似度搜索。"
            "常见向量数据库：Chroma（开发）、Pinecone（云托管）、Milvus（企业部署）。"
        ),
        metadata={"source": "vector-db", "version": 1},    # same version
    ),
    Document(
        page_content=(
            "Rerank（重排序）使用Cross-Encoder对初检结果重新评分，是提升精确率的关键步骤。"
            "常用Rerank模型：BAAI/bge-reranker-v2-m3（中英双语均佳）。"
        ),
        metadata={"source": "rerank", "version": 1},       # same version
    ),

    # ── MODIFIED (same source key, different content) ────────────────────────
    Document(
        page_content=(
            "RAGAS是专为RAG系统设计的评估框架，由Es等人在2023年提出。"
            "RAGAS的核心指标：context_recall、context_precision、faithfulness、answer_relevancy。"
            # ↑ 新增内容 ↓
            "其中faithfulness最难提升，因为需要LLM严格约束自己只说文档里有的内容。"
            "RAGAS支持使用本地LLM和Embedding模型运行评估，无需OpenAI API。"
        ),
        metadata={"source": "ragas", "version": 2},         # modified
    ),
    Document(
        page_content=(
            "文档分块策略影响RAG检索质量：固定大小分块适合通用场景。"
            "父子分块：子chunk用于精准检索，父chunk用于完整生成。"
            # ↑ 新增内容 ↓
            "上下文感知分块（Contextual Retrieval）：为每个chunk添加LLM生成的上下文描述。"
            "语义分块：按语义边界切分，而非固定字符数，保留完整的语义单元。"
        ),
        metadata={"source": "chunking", "version": 2},      # modified
    ),

    # ── ADDED (new sources not in V1) ────────────────────────────────────────
    Document(
        page_content=(
            "高级RAG架构包括Self-RAG、CRAG、Graph RAG和Agentic RAG。"
            "Self-RAG通过反思令牌决定是否检索；CRAG在检索后评估质量，不合格触发网络搜索。"
            "Graph RAG将文档构建为知识图谱，通过图遍历解决多跳关系推理问题。"
            "Agentic RAG由Agent动态决定检索策略，包含质量评估和重试机制。"
        ),
        metadata={"source": "advanced-rag", "version": 1},  # new
    ),
    Document(
        page_content=(
            "对话式RAG（Conversational RAG）在多轮对话中保持上下文理解能力。"
            "核心问题：追问使用代词（它、这个）时，直接检索会失效。"
            "解决方案：History-Aware Retriever用LLM将当前问题结合历史改写为独立完整的问题。"
        ),
        metadata={"source": "conv-rag", "version": 1},      # new
    ),

    # NOTE: "embedding" doc is NOT included → will be deleted with cleanup="full"
]

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR   = "./chroma_index"
DB_URL       = "sqlite:///record_manager.db"
NAMESPACE    = "chroma/rag_knowledge_base"
COLLECTION   = "rag_knowledge_base"
SOURCE_KEY   = "source"              # the metadata key that identifies a document


# ─── Helper: Pretty Print Index Result ───────────────────────────────────────

def print_index_result(label: str, result: dict, elapsed: float, embed_count: int) -> None:
    print(f"\n  [{label}]")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  added:    {result['num_added']:>4}  (newly embedded)         │")
    print(f"  │  skipped:  {result['num_skipped']:>4}  (content unchanged)    │")
    print(f"  │  deleted:  {result['num_deleted']:>4}  (removed/replaced)     │")
    print(f"  │  updated:  {result['num_updated']:>4}                         │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  embed calls: {embed_count:>4}                        │")
    print(f"  │  wall time:   {elapsed:.2f}s                       │")
    print(f"  └─────────────────────────────────────────┘")


# ─── Setup ────────────────────────────────────────────────────────────────────

# Clean up any previous run artifacts
for path in [CHROMA_DIR, "record_manager.db"]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

record_manager = SQLRecordManager(NAMESPACE, db_url=DB_URL)
record_manager.create_schema()

# ─── Scenario 1: Initial Index (V1) ──────────────────────────────────────────

print("=" * 70)
print("  Scenario 1: Initial Index (V1 — 6 documents)")
print("=" * 70)

embeddings.reset_count()
t0 = time.perf_counter()

result_v1 = index(
    DOCS_V1,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key=SOURCE_KEY,
)

elapsed_v1  = time.perf_counter() - t0
embed_v1    = embeddings.embed_count

print_index_result("Initial Index", result_v1, elapsed_v1, embed_v1)

# Verify what's in the vectorstore
all_docs = vectorstore._collection.get()
print(f"\n  Vectorstore now contains {len(all_docs['ids'])} documents")
print(f"  Sources: {sorted(set(m['source'] for m in all_docs['metadatas']))}")

# ─── Scenario 2: Incremental Update (V2) ─────────────────────────────────────

print("\n" + "=" * 70)
print("  Scenario 2: Incremental Update (V2 — changes from V1)")
print("=" * 70)
print("""
  Change summary:
    UNCHANGED: rag-intro, vector-db, rerank           (3 docs)
    MODIFIED:  ragas (added detail), chunking (added) (2 docs)
    DELETED:   embedding                              (1 doc)
    ADDED:     advanced-rag, conv-rag                 (2 docs)
""")

embeddings.reset_count()
t0 = time.perf_counter()

result_v2 = index(
    DOCS_V2,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key=SOURCE_KEY,
)

elapsed_v2 = time.perf_counter() - t0
embed_v2   = embeddings.embed_count

print_index_result("Incremental Update", result_v2, elapsed_v2, embed_v2)

all_docs = vectorstore._collection.get()
print(f"\n  Vectorstore now contains {len(all_docs['ids'])} documents")
print(f"  Sources: {sorted(set(m['source'] for m in all_docs['metadatas']))}")

# ─── Scenario 3: Full Rebuild (V2, no record manager) ────────────────────────
# Simulate what would happen without the Indexing API:
# wipe the record manager → everything looks "new" → embed all docs

print("\n" + "=" * 70)
print("  Scenario 3: Full Rebuild (V2 — record manager wiped)")
print("=" * 70)
print("  (Simulates a system without incremental tracking)")

# Reset record manager — make every doc look "unseen"
record_manager.delete_keys(record_manager.list_keys())

embeddings.reset_count()
t0 = time.perf_counter()

result_full = index(
    DOCS_V2,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key=SOURCE_KEY,
)

elapsed_full = time.perf_counter() - t0
embed_full   = embeddings.embed_count

print_index_result("Full Rebuild", result_full, elapsed_full, embed_full)

# ─── Comparison Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Cost Comparison: Incremental vs Full Rebuild")
print("=" * 70)
print(f"""
  ┌──────────────────────┬───────────────┬───────────────┐
  │                      │   Incremental │  Full Rebuild │
  ├──────────────────────┼───────────────┼───────────────┤
  │  Documents embedded  │    {embed_v2:>4}        │    {embed_full:>4}        │
  │  Documents skipped   │    {result_v2['num_skipped']:>4}        │       0       │
  │  Wall time           │   {elapsed_v2:>5.2f}s       │   {elapsed_full:>5.2f}s       │
  │  Embedding savings   │  {100*(embed_full-embed_v2)/max(embed_full,1):.0f}%           │   0%          │
  └──────────────────────┴───────────────┴───────────────┘

  At scale (1000 docs, 5% daily change):
    Full rebuild:   1000 embeds/day
    Incremental:      50 embeds/day  →  95% cost reduction
""")

# ─── Demo Query: Verify KB Is Correct After Update ───────────────────────────

print("=" * 70)
print("  Query Demo: Verify KB is correct after incremental update")
print("=" * 70)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个RAG技术专家。根据以下参考资料回答问题，保持简洁。\n"
     "参考资料：\n{context}"),
    ("human", "{input}"),
])
qa_chain  = create_stuff_documents_chain(llm, QA_PROMPT)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Q1: Tests updated RAGAS doc (V2 content should be present)
q1 = "RAGAS 中哪个指标最难提升，为什么？"
# Q2: Tests that "embedding" doc was deleted (should not appear)
q2 = "RAG 评估框架有哪些，对话式RAG解决什么问题？"

for q in [q1, q2]:
    result = rag_chain.invoke({"input": q})
    print(f"\n  Q: {q}")
    print(f"  A: {result['answer'][:200]}...")
    print(f"  Sources: {[d.metadata['source'] for d in result['context']]}")

# ─── Save Report ──────────────────────────────────────────────────────────────

report = {
    "initial_index": {
        "num_docs": len(DOCS_V1),
        "num_added":   result_v1["num_added"],
        "num_skipped": result_v1["num_skipped"],
        "num_deleted": result_v1["num_deleted"],
        "embed_calls": embed_v1,
        "elapsed_s":   round(elapsed_v1, 3),
    },
    "incremental_update": {
        "num_docs": len(DOCS_V2),
        "num_added":   result_v2["num_added"],
        "num_skipped": result_v2["num_skipped"],
        "num_deleted": result_v2["num_deleted"],
        "embed_calls": embed_v2,
        "elapsed_s":   round(elapsed_v2, 3),
    },
    "full_rebuild": {
        "num_docs": len(DOCS_V2),
        "num_added":   result_full["num_added"],
        "num_skipped": result_full["num_skipped"],
        "num_deleted": result_full["num_deleted"],
        "embed_calls": embed_full,
        "elapsed_s":   round(elapsed_full, 3),
    },
    "savings": {
        "embed_calls_saved": embed_full - embed_v2,
        "embed_savings_pct": round(100 * (embed_full - embed_v2) / max(embed_full, 1), 1),
        "time_saved_s":      round(elapsed_full - elapsed_v2, 3),
    },
}

with open("incremental_update_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("  Report saved to incremental_update_report.json")
print("=" * 70)
