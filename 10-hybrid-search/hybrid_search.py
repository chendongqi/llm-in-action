"""
混合检索实验：纯向量 vs 纯 BM25 vs 混合检索（RRF）

核心结论：
- BM25   擅长精确关键词匹配（型号、术语、参数值）
- Vector 擅长语义理解（同义词、换一种说法）
- Hybrid 两者兼顾，Precision 最高

运行方式：
    conda activate dev_base
    python hybrid_search.py
"""

import json
import os
import jieba
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH = "../08-ragas-eval/data/knowledge_base.json"
CHROMA_DIR = "./chroma_hybrid"
TOP_K = 3

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")

# ── 测试用例：3 个关键词查询 + 3 个语义查询 ───────────────────────────────────
#
# 关键词查询：包含文档中的精确术语/型号/参数，BM25 应胜出
# 语义查询：概念性问法，和文档用词不同，向量检索应胜出
#
TESTSET = [
    # ── 关键词查询 ─────────────────────────────────────────────────────────
    {
        "query": "BAAI/bge-large-zh-v1.5 维度",
        "type": "keyword",
        "relevant_doc_id": "doc-003",
        "note": "包含精确模型名，BM25 应能匹配",
    },
    {
        "query": "RRF score sum 1/(k+rank) 公式",
        "type": "keyword",
        "relevant_doc_id": "doc-006",
        "note": "包含精确公式字符串，BM25 应能匹配",
    },
    {
        "query": "chunk_size 256 1024 overlap 推荐",
        "type": "keyword",
        "relevant_doc_id": "doc-004",
        "note": "包含精确参数值，BM25 应能匹配",
    },
    # ── 语义查询 ─────────────────────────────────────────────────────────
    {
        "query": "AI 助手总是给出过时的答案，有什么方法让它了解最新信息",
        "type": "semantic",
        "relevant_doc_id": "doc-001",
        "note": "问知识更新问题，没有提到 RAG，向量检索应胜出",
    },
    {
        "query": "多个团队共用一套问答系统，怎么保证不同团队的资料互相看不到",
        "type": "semantic",
        "relevant_doc_id": "doc-008",
        "note": "问数据隔离，没有提到多租户，向量检索应胜出",
    },
    {
        "query": "换一种问法，检索结果就完全不同，怎么解决这种不稳定性",
        "type": "semantic",
        "relevant_doc_id": "doc-007",
        "note": "问查询稳定性问题，没有 Multi-Query/HyDE 等关键词，BM25 完全 miss",
    },
]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def chinese_tokenizer(text: str) -> list[str]:
    """jieba 中文分词，用于 BM25"""
    return list(jieba.cut(text))


def load_docs(path: str = DATA_PATH) -> list[Document]:
    """加载知识库，每条文档保留原始 id"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"标题：{item['title']}\n{item['content']}"
        docs.append(Document(
            page_content=content,
            metadata={"doc_id": item["id"], "title": item["title"]},
        ))
    return docs


def build_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )


def build_retrievers(docs: list[Document]):
    """构建三种检索器"""
    print("[1/3] 构建 BM25 检索器...")
    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k=TOP_K,
        preprocess_func=chinese_tokenizer,
    )

    print("[2/3] 构建向量检索器（Chroma + BGE）...")
    embeddings = build_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print("[3/3] 构建混合检索器（EnsembleRetriever，权重各 0.5）...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    return bm25_retriever, vector_retriever, hybrid_retriever


def reciprocal_rank(retrieved_docs: list[Document], relevant_doc_id: str) -> float:
    """Reciprocal Rank：正确文档排在第几位？
    rank=1 → 1.0, rank=2 → 0.5, rank=3 → 0.33, miss → 0.0
    """
    for i, doc in enumerate(retrieved_docs):
        if doc.metadata.get("doc_id") == relevant_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(retrieved_docs: list[Document], relevant_doc_id: str, k: int = 1) -> bool:
    """Hit@K：正确文档是否在前 K 条中"""
    return any(d.metadata.get("doc_id") == relevant_doc_id for d in retrieved_docs[:k])


def run_experiment(bm25, vector, hybrid):
    """逐条查询，收集三种检索的 RR 和 Hit@1"""
    results = []
    print("\n" + "=" * 70)
    print("  逐条查询结果  (RR = Reciprocal Rank，越高越好；Hit@1 = 正确文档是否排第一)")
    print("=" * 70)

    for item in TESTSET:
        q = item["query"]
        rel = item["relevant_doc_id"]
        qtype = item["type"]

        bm25_docs   = bm25.invoke(q)
        vector_docs = vector.invoke(q)
        hybrid_docs = hybrid.invoke(q)

        def rank_label(docs, rel):
            for i, d in enumerate(docs):
                if d.metadata.get("doc_id") == rel:
                    return i + 1
            return "miss"

        bm25_rr   = reciprocal_rank(bm25_docs, rel)
        vector_rr = reciprocal_rank(vector_docs, rel)
        hybrid_rr = reciprocal_rank(hybrid_docs, rel)

        bm25_h1   = "✓" if hit_at_k(bm25_docs, rel, 1) else "✗"
        vector_h1 = "✓" if hit_at_k(vector_docs, rel, 1) else "✗"
        hybrid_h1 = "✓" if hit_at_k(hybrid_docs, rel, 1) else "✗"

        results.append({
            "query": q,
            "type": qtype,
            "relevant_doc_id": rel,
            "bm25_rr": bm25_rr,
            "vector_rr": vector_rr,
            "hybrid_rr": hybrid_rr,
        })

        print(f"\n  [{qtype.upper():8}] {q[:50]}")
        print(f"    期望文档: {rel}")
        print(f"    BM25   [H@1={bm25_h1}] RR={bm25_rr:.2f} | "
              f"rank={rank_label(bm25_docs, rel)} | "
              f"召回: {[d.metadata['doc_id'] for d in bm25_docs]}")
        print(f"    Vector [H@1={vector_h1}] RR={vector_rr:.2f} | "
              f"rank={rank_label(vector_docs, rel)} | "
              f"召回: {[d.metadata['doc_id'] for d in vector_docs]}")
        print(f"    Hybrid [H@1={hybrid_h1}] RR={hybrid_rr:.2f} | "
              f"rank={rank_label(hybrid_docs, rel)} | "
              f"召回: {[d.metadata['doc_id'] for d in hybrid_docs]}")

    return results


def print_summary(results: list[dict]):
    """打印 MRR 汇总对比表"""
    keyword_cases = [r for r in results if r["type"] == "keyword"]
    semantic_cases = [r for r in results if r["type"] == "semantic"]

    def avg(cases, key):
        return sum(c[key] for c in cases) / len(cases) if cases else 0

    print("\n" + "=" * 70)
    print("  MRR（Mean Reciprocal Rank）汇总对比")
    print("  MRR=1.0 → 每次都排第一；MRR=0.5 → 平均排第二；MRR=0.0 → 全未命中")
    print("=" * 70)
    print(f"\n  {'查询类型':<12} {'BM25':>10} {'Vector':>10} {'Hybrid':>10}  最佳")
    print("  " + "─" * 56)

    for label, cases in [("关键词查询", keyword_cases), ("语义查询", semantic_cases), ("总体", results)]:
        b = avg(cases, "bm25_rr")
        v = avg(cases, "vector_rr")
        h = avg(cases, "hybrid_rr")
        best_val = max(b, v, h)
        best_name = ["BM25", "Vector", "Hybrid"][[b, v, h].index(best_val)]
        print(f"  {label:<12} {b:>10.3f} {v:>10.3f} {h:>10.3f}  {best_name}")

    print("=" * 70)

    kw_bm25   = avg(keyword_cases,  "bm25_rr")
    kw_vector = avg(keyword_cases,  "vector_rr")
    sem_bm25  = avg(semantic_cases, "bm25_rr")
    sem_vector = avg(semantic_cases, "vector_rr")
    total_hybrid = avg(results, "hybrid_rr")
    total_bm25   = avg(results, "bm25_rr")
    total_vector = avg(results, "vector_rr")

    print("\n  结论：")
    if kw_bm25 >= kw_vector:
        print("  ✓ 关键词查询：BM25 MRR 更高（精确词匹配优势）")
    if sem_vector >= sem_bm25:
        print("  ✓ 语义查询：Vector MRR 更高（语义理解优势）")
    if total_hybrid >= max(total_bm25, total_vector):
        print("  ✓ 混合检索：总体 MRR 最高，兼顾两类查询")


def save_report(results: list[dict]):
    with open("./hybrid_search_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n  报告已保存：./hybrid_search_report.json")


def main():
    print("=" * 70)
    print("  混合检索实验：BM25 vs Vector vs Hybrid (RRF)")
    print("=" * 70)

    print("\n加载知识库...")
    docs = load_docs()
    print(f"共 {len(docs)} 篇文档")

    bm25, vector, hybrid = build_retrievers(docs)

    results = run_experiment(bm25, vector, hybrid)
    print_summary(results)
    save_report(results)

    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
