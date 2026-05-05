"""
RAG 系列第七篇：检索策略对比实验
演示 4 种检索策略的差异：
  1. 相似度检索（默认）
  2. MMR（最大边际相关性，去重）
  3. 相似度阈值过滤
  4. Self-Query（查询解析 + 元数据过滤）

用法：
    python retrieval_strategies.py
"""

import json
import os
import shutil
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ─── 配置 ──────────────────────────────────────────────────────────────
DATA_PATH = "./data/sample_articles.json"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_documents(path: str) -> list[Document]:
    """加载 JSON 文章数据为 Document 列表"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"标题：{item['title']}\n{item['content']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "title": item["title"],
                "year": item["year"],
                "category": item["category"],
                "tags": ",".join(item["tags"]),
            }
        ))
    return docs


def build_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )


def print_results(results, label=""):
    """打印检索结果"""
    print(f"\n📋 {label} 召回 {len(results)} 条：")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        preview = doc.page_content.replace("\n", " ")[:80]
        print(f"   [{i}] [{meta.get('year','')}] [{meta.get('category','')}] {preview}...")


# ═══════════════════════════════════════════════════════════════════════
# Self-Query：查询解析器
# ═══════════════════════════════════════════════════════════════════════
def parse_query(query: str) -> dict:
    """
    解析自然语言查询，提取语义查询和元数据过滤条件。
    生产环境中可用 LLM（如 SelfQueryRetriever）替代此规则解析。
    """
    filters = {}
    semantic = query

    # 提取年份：2024年、2023年
    year_match = re.search(r'(20\d{2})\s*年', query)
    if year_match:
        filters["year"] = int(year_match.group(1))
        semantic = semantic.replace(year_match.group(0), "")

    # 提取类别
    categories = ["后端开发", "前端开发", "系统编程", "人工智能", "云原生", "数据库"]
    for cat in categories:
        if cat in query:
            filters["category"] = cat
            semantic = semantic.replace(cat, "")

    # 提取标签（含大小写）
    tags = ["Python", "JavaScript", "Go", "Rust", "React", "Vue", "PyTorch"]
    for tag in tags:
        if tag in query or tag.lower() in query.lower():
            filters["tags"] = tag
            semantic = semantic.replace(tag, "").replace(tag.lower(), "")

    # 清理语义查询中的常见冗余词和标点
    semantic = re.sub(r'(关于|的|篇章|文章|相关|类别)+', ' ', semantic).strip()
    semantic = re.sub(r'\s+', ' ', semantic).strip()
    if len(semantic) < 2:
        # 如果语义查询太短，用过滤条件组合一个合理的查询
        parts = []
        if "category" in filters:
            parts.append(filters["category"])
        if "tags" in filters:
            parts.append(filters["tags"])
        semantic = " ".join(parts) if parts else "技术"

    return {"semantic_query": semantic, "filters": filters}


def apply_filter(docs: list[Document], filters: dict) -> list[Document]:
    """根据过滤条件筛选文档"""
    result = docs
    if "year" in filters:
        result = [d for d in result if d.metadata.get("year") == filters["year"]]
    if "category" in filters:
        result = [d for d in result if d.metadata.get("category") == filters["category"]]
    if "tags" in filters:
        tag = filters["tags"]
        result = [d for d in result if tag in d.metadata.get("tags", "")]
    return result


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RAG 系列第七篇：检索策略对比实验")
    print("=" * 60)

    # 加载文档
    docs = load_documents(DATA_PATH)
    print(f"\n📄 加载 {len(docs)} 篇文章")
    for i, d in enumerate(docs, 1):
        print(f"   [{i}] {d.metadata['title']} ({d.metadata['year']}, {d.metadata['category']})")

    # 清理旧的向量库
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"\n🗑️  清理旧的向量库: {CHROMA_PATH}")

    # 构建 Embedding 和向量库
    embeddings = build_embeddings()
    print("\n🔧 构建向量库...")
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("   ✅ 向量库构建完成")
    except Exception as e:
        print(f"   ❌ 向量库构建失败: {e}")
        return

    # ── 策略 1：相似度检索 ────────────────────────────────────────────
    print_separator("策略 1：相似度检索（Similarity Search）")
    print("   原理：按向量相似度排序，返回最相似的 K 个")
    print("   特点：追求最高相关性，但结果可能集中在同一主题")

    query = "Python 异步编程"
    print(f"\n🔍 查询：「{query}」")

    results = vectorstore.similarity_search(query, k=4)
    print_results(results, "相似度检索")

    # ── 策略 2：MMR ──────────────────────────────────────────────────
    print_separator("策略 2：MMR（Maximum Marginal Relevance）")
    print("   原理：在相关性和多样性之间做权衡")
    print("   公式：MMR = λ×Sim(q,di) - (1-λ)×max(Sim(di,dj))")
    print("   lambda_mult=0.5：相关性多样性各一半")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "lambda_mult": 0.5, "fetch_k": 20},
    )
    results = retriever.invoke(query)
    print_results(results, "MMR 检索")

    print("\n📊 MMR vs 相似度检索对比：")
    sim_cats = set()
    mmr_cats = set()
    for r in vectorstore.similarity_search(query, k=4):
        sim_cats.add(r.metadata["category"])
    for r in results:
        mmr_cats.add(r.metadata["category"])
    print(f"   • 相似度检索覆盖 {len(sim_cats)} 个类别: {sim_cats}")
    print(f"   • MMR 检索覆盖 {len(mmr_cats)} 个类别: {mmr_cats}")
    print(f"   • MMR 类别更多样，避免内容重复")

    # ── 策略 3：相似度阈值过滤 ────────────────────────────────────────
    print_separator("策略 3：相似度阈值过滤（Similarity Score Threshold）")
    print("   原理：只返回相似度分数超过阈值的结果")
    print("   注意：Chroma 返回的是距离（越小越相似），不是相似度分数")

    # 先查看分数分布
    print("\n📊 先查看距离分布（距离越小越相似）：")
    results_with_score = vectorstore.similarity_search_with_score(query, k=10)
    for doc, score in results_with_score:
        print(f"   距离={score:.4f} | {doc.metadata['title'][:40]}...")

    # 手动实现阈值过滤：距离 <= 阈值的才保留
    threshold = 0.89
    print(f"\n📐 手动阈值过滤（距离 <= {threshold}）：")
    filtered_by_threshold = [(doc, score) for doc, score in results_with_score if score <= threshold]
    print(f"   超过阈值的有 {len(filtered_by_threshold)} 条")
    for doc, score in filtered_by_threshold[:4]:
        print(f"   距离={score:.4f} | {doc.metadata['title'][:40]}...")

    # ── 策略 4：Self-Query ────────────────────────────────────────────
    print_separator("策略 4：Self-Query（查询解析 + 元数据过滤）")
    print("   原理：把自然语言查询解析成结构化过滤条件")
    print("   流程：自然语言查询 → 解析 → 元数据过滤 → 向量检索")

    queries = [
        "2024 年关于 Python 的文章",
        "后端开发类别的文章",
        "2023 年前端相关的文章",
    ]

    for q in queries:
        print(f"\n🔍 查询：「{q}」")

        # 第 1 步：解析查询
        parsed = parse_query(q)
        semantic_query = parsed["semantic_query"]
        filters = parsed["filters"]

        print(f"   🤖 解析结果：")
        print(f"      语义查询：{semantic_query}")
        print(f"      过滤条件：{filters}")

        # 第 2 步：元数据过滤
        filtered_docs = apply_filter(docs, filters)
        print(f"   📁 元数据过滤后剩余 {len(filtered_docs)} 篇")
        for d in filtered_docs:
            print(f"      - {d.metadata['title']}")

        # 第 3 步：向量检索
        if filtered_docs:
            try:
                temp_store = Chroma.from_documents(
                    documents=filtered_docs,
                    embedding=embeddings,
                )
                results = temp_store.similarity_search(semantic_query, k=3)
                print_results(results, "Self-Query 最终")
            except Exception as e:
                print(f"   ⚠️ 向量检索失败（API 临时错误）: {str(e)[:60]}")
                print("   📝 元数据过滤结果已展示上述列表，可作为检索结果")
        else:
            print("   ⚠️ 过滤后无文档")

    # ── 总结 ──────────────────────────────────────────────────────────
    print_separator("四种策略对比总结")
    print("""
📌 策略选择建议：

   ┌─────────────────┬──────────────────────────────┬─────────────────┐
   │     策略        │          适用场景            │      注意点     │
   ├─────────────────┼──────────────────────────────┼─────────────────┤
   │ 相似度检索      │ 通用场景，追求最高相关性     │ 结果可能重复    │
   │ MMR             │ 结果多样性要求高             │ 参数需调优      │
   │ 阈值过滤        │ 质量要求高，宁可少不可错     │ 阈值需实验确定  │
   │ Self-Query      │ 查询含时间/类别等明确条件    │ 需要解析器支持  │
   └─────────────────┴──────────────────────────────┴─────────────────┘

💡 组合使用效果更佳：
   • Self-Query 先做元数据过滤缩小范围
   • MMR 在结果中保证多样性
   • 阈值过滤剔除低质量匹配

⚙️  MMR 参数调优指南：
   • lambda_mult=1.0 → 只考虑相关性
   • lambda_mult=0.0 → 只考虑多样性
   • lambda_mult=0.5 → 平衡（推荐）
   • fetch_k 越大 → 候选池越大 → 多样性越好
""")


if __name__ == "__main__":
    main()
