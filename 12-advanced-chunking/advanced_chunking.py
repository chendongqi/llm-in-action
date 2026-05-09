"""
高级分块策略实验：Naive Chunking vs Parent-Child vs Contextual Retrieval

核心结论：
- Naive Chunking：小块精准匹配，但返回片段缺乏上下文
- Parent-Child：小块检索 + 大块返回，兼顾精度和完整性
- Contextual Retrieval：LLM 给每个 Chunk 加上文档上下文，提升 Embedding 语义质量

评估指标：context_recall（上下文覆盖率）和 context_precision（排序质量）

运行方式：
    conda activate dev_base
    python advanced_chunking.py
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from ragas import evaluate
from ragas.metrics import (
    faithfulness as ragas_faithfulness,
    answer_relevancy as ragas_answer_relevancy,
    context_precision as ragas_context_precision,
    context_recall as ragas_context_recall,
)
from ragas.llms import _LangchainLLMWrapper as LangchainLLMWrapper
from ragas.embeddings import _LangchainEmbeddingsWrapper as LangchainEmbeddingsWrapper
from datasets import Dataset

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH = "../08-ragas-eval/data/knowledge_base.json"
TESTSET_PATH = "../08-ragas-eval/data/manual_testset.json"

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 分块参数
NAIVE_CHUNK_SIZE = 512
NAIVE_CHUNK_OVERLAP = 50
CHILD_CHUNK_SIZE = 200    # Parent-Child：小块用于检索
PARENT_CHUNK_SIZE = 800   # Parent-Child：大块用于返回给 LLM
TOP_K = 4


# ── 上下文描述 Prompt（Contextual Retrieval 用）────────────────────────────────
CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个文档分析助手。"),
    ("human",
     "以下是一篇完整文档：\n\n<document>\n{doc_content}\n</document>\n\n"
     "以下是文档中的一个片段：\n\n<chunk>\n{chunk_content}\n</chunk>\n\n"
     "请用 1-2 句话概括这个片段在整篇文档中的作用和背景，"
     "帮助理解该片段的含义。只输出描述文字，不要加任何前缀。"),
])


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_raw_data(path: str = DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_testset(path: str = TESTSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )


def build_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )


def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的技术问答助手。请严格根据提供的参考资料回答问题。"
                   "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
        ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
    ])


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── 策略 A：Naive Chunking ─────────────────────────────────────────────────────

def build_naive_retriever(raw_data: list[dict], embeddings) -> object:
    """标准分块：RecursiveCharacterTextSplitter，直接 top-k 检索"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NAIVE_CHUNK_SIZE,
        chunk_overlap=NAIVE_CHUNK_OVERLAP,
    )
    docs = []
    for item in raw_data:
        content = f"标题：{item['title']}\n{item['content']}"
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"doc_id": item["id"], "title": item["title"], "chunk_idx": i},
            ))

    vectorstore = Chroma(
        collection_name="naive",
        embedding_function=embeddings,
    )
    vectorstore.add_documents(docs)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ── 策略 B：Parent-Child Chunking ─────────────────────────────────────────────

def build_parent_child_retriever(raw_data: list[dict], embeddings) -> object:
    """
    Parent-Child：
    - child_splitter：小块（200字），用于向量检索（精准匹配）
    - parent_splitter：大块（800字），检索命中后返回对应的大块给 LLM
    """
    parent_docs = []
    for item in raw_data:
        content = f"标题：{item['title']}\n{item['content']}"
        parent_docs.append(Document(
            page_content=content,
            metadata={"doc_id": item["id"], "title": item["title"]},
        ))

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=20,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=50,
    )

    vectorstore = Chroma(
        collection_name="parent_child",
        embedding_function=embeddings,
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": TOP_K},
    )
    retriever.add_documents(parent_docs)
    return retriever


# ── 策略 C：Contextual Retrieval ──────────────────────────────────────────────

def build_contextual_retriever(raw_data: list[dict], embeddings, llm) -> object:
    """
    Contextual Retrieval（Anthropic 方案）：
    - 用 LLM 为每个 Chunk 生成一段"上下文描述"
    - 将描述前缀拼接到 Chunk 内容，再做 Embedding
    - 检索时每个 Chunk 都携带了完整的文档背景信息
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NAIVE_CHUNK_SIZE,
        chunk_overlap=NAIVE_CHUNK_OVERLAP,
    )
    context_chain = CONTEXT_PROMPT | llm | StrOutputParser()

    print("    生成 Contextual 描述（每个 chunk 调用一次 LLM）...")
    docs = []
    for item in raw_data:
        full_content = f"标题：{item['title']}\n{item['content']}"
        chunks = splitter.split_text(full_content)
        for i, chunk in enumerate(chunks):
            # 用 LLM 生成该 chunk 在文档中的上下文描述
            context_desc = context_chain.invoke({
                "doc_content": full_content,
                "chunk_content": chunk,
            })
            # 将上下文描述前缀拼接到 chunk
            enriched_content = f"{context_desc}\n\n{chunk}"
            docs.append(Document(
                page_content=enriched_content,
                metadata={
                    "doc_id": item["id"],
                    "title": item["title"],
                    "chunk_idx": i,
                    "context_desc": context_desc,
                },
            ))
            time.sleep(0.2)  # 避免 LLM API 限速

    vectorstore = Chroma(
        collection_name="contextual",
        embedding_function=embeddings,
    )
    vectorstore.add_documents(docs)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ── Pipeline & Evaluation ──────────────────────────────────────────────────────

def run_pipeline(retriever, llm, prompt, question: str) -> tuple[str, list[Document]]:
    docs = retriever.invoke(question)
    context = format_context(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


def build_ragas_dataset(
    testset: list[dict], retriever, llm, prompt, label: str
) -> Dataset:
    print(f"    [{label}] 运行 {len(testset)} 条问题...")
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for item in testset:
        q = item["question"]
        gt = item["ground_truth"]
        answer, docs = run_pipeline(retriever, llm, prompt, q)
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([d.page_content for d in docs])
        rows["ground_truth"].append(gt)
        time.sleep(0.5)
    return Dataset.from_dict(rows)


def run_ragas_eval(dataset: Dataset, llm, embeddings) -> dict:
    result = evaluate(
        dataset=dataset,
        metrics=[
            ragas_faithfulness,
            ragas_answer_relevancy,
            ragas_context_precision,
            ragas_context_recall,
        ],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    df = result.to_pandas()
    return {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  高级分块策略实验：Naive vs Parent-Child vs Contextual Retrieval")
    print("=" * 70)

    print("\n[1/6] 加载知识库和测试集...")
    raw_data = load_raw_data()
    testset = load_testset()
    print(f"  知识库：{len(raw_data)} 篇文档，测试集：{len(testset)} 条问题")

    print("\n[2/6] 初始化 Embedding 模型和 LLM...")
    embeddings = build_embeddings()
    llm = build_llm()
    prompt = build_prompt()

    print("\n[3/6] 构建三种检索器...")

    print("  构建 Naive 检索器（chunk_size=512）...")
    naive_retriever = build_naive_retriever(raw_data, embeddings)

    print("  构建 Parent-Child 检索器（child=200, parent=800）...")
    pc_retriever = build_parent_child_retriever(raw_data, embeddings)

    print("  构建 Contextual Retrieval 检索器...")
    ctx_retriever = build_contextual_retriever(raw_data, embeddings, llm)

    print("\n[4/6] 运行三种策略的问答...")
    naive_dataset = build_ragas_dataset(testset, naive_retriever, llm, prompt, "Naive")
    pc_dataset = build_ragas_dataset(testset, pc_retriever, llm, prompt, "Parent-Child")
    ctx_dataset = build_ragas_dataset(testset, ctx_retriever, llm, prompt, "Contextual")

    print("\n[5/6] 计算 RAGAS 指标...")
    naive_scores = run_ragas_eval(naive_dataset, llm, embeddings)
    pc_scores = run_ragas_eval(pc_dataset, llm, embeddings)
    ctx_scores = run_ragas_eval(ctx_dataset, llm, embeddings)

    # ── 输出对比报告
    print("\n" + "=" * 78)
    print("  RAGAS 指标对比（三种分块策略）")
    print("=" * 78)
    print(f"\n  {'指标':<22} {'Naive':>10} {'Parent-Child':>14} {'Contextual':>12}")
    print("  " + "─" * 60)

    metrics = [
        ("context_recall",    "context_recall   "),
        ("context_precision",  "context_precision"),
        ("faithfulness",       "faithfulness     "),
        ("answer_relevancy",   "answer_relevancy "),
    ]

    report_rows = []
    for key, label in metrics:
        n = naive_scores[key]
        p = pc_scores[key]
        c = ctx_scores[key]
        best = max(n, p, c)
        def fmt(v): return f"{v:.3f}" + (" ◀" if abs(v - best) < 0.001 else "   ")
        print(f"  {label:<22} {fmt(n):>12} {fmt(p):>16} {fmt(c):>14}")
        report_rows.append({
            "metric": key,
            "naive": round(n, 4),
            "parent_child": round(p, 4),
            "contextual": round(c, 4),
        })

    print("=" * 78)

    # 结论
    cr_n = naive_scores["context_recall"]
    cr_p = pc_scores["context_recall"]
    cr_c = ctx_scores["context_recall"]
    cp_n = naive_scores["context_precision"]
    cp_p = pc_scores["context_precision"]
    cp_c = ctx_scores["context_precision"]

    print("\n  结论：")
    if cr_p > cr_n:
        print(f"  ✓ Parent-Child 将 context_recall 从 {cr_n:.3f} 提升到 {cr_p:.3f}")
        print("    → 大块返回保留了更多上下文，LLM 能看到完整信息")
    if cr_c >= cr_n:
        print(f"  ✓ Contextual Retrieval context_recall: {cr_c:.3f}")
        print("    → 上下文描述增强了 Embedding 的语义质量，召回更准")
    if cp_c > cp_n or cp_p > cp_n:
        best_cp = max(cp_p, cp_c)
        best_name = "Parent-Child" if cp_p >= cp_c else "Contextual"
        print(f"  ✓ {best_name} 将 context_precision 从 {cp_n:.3f} 提升到 {best_cp:.3f}")

    # 保存报告
    print("\n[6/6] 保存报告...")
    report = {
        "config": {
            "naive_chunk_size": NAIVE_CHUNK_SIZE,
            "child_chunk_size": CHILD_CHUNK_SIZE,
            "parent_chunk_size": PARENT_CHUNK_SIZE,
            "top_k": TOP_K,
        },
        "scores": report_rows,
        "naive_dataset": naive_dataset.to_dict(),
        "parent_child_dataset": pc_dataset.to_dict(),
        "contextual_dataset": ctx_dataset.to_dict(),
    }
    with open("./advanced_chunking_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  报告已保存：./advanced_chunking_report.json")
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
