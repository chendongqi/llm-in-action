"""
Rerank 实验：向量检索 vs 向量检索 + Rerank

核心结论：
- 向量检索召回 top-N，但排序质量参差不齐
- Cross-Encoder Reranker 重新计算 query-document 相关性，提升排序精度
- context_precision 是衡量排序质量最直接的指标

运行方式：
    conda activate dev_base
    python rerank.py
"""

import json
import os
import time
import warnings
import requests
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever

from typing import Optional, Sequence
from ragas import evaluate
from ragas.metrics import (  # noqa: deprecated path, but still works in 0.4.x
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
CHROMA_DIR = "./chroma_rerank"

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

BASELINE_TOP_K = 4   # 向量检索直接返回 top 4
RECALL_TOP_K = 10    # 召回阶段放大到 top 10，再由 reranker 精排到 4
RERANK_TOP_N = 4     # reranker 最终保留 4 篇


# ── SiliconFlow Reranker ──────────────────────────────────────────────────────

class SiliconFlowReranker(BaseDocumentCompressor):
    """
    调用 SiliconFlow /v1/rerank 接口的 Cross-Encoder Reranker。

    工作方式：
    - 输入：query + 一批候选文档（bi-encoder 已初筛）
    - 模型：BAAI/bge-reranker-v2-m3（cross-encoder，同时看 query 和 document）
    - 输出：按相关性重新排序后的 top_n 篇文档
    """

    model: str = RERANK_MODEL
    api_key: str = ""
    api_base: str = "https://api.siliconflow.cn/v1"
    top_n: int = RERANK_TOP_N

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        doc_texts = [d.page_content for d in documents]

        url = f"{self.api_base.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": self.top_n,
            "return_documents": True,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            doc = documents[idx]
            doc.metadata["rerank_score"] = item["relevance_score"]
            reranked.append(doc)

        return reranked


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_docs(path: str = DATA_PATH) -> list[Document]:
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


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(retriever, llm, prompt, question: str) -> tuple[str, list[Document]]:
    """运行单次检索 + 生成，返回 (answer, retrieved_docs)"""
    docs = retriever.invoke(question)
    context = format_context(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


def build_ragas_dataset(testset: list[dict], retriever, llm, prompt) -> Dataset:
    """构建 RAGAS 评估数据集"""
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
        time.sleep(0.5)   # 避免 API 限速

    return Dataset.from_dict(rows)


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_ragas_eval(dataset: Dataset, llm, embeddings) -> dict:
    """跑 RAGAS 评估，返回指标均值字典"""
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
    print("  Rerank 实验：向量检索 vs 向量检索 + Cross-Encoder Rerank")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载知识库和测试集...")
    docs = load_docs()
    testset = load_testset()
    print(f"  知识库：{len(docs)} 篇文档，测试集：{len(testset)} 条问题")

    # 构建基础组件
    print("\n[2/5] 构建向量库（Chroma + BGE）...")
    embeddings = build_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    llm = build_llm()
    prompt = build_prompt()

    # ── 策略 A：基准检索（直接 top-4）
    print("\n[3/5] 运行策略 A：基准向量检索（top-4 直接返回）...")
    baseline_retriever = vectorstore.as_retriever(
        search_kwargs={"k": BASELINE_TOP_K}
    )
    baseline_dataset = build_ragas_dataset(testset, baseline_retriever, llm, prompt)

    # ── 策略 B：Rerank（召回 top-10，精排到 top-4）
    print("\n[4/5] 运行策略 B：向量检索 + Reranker（top-10 → rerank → top-4）...")
    recall_retriever = vectorstore.as_retriever(
        search_kwargs={"k": RECALL_TOP_K}
    )
    reranker = SiliconFlowReranker(
        model=RERANK_MODEL,
        api_key=EMBEDDING_API_KEY,
        api_base=RERANK_API_BASE,
        top_n=RERANK_TOP_N,
    )
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=recall_retriever,
    )
    rerank_dataset = build_ragas_dataset(testset, rerank_retriever, llm, prompt)

    # ── RAGAS 评估
    print("\n[5/5] 计算 RAGAS 指标...")
    baseline_scores = run_ragas_eval(baseline_dataset, llm, embeddings)
    rerank_scores = run_ragas_eval(rerank_dataset, llm, embeddings)

    # ── 输出对比报告
    print("\n" + "=" * 70)
    print("  RAGAS 指标对比（向量检索 vs 向量检索 + Rerank）")
    print("=" * 70)
    print(f"\n  {'指标':<20} {'基准 (top-4)':>14} {'Rerank (10→4)':>14}  {'变化':>8}")
    print("  " + "─" * 58)

    metric_names = {
        "context_precision":  "context_precision",
        "context_recall":     "context_recall",
        "faithfulness":       "faithfulness",
        "answer_relevancy":   "answer_relevancy",
    }
    highlight = "context_precision"   # 本篇核心指标

    report_rows = []
    for key, label in metric_names.items():
        b = baseline_scores[key]
        r = rerank_scores[key]
        delta = r - b
        flag = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        marker = " ◀ 核心指标" if key == highlight else ""
        print(f"  {label:<20} {b:>14.3f} {r:>14.3f}  {flag}{delta:+.3f}{marker}")
        report_rows.append({
            "metric": key,
            "baseline": round(b, 4),
            "rerank": round(r, 4),
            "delta": round(delta, 4),
        })

    print("=" * 70)

    cp_b = baseline_scores["context_precision"]
    cp_r = rerank_scores["context_precision"]
    print("\n  结论：")
    if cp_r > cp_b:
        print(f"  ✓ Rerank 将 context_precision 从 {cp_b:.3f} 提升到 {cp_r:.3f}")
        print("    → 更相关的文档排在前面，LLM 能看到更高质量的上下文")
    else:
        print(f"  ~ context_precision 变化不显著 ({cp_b:.3f} → {cp_r:.3f})")
        print("    → 在当前数据集上，向量检索的排序已足够准确")

    # ── 保存报告
    report = {
        "config": {
            "baseline_top_k": BASELINE_TOP_K,
            "recall_top_k": RECALL_TOP_K,
            "rerank_top_n": RERANK_TOP_N,
            "rerank_model": RERANK_MODEL,
        },
        "scores": report_rows,
        "baseline_dataset": baseline_dataset.to_dict(),
        "rerank_dataset": rerank_dataset.to_dict(),
    }
    with open("./rerank_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n  报告已保存：./rerank_report.json")
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
