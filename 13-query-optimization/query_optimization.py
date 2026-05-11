"""
查询优化实验：Naive vs Multi-Query vs HyDE vs Query Decomposition

核心结论：
- Naive：原始问题直接检索，单一视角，容易漏掉相关文档
- Multi-Query：LLM 生成多个问法，多角度检索后合并去重，提升召回率
- HyDE：先让 LLM 生成假设答案，用假设答案的 embedding 检索，更接近"答案空间"
- Query Decomposition：复杂问题拆成多个子问题分别检索，再合并给 LLM

评估：用 context_recall（召回率）衡量查询优化的核心价值

运行方式：
    conda activate dev_base
    python query_optimization.py
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever

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

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4


# ── Prompt 定义 ────────────────────────────────────────────────────────────────

# Multi-Query：生成中文多角度问法
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的问题改写助手。"),
    ("human",
     "请将以下问题改写为 3 个不同的表达方式，从不同角度提问，"
     "以便在向量数据库中检索到更多相关内容。\n"
     "每行输出一个问题，不要编号，不要解释。\n\n"
     "原始问题：{question}"),
])

# HyDE：生成假设答案
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个技术知识助手。"),
    ("human",
     "请为以下问题写一段假设性的回答，约 100 字。"
     "这段回答将用于向量检索，不需要完全准确，"
     "只需要在语义上与真实答案接近。\n\n"
     "问题：{question}"),
])

# Query Decomposition：将复杂问题拆分为子问题
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个问题分析助手。"),
    ("human",
     "请将以下复杂问题拆分为 2-3 个简单的子问题，"
     "每个子问题可以独立检索。\n"
     "每行输出一个子问题，不要编号，不要解释。\n\n"
     "原始问题：{question}"),
])

# RAG 生成 Prompt
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请严格根据提供的参考资料回答问题。"
               "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
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
        temperature=0.3,  # 生成多样化问法需要适当的随机性
    )


def build_vectorstore(raw_data: list[dict], embeddings, collection_name: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(docs)
    return vectorstore


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def dedup_docs(docs: list[Document]) -> list[Document]:
    """按内容去重"""
    seen = set()
    result = []
    for doc in docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


# ── 策略 A：Naive ──────────────────────────────────────────────────────────────

def retrieve_naive(vectorstore, question: str) -> list[Document]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(question)


# ── 策略 B：Multi-Query ────────────────────────────────────────────────────────

def retrieve_multi_query(vectorstore, llm, question: str) -> list[Document]:
    """
    步骤：
    1. LLM 生成 3 个不同角度的问法
    2. 对原始问题 + 3 个改写问题分别检索
    3. 合并结果并去重，取前 TOP_K 篇
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    multi_query_chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    variants_text = multi_query_chain.invoke({"question": question})
    variants = [q.strip() for q in variants_text.strip().split("\n") if q.strip()]

    all_docs = base_retriever.invoke(question)  # 原始问题
    for variant in variants:
        all_docs.extend(base_retriever.invoke(variant))

    return dedup_docs(all_docs)[:TOP_K]


# ── 策略 C：HyDE ──────────────────────────────────────────────────────────────

def retrieve_hyde(vectorstore, llm, embeddings, question: str) -> list[Document]:
    """
    步骤：
    1. LLM 生成一段假设性答案（约 100 字）
    2. 用假设答案的 embedding 去检索（而不是用问题的 embedding）
    3. 假设答案的向量空间更接近真实答案，检索更准
    """
    hyde_chain = HYDE_PROMPT | llm | StrOutputParser()
    hypothetical_answer = hyde_chain.invoke({"question": question})

    # 用假设答案的 embedding 检索
    hyp_embedding = embeddings.embed_query(hypothetical_answer)
    results = vectorstore.similarity_search_by_vector(hyp_embedding, k=TOP_K)
    return results


# ── 策略 D：Query Decomposition ───────────────────────────────────────────────

def retrieve_decomposed(vectorstore, llm, question: str) -> list[Document]:
    """
    步骤：
    1. LLM 将复杂问题拆解为 2-3 个子问题
    2. 对每个子问题分别检索
    3. 合并去重，取前 TOP_K 篇（较多的子问题能覆盖更多角度）
    """
    decompose_chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    sub_questions_text = decompose_chain.invoke({"question": question})
    sub_questions = [q.strip() for q in sub_questions_text.strip().split("\n") if q.strip()]

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    all_docs = []
    for sub_q in sub_questions:
        all_docs.extend(base_retriever.invoke(sub_q))

    return dedup_docs(all_docs)[:TOP_K]


# ── Pipeline & Evaluation ──────────────────────────────────────────────────────

def run_pipeline(docs: list[Document], llm, question: str) -> str:
    context = format_context(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def build_ragas_dataset(
    testset: list[dict],
    vectorstore,
    llm,
    embeddings,
    strategy: str,
) -> Dataset:
    print(f"    [{strategy}] 运行 {len(testset)} 条问题...")
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for item in testset:
        q = item["question"]
        gt = item["ground_truth"]

        if strategy == "naive":
            docs = retrieve_naive(vectorstore, q)
        elif strategy == "multi_query":
            docs = retrieve_multi_query(vectorstore, llm, q)
        elif strategy == "hyde":
            docs = retrieve_hyde(vectorstore, llm, embeddings, q)
        elif strategy == "decomposed":
            docs = retrieve_decomposed(vectorstore, llm, q)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        answer = run_pipeline(docs, llm, q)
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
    print("=" * 78)
    print("  查询优化实验：Naive vs Multi-Query vs HyDE vs Query Decomposition")
    print("=" * 78)

    print("\n[1/6] 加载数据...")
    raw_data = load_raw_data()
    testset = load_testset()
    print(f"  知识库：{len(raw_data)} 篇，测试集：{len(testset)} 条")

    print("\n[2/6] 初始化模型和向量库...")
    embeddings = build_embeddings()
    llm = build_llm()
    vectorstore = build_vectorstore(raw_data, embeddings, "query_opt")
    print(f"  向量库构建完成（chunk_size={CHUNK_SIZE}）")

    print("\n[3/6] 运行四种查询策略...")
    strategies = ["naive", "multi_query", "hyde", "decomposed"]
    datasets = {}
    for s in strategies:
        datasets[s] = build_ragas_dataset(testset, vectorstore, llm, embeddings, s)

    print("\n[4/6] 计算 RAGAS 指标...")
    # 生成时用 temperature=0 保持稳定
    eval_llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )
    scores = {}
    for s in strategies:
        print(f"  评估 [{s}]...")
        scores[s] = run_ragas_eval(datasets[s], eval_llm, embeddings)

    # ── 输出对比报告
    labels = {
        "naive":      "Naive       ",
        "multi_query": "Multi-Query ",
        "hyde":       "HyDE        ",
        "decomposed": "Decomposed  ",
    }

    print("\n" + "=" * 82)
    print("  RAGAS 指标对比（四种查询优化策略）")
    print("=" * 82)
    print(f"\n  {'指标':<22} {'Naive':>10} {'Multi-Query':>13} {'HyDE':>8} {'Decomposed':>12}")
    print("  " + "─" * 66)

    metrics = [
        ("context_recall",   "context_recall   "),
        ("context_precision", "context_precision"),
        ("faithfulness",      "faithfulness     "),
        ("answer_relevancy",  "answer_relevancy "),
    ]

    report_rows = []
    for key, label in metrics:
        vals = {s: scores[s][key] for s in strategies}
        best = max(vals.values())
        def fmt(v): return f"{v:.3f}" + ("◀" if abs(v - best) < 0.001 else " ")
        line = f"  {label:<22}"
        for s in strategies:
            line += f"  {fmt(vals[s]):>9}"
        print(line)
        report_rows.append({
            "metric": key,
            **{s: round(vals[s], 4) for s in strategies},
        })

    print("=" * 82)

    # 结论
    print("\n  结论：")
    cr = {s: scores[s]["context_recall"] for s in strategies}
    best_s = max(cr, key=lambda x: cr[x])
    print(f"  context_recall 最优策略：{best_s}（{cr[best_s]:.3f}）")
    print(f"  vs Naive：{cr['naive']:.3f}  →  提升 {cr[best_s] - cr['naive']:+.3f}")

    # ── 保存报告
    print("\n[5/6] 保存报告...")
    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "top_k": TOP_K,
            "llm_model": LLM_MODEL,
        },
        "scores": report_rows,
        **{f"{s}_dataset": datasets[s].to_dict() for s in strategies},
    }
    with open("./query_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  报告已保存：./query_optimization_report.json")
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
