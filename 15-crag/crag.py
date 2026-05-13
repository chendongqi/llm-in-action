"""
CRAG（Corrective RAG）实验：固定检索 vs 自我纠偏检索

CRAG 的核心思路：
  检索结果不好时不要将就——自动触发网络搜索作为兜底。

流程：
  1. 向量检索（知识库）
  2. 相关性评分：每篇文档打分（CORRECT / AMBIGUOUS / INCORRECT）
  3. 决策：
     - 高分（CORRECT）  → 直接使用知识库文档生成
     - 低分（INCORRECT）→ 丢弃知识库结果，改用网络搜索
     - 中间（AMBIGUOUS）→ 知识库 + 网络搜索结果合并使用
  4. 基于最终文档生成答案

使用 LangGraph 实现完整流程，网络搜索 fallback 使用 DuckDuckGo。

运行方式：
    conda activate dev_base
    python crag.py
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

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

# 相关性评分阈值
CORRECT_THRESHOLD = 0.7    # 高于此值：知识库文档可信，直接使用
INCORRECT_THRESHOLD = 0.3  # 低于此值：知识库文档不可信，触发网络搜索


# ── Prompts ────────────────────────────────────────────────────────────────────

# 相关性评分：对单篇文档打分
RELEVANCE_SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个文档相关性评估器。\n"
     "给出一个 0.0 到 1.0 之间的分数，表示文档与问题的相关程度。\n"
     "- 1.0：文档直接、完整地回答了问题\n"
     "- 0.5：文档部分相关，但不完整\n"
     "- 0.0：文档与问题完全无关\n\n"
     "只输出一个浮点数，不要任何解释。"),
    ("human", "问题：{question}\n\n文档：{document}"),
])

# RAG 生成
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请严格根据提供的参考资料回答问题。"
               "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
])

# 知识精炼：从网络搜索结果中提取关键信息
REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个信息提取助手。从以下网络搜索结果中，"
     "提取与问题最相关的关键信息，整理成简洁的参考资料。"
     "去除无关内容，保留核心事实。"),
    ("human", "问题：{question}\n\n搜索结果：\n{search_results}\n\n请提取关键信息："),
])


# ── CRAG State ─────────────────────────────────────────────────────────────────

class CRAGState(TypedDict):
    question: str
    retrieved_docs: list[Document]
    doc_scores: list[float]          # 每篇文档的相关性分数
    overall_score: float             # 综合评分
    retrieval_verdict: str           # "correct" | "ambiguous" | "incorrect"
    web_results: str                 # 网络搜索原始结果
    refined_web_docs: list[Document] # 精炼后的网络搜索文档
    final_docs: list[Document]       # 最终送入 LLM 的文档
    answer: str
    path: list[str]


# ── Graph 节点函数 ─────────────────────────────────────────────────────────────

def make_retrieve_node(retriever):
    """节点1：向量检索"""
    def retrieve(state: CRAGState) -> CRAGState:
        docs = retriever.invoke(state["question"])
        return {
            **state,
            "retrieved_docs": docs,
            "path": state.get("path", []) + ["retrieve"],
        }
    return retrieve


def make_score_node(llm):
    """节点2：对每篇检索文档打相关性分数"""
    chain = RELEVANCE_SCORE_PROMPT | llm | StrOutputParser()

    def score_docs(state: CRAGState) -> CRAGState:
        scores = []
        for doc in state["retrieved_docs"]:
            raw = chain.invoke({
                "question": state["question"],
                "document": doc.page_content[:400],
            })
            try:
                score = float(raw.strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.5
            scores.append(score)

        overall = sum(scores) / len(scores) if scores else 0.0

        if overall >= CORRECT_THRESHOLD:
            verdict = "correct"
        elif overall <= INCORRECT_THRESHOLD:
            verdict = "incorrect"
        else:
            verdict = "ambiguous"

        return {
            **state,
            "doc_scores": scores,
            "overall_score": overall,
            "retrieval_verdict": verdict,
            "path": state.get("path", []) + [f"score({verdict},{overall:.2f})"],
        }

    return score_docs


def make_web_search_node(search_tool, llm):
    """节点3：网络搜索 + 知识精炼（网络不可用时回退到知识库文档）"""
    refine_chain = REFINE_PROMPT | llm | StrOutputParser()

    def web_search(state: CRAGState) -> CRAGState:
        try:
            raw_results = search_tool.invoke(state["question"])
            if not raw_results or len(raw_results.strip()) < 20:
                raise ValueError("empty results")

            # 用 LLM 精炼搜索结果，去噪提取关键信息
            refined = refine_chain.invoke({
                "question": state["question"],
                "search_results": raw_results[:2000],
            })
            web_doc = Document(
                page_content=refined,
                metadata={"source": "web_search", "query": state["question"]},
            )
            print(f"      [web_search] 搜索成功，已精炼结果")
            return {
                **state,
                "web_results": raw_results,
                "refined_web_docs": [web_doc],
                "path": state.get("path", []) + ["web_search(ok)"],
            }

        except Exception as e:
            # 网络不可用时：回退到知识库检索结果（即便分数低也保留）
            print(f"      [web_search] 网络不可用（{type(e).__name__}），回退到知识库文档")
            return {
                **state,
                "web_results": "",
                "refined_web_docs": [],   # assemble 节点会用知识库文档兜底
                "path": state.get("path", []) + ["web_search(fallback)"],
            }

    return web_search


def make_assemble_node():
    """节点4：根据 verdict 组装最终文档"""
    def assemble(state: CRAGState) -> CRAGState:
        verdict = state["retrieval_verdict"]

        if verdict == "correct":
            # 只用知识库文档（按分数过滤，保留高分的）
            scored = list(zip(state["retrieved_docs"], state["doc_scores"]))
            scored.sort(key=lambda x: x[1], reverse=True)
            final = [doc for doc, score in scored if score >= INCORRECT_THRESHOLD]
            if not final:
                final = [scored[0][0]]  # 至少保留最高分的一篇

        elif verdict == "incorrect":
            # 优先用网络搜索结果；网络不可用时回退到知识库文档
            final = state.get("refined_web_docs", [])
            if not final:
                scored = list(zip(state["retrieved_docs"], state["doc_scores"]))
                scored.sort(key=lambda x: x[1], reverse=True)
                final = [scored[0][0]] if scored else []

        else:  # ambiguous
            # 知识库高分文档 + 网络搜索结果合并
            scored = list(zip(state["retrieved_docs"], state["doc_scores"]))
            kb_docs = [doc for doc, score in scored if score >= INCORRECT_THRESHOLD]
            web_docs = state.get("refined_web_docs", [])
            final = kb_docs + web_docs
            # 如果 web_docs 为空（网络失败），至少保留知识库文档
            if not final:
                final = [doc for doc, _ in scored[:2]]

        return {
            **state,
            "final_docs": final,
            "path": state.get("path", []) + [f"assemble({len(final)}docs)"],
        }

    return assemble


def make_generate_node(llm):
    """节点5：生成最终答案"""
    chain = RAG_PROMPT | llm | StrOutputParser()

    def generate(state: CRAGState) -> CRAGState:
        context = "\n\n---\n\n".join(d.page_content for d in state["final_docs"])
        answer = chain.invoke({"context": context, "question": state["question"]})
        return {
            **state,
            "answer": answer,
            "path": state.get("path", []) + ["generate"],
        }

    return generate


# ── 路由：score 之后根据 verdict 决定是否需要网络搜索 ───────────────────────────

def route_after_score(state: CRAGState) -> Literal["web_search", "assemble"]:
    verdict = state["retrieval_verdict"]
    if verdict in ("incorrect", "ambiguous"):
        return "web_search"
    return "assemble"


# ── 构建 CRAG Graph ────────────────────────────────────────────────────────────

def build_crag_graph(llm, retriever, search_tool):
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve",   make_retrieve_node(retriever))
    graph.add_node("score",      make_score_node(llm))
    graph.add_node("web_search", make_web_search_node(search_tool, llm))
    graph.add_node("assemble",   make_assemble_node())
    graph.add_node("generate",   make_generate_node(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "score")
    graph.add_conditional_edges(
        "score",
        route_after_score,
        {"web_search": "web_search", "assemble": "assemble"},
    )
    graph.add_edge("web_search", "assemble")
    graph.add_edge("assemble",   "generate")
    graph.add_edge("generate",   END)

    return graph.compile()


# ── 基准：固定检索 Pipeline ────────────────────────────────────────────────────

def run_always_retrieve(retriever, llm, question: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "docs": docs}


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_raw_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
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


def build_vectorstore(raw_data, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for item in raw_data:
        content = f"标题：{item['title']}\n{item['content']}"
        for i, chunk in enumerate(splitter.split_text(content)):
            docs.append(Document(
                page_content=chunk,
                metadata={"doc_id": item["id"], "title": item["title"], "chunk_idx": i},
            ))
    vs = Chroma(collection_name="crag", embedding_function=embeddings)
    vs.add_documents(docs)
    return vs


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
    print("  CRAG 实验：固定检索 vs Corrective RAG（LangGraph）")
    print("=" * 70)

    print("\n[1/6] 加载数据...")
    raw_data = load_raw_data()
    testset = load_testset()
    print(f"  知识库：{len(raw_data)} 篇，测试集：{len(testset)} 条")

    print("\n[2/6] 初始化模型、向量库和搜索工具...")
    embeddings = build_embeddings()
    llm = build_llm()
    vectorstore = build_vectorstore(raw_data, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    search_tool = DuckDuckGoSearchRun()

    print("\n[3/6] 构建 CRAG Graph...")
    crag_graph = build_crag_graph(llm, retriever, search_tool)

    print("\n[4/6] 运行两种策略...")

    # 固定检索
    print("  运行固定检索（Always Retrieve）...")
    always_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in testset:
        result = run_always_retrieve(retriever, llm, item["question"])
        always_rows["question"].append(item["question"])
        always_rows["answer"].append(result["answer"])
        always_rows["contexts"].append([d.page_content for d in result["docs"]])
        always_rows["ground_truth"].append(item["ground_truth"])
        time.sleep(0.3)

    # CRAG
    print("  运行 CRAG（Corrective）...")
    crag_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    crag_paths = []
    verdict_counts = {"correct": 0, "ambiguous": 0, "incorrect": 0}

    for item in testset:
        init_state: CRAGState = {
            "question": item["question"],
            "retrieved_docs": [],
            "doc_scores": [],
            "overall_score": 0.0,
            "retrieval_verdict": "",
            "web_results": "",
            "refined_web_docs": [],
            "final_docs": [],
            "answer": "",
            "path": [],
        }
        result = crag_graph.invoke(init_state)
        crag_rows["question"].append(item["question"])
        crag_rows["answer"].append(result["answer"])
        crag_rows["contexts"].append([d.page_content for d in result["final_docs"]])
        crag_rows["ground_truth"].append(item["ground_truth"])
        crag_paths.append(result["path"])
        verdict_counts[result["retrieval_verdict"]] += 1
        time.sleep(0.5)

    # 打印执行路径
    print("\n  CRAG 执行路径明细：")
    for i, (item, path) in enumerate(zip(testset, crag_paths)):
        print(f"  Q{i+1}: {' → '.join(path)}")
        print(f"       {item['question'][:50]}")

    print(f"\n  相关性评分分布：correct={verdict_counts['correct']}，"
          f"ambiguous={verdict_counts['ambiguous']}，"
          f"incorrect={verdict_counts['incorrect']}")

    print("\n[5/6] 计算 RAGAS 指标...")
    always_dataset = Dataset.from_dict(always_rows)
    crag_dataset = Dataset.from_dict(crag_rows)

    always_scores = run_ragas_eval(always_dataset, llm, embeddings)
    crag_scores = run_ragas_eval(crag_dataset, llm, embeddings)

    # ── 输出报告
    print("\n" + "=" * 70)
    print("  RAGAS 指标对比（固定检索 vs CRAG）")
    print("=" * 70)
    print(f"\n  {'指标':<22} {'固定检索':>12} {'CRAG':>12}  {'变化':>8}")
    print("  " + "─" * 56)

    metrics = [
        ("context_recall",   "context_recall   "),
        ("context_precision", "context_precision"),
        ("faithfulness",      "faithfulness     "),
        ("answer_relevancy",  "answer_relevancy "),
    ]
    report_rows = []
    for key, label in metrics:
        a = always_scores[key]
        c = crag_scores[key]
        delta = c - a
        flag = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"  {label:<22} {a:>12.3f} {c:>12.3f}  {flag}{delta:+.3f}")
        report_rows.append({
            "metric": key,
            "always_retrieve": round(a, 4),
            "crag": round(c, 4),
            "delta": round(delta, 4),
        })

    print("=" * 70)

    # 保存报告
    print("\n[6/6] 保存报告...")
    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "top_k": TOP_K,
            "correct_threshold": CORRECT_THRESHOLD,
            "incorrect_threshold": INCORRECT_THRESHOLD,
        },
        "verdict_counts": verdict_counts,
        "execution_paths": crag_paths,
        "scores": report_rows,
        "always_dataset": always_rows,
        "crag_dataset": crag_rows,
    }
    with open("./crag_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  报告已保存：./crag_report.json")
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
