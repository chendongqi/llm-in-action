"""
Self-RAG 实验：固定检索 vs 自适应检索

传统 RAG 每次都检索，即使问题根本不需要外部知识。
Self-RAG 让模型自己判断：
  1. 这个问题需要检索吗？（Retrieve 决策）
  2. 检索到的文档和问题相关吗？（Relevance 评估）
  3. 最终回答是否基于文档？（Support 评估）

本实验用 LangGraph 实现简化版 Self-RAG 流程，
并和固定检索对比：回答质量 + Token 消耗。

运行方式：
    conda activate dev_base
    python self_rag.py
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


# ── Prompts ────────────────────────────────────────────────────────────────────

# 1. 检索决策：判断这个问题是否需要检索外部知识
RETRIEVE_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个 RAG 系统的路由决策器。判断以下问题是否需要检索外部知识库才能回答。\n\n"
     "需要检索的情况：\n"
     "- 问题涉及具体技术细节、参数、推荐选型等事实性内容\n"
     "- 回答需要参考特定领域的专业知识\n\n"
     "不需要检索的情况：\n"
     "- 问题是简单的常识性问题\n"
     "- 问题是数学计算、逻辑推理\n"
     "- 问题是闲聊、问候\n\n"
     "只输出 yes 或 no，不要解释。"),
    ("human", "问题：{question}"),
])

# 2. 相关性评估：判断检索到的文档是否与问题相关
RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "判断以下文档是否与问题相关，能够帮助回答该问题。\n"
     "只输出 relevant 或 irrelevant，不要解释。"),
    ("human", "问题：{question}\n\n文档：{document}"),
])

# 3. 基于检索结果生成答案
RAG_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请严格根据提供的参考资料回答问题。"
               "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
])

# 4. 直接生成答案（不使用检索结果）
DIRECT_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请根据你的知识直接回答问题。回答要简洁准确。"),
    ("human", "问题：{question}\n\n请回答："),
])

# 5. Support 评估：回答是否基于文档支撑
SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "判断以下回答是否主要基于给出的文档内容，而不是凭空生成的。\n"
     "只输出 supported 或 unsupported，不要解释。"),
    ("human", "文档：{context}\n\n回答：{answer}"),
])


# ── LangGraph State ────────────────────────────────────────────────────────────

class SelfRAGState(TypedDict):
    question: str
    need_retrieve: str           # "yes" | "no"
    retrieved_docs: list[Document]
    relevant_docs: list[Document]
    answer: str
    support_verdict: str         # "supported" | "unsupported"
    token_count: int             # 估算的 token 消耗（调用次数 × 平均token）
    path: list[str]              # 记录执行路径，方便分析


# ── Graph 节点函数 ─────────────────────────────────────────────────────────────

def make_decide_node(llm):
    """节点1：决定是否需要检索"""
    chain = RETRIEVE_DECISION_PROMPT | llm | StrOutputParser()

    def decide(state: SelfRAGState) -> SelfRAGState:
        result = chain.invoke({"question": state["question"]})
        verdict = "yes" if "yes" in result.lower() else "no"
        return {
            **state,
            "need_retrieve": verdict,
            "token_count": state.get("token_count", 0) + 150,   # 估算决策调用的 token
            "path": state.get("path", []) + [f"decide→{verdict}"],
        }

    return decide


def make_retrieve_node(retriever):
    """节点2：执行检索"""
    def retrieve(state: SelfRAGState) -> SelfRAGState:
        docs = retriever.invoke(state["question"])
        return {
            **state,
            "retrieved_docs": docs,
            "path": state.get("path", []) + ["retrieve"],
        }

    return retrieve


def make_filter_node(llm):
    """节点3：过滤不相关文档"""
    chain = RELEVANCE_PROMPT | llm | StrOutputParser()

    def filter_docs(state: SelfRAGState) -> SelfRAGState:
        relevant = []
        token_used = state.get("token_count", 0)
        for doc in state["retrieved_docs"]:
            result = chain.invoke({
                "question": state["question"],
                "document": doc.page_content[:300],   # 截断避免超长
            })
            token_used += 200   # 估算每次相关性判断的 token
            if "relevant" in result.lower() and "irrelevant" not in result.lower():
                relevant.append(doc)

        # 如果全部被过滤掉，保留原始检索结果（兜底）
        if not relevant:
            relevant = state["retrieved_docs"]

        return {
            **state,
            "relevant_docs": relevant,
            "token_count": token_used,
            "path": state.get("path", []) + [f"filter({len(relevant)}/{len(state['retrieved_docs'])})"],
        }

    return filter_docs


def make_rag_generate_node(llm):
    """节点4a：基于检索结果生成答案"""
    chain = RAG_GENERATE_PROMPT | llm | StrOutputParser()

    def rag_generate(state: SelfRAGState) -> SelfRAGState:
        docs = state.get("relevant_docs") or state.get("retrieved_docs", [])
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer = chain.invoke({"context": context, "question": state["question"]})
        return {
            **state,
            "answer": answer,
            "token_count": state.get("token_count", 0) + 600,
            "path": state.get("path", []) + ["rag_generate"],
        }

    return rag_generate


def make_direct_generate_node(llm):
    """节点4b：不检索，直接生成答案"""
    chain = DIRECT_GENERATE_PROMPT | llm | StrOutputParser()

    def direct_generate(state: SelfRAGState) -> SelfRAGState:
        answer = chain.invoke({"question": state["question"]})
        return {
            **state,
            "answer": answer,
            "retrieved_docs": [],
            "relevant_docs": [],
            "token_count": state.get("token_count", 0) + 400,
            "path": state.get("path", []) + ["direct_generate"],
        }

    return direct_generate


def make_support_node(llm):
    """节点5：评估回答是否有文档支撑"""
    chain = SUPPORT_PROMPT | llm | StrOutputParser()

    def check_support(state: SelfRAGState) -> SelfRAGState:
        docs = state.get("relevant_docs") or state.get("retrieved_docs", [])
        if not docs:
            return {**state, "support_verdict": "unsupported",
                    "path": state.get("path", []) + ["support→skip"]}

        context = "\n\n".join(d.page_content[:200] for d in docs[:2])
        result = chain.invoke({"context": context, "answer": state["answer"]})
        verdict = "supported" if "supported" in result.lower() and "unsupported" not in result.lower() else "unsupported"
        return {
            **state,
            "support_verdict": verdict,
            "token_count": state.get("token_count", 0) + 250,
            "path": state.get("path", []) + [f"support→{verdict}"],
        }

    return check_support


# ── 路由函数 ───────────────────────────────────────────────────────────────────

def route_after_decide(state: SelfRAGState) -> Literal["retrieve", "direct_generate"]:
    return "retrieve" if state["need_retrieve"] == "yes" else "direct_generate"


def route_after_support(state: SelfRAGState) -> Literal["end"]:
    # 简化版：无论 support 结果如何，直接结束
    # 完整版 Self-RAG 会在 unsupported 时重新生成
    return "end"


# ── 构建 Self-RAG Graph ────────────────────────────────────────────────────────

def build_self_rag_graph(llm, retriever):
    graph = StateGraph(SelfRAGState)

    # 添加节点
    graph.add_node("decide", make_decide_node(llm))
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("filter", make_filter_node(llm))
    graph.add_node("rag_generate", make_rag_generate_node(llm))
    graph.add_node("direct_generate", make_direct_generate_node(llm))
    graph.add_node("support_check", make_support_node(llm))

    # 设置入口
    graph.set_entry_point("decide")

    # 条件路由：decide → retrieve 或 direct_generate
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "retrieve": "retrieve",
            "direct_generate": "direct_generate",
        },
    )

    # 固定边
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "rag_generate")
    graph.add_edge("rag_generate", "support_check")
    graph.add_edge("direct_generate", "support_check")
    graph.add_edge("support_check", END)

    return graph.compile()


# ── 基准：固定检索 Pipeline ────────────────────────────────────────────────────

def run_always_retrieve(retriever, llm, question: str) -> dict:
    """传统 RAG：无论什么问题都检索"""
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = RAG_GENERATE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {
        "answer": answer,
        "retrieved_docs": docs,
        "token_count": 600,   # 仅生成调用
        "path": ["always_retrieve", "rag_generate"],
    }


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


def build_llm(temperature=0):
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
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
    vs = Chroma(collection_name="self_rag", embedding_function=embeddings)
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
    print("  Self-RAG 实验：固定检索 vs 自适应检索（LangGraph）")
    print("=" * 70)

    print("\n[1/6] 加载数据...")
    raw_data = load_raw_data()
    testset = load_testset()
    # 额外添加几条不需要检索的问题，体现自适应检索的价值
    mixed_testset = testset + [
        {
            "question": "1 + 1 等于几？",
            "ground_truth": "2",
            "relevant_doc_ids": [],
        },
        {
            "question": "今天天气怎么样？",
            "ground_truth": "我无法获取实时天气信息。",
            "relevant_doc_ids": [],
        },
        {
            "question": "用 Python 写一个求最大公约数的函数",
            "ground_truth": "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
            "relevant_doc_ids": [],
        },
    ]
    print(f"  知识库：{len(raw_data)} 篇，测试集：{len(mixed_testset)} 条（含 {len(mixed_testset)-len(testset)} 条无需检索）")

    print("\n[2/6] 初始化模型和向量库...")
    embeddings = build_embeddings()
    llm = build_llm()
    vectorstore = build_vectorstore(raw_data, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print("\n[3/6] 构建 Self-RAG Graph...")
    self_rag_graph = build_self_rag_graph(llm, retriever)

    print("\n[4/6] 运行两种策略...")

    # 固定检索
    print("  运行固定检索（Always Retrieve）...")
    always_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    always_tokens = 0
    always_paths = []
    for item in mixed_testset:
        result = run_always_retrieve(retriever, llm, item["question"])
        always_rows["question"].append(item["question"])
        always_rows["answer"].append(result["answer"])
        always_rows["contexts"].append([d.page_content for d in result["retrieved_docs"]])
        always_rows["ground_truth"].append(item["ground_truth"])
        always_tokens += result["token_count"]
        always_paths.append(result["path"])
        time.sleep(0.3)

    # Self-RAG
    print("  运行 Self-RAG（Adaptive）...")
    self_rag_rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    self_rag_tokens = 0
    self_rag_paths = []
    retrieve_decisions = {"yes": 0, "no": 0}
    for item in mixed_testset:
        init_state: SelfRAGState = {
            "question": item["question"],
            "need_retrieve": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "answer": "",
            "support_verdict": "",
            "token_count": 0,
            "path": [],
        }
        result = self_rag_graph.invoke(init_state)
        self_rag_rows["question"].append(item["question"])
        self_rag_rows["answer"].append(result["answer"])
        self_rag_rows["contexts"].append(
            [d.page_content for d in result.get("relevant_docs") or result.get("retrieved_docs", [])]
        )
        self_rag_rows["ground_truth"].append(item["ground_truth"])
        self_rag_tokens += result["token_count"]
        self_rag_paths.append(result["path"])
        retrieve_decisions[result["need_retrieve"]] += 1
        time.sleep(0.3)

    # 打印决策明细
    print("\n  Self-RAG 检索决策明细：")
    for q, path, item in zip(
        [i["question"] for i in mixed_testset], self_rag_paths, mixed_testset
    ):
        decision = "✓ 检索" if "decide→yes" in str(path) else "✗ 跳过"
        print(f"    [{decision}] {q[:45]}")

    print(f"\n  检索决策统计：需要检索 {retrieve_decisions['yes']} 条，"
          f"直接回答 {retrieve_decisions['no']} 条")

    print("\n[5/6] 计算 RAGAS 指标（仅对知识库相关问题评估）...")
    # RAGAS 只评估有 ground_truth 且需要检索的部分
    rag_questions = len(testset)
    always_rag_dataset = Dataset.from_dict({
        k: v[:rag_questions] for k, v in always_rows.items()
    })
    self_rag_rag_dataset = Dataset.from_dict({
        k: v[:rag_questions] for k, v in self_rag_rows.items()
    })

    eval_llm = build_llm(temperature=0)
    always_scores = run_ragas_eval(always_rag_dataset, eval_llm, embeddings)
    self_rag_scores = run_ragas_eval(self_rag_rag_dataset, eval_llm, embeddings)

    # ── 输出报告
    print("\n" + "=" * 70)
    print("  RAGAS 指标对比（知识库相关问题）")
    print("=" * 70)
    print(f"\n  {'指标':<22} {'固定检索':>12} {'Self-RAG':>12}  {'变化':>8}")
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
        s = self_rag_scores[key]
        delta = s - a
        flag = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"  {label:<22} {a:>12.3f} {s:>12.3f}  {flag}{delta:+.3f}")
        report_rows.append({"metric": key, "always_retrieve": round(a, 4), "self_rag": round(s, 4), "delta": round(delta, 4)})

    print("=" * 70)

    print("\n  Token 消耗对比（全部问题）：")
    print(f"  固定检索总消耗：~{always_tokens:,} tokens")
    print(f"  Self-RAG 总消耗：~{self_rag_tokens:,} tokens")
    saved_pct = (always_tokens - self_rag_tokens) / always_tokens * 100
    if saved_pct > 0:
        print(f"  Self-RAG 节省：~{saved_pct:.1f}%（跳过了 {retrieve_decisions['no']} 次检索）")
    else:
        print(f"  Self-RAG 额外消耗：~{abs(saved_pct):.1f}%（决策和评估节点的开销）")

    # 保存报告
    print("\n[6/6] 保存报告...")
    report = {
        "config": {"chunk_size": CHUNK_SIZE, "top_k": TOP_K, "llm_model": LLM_MODEL},
        "retrieve_decisions": retrieve_decisions,
        "token_comparison": {
            "always_retrieve": always_tokens,
            "self_rag": self_rag_tokens,
        },
        "execution_paths": {
            "always_retrieve": always_paths,
            "self_rag": self_rag_paths,
        },
        "scores": report_rows,
        "always_dataset": always_rag_dataset.to_dict(),
        "self_rag_dataset": self_rag_rag_dataset.to_dict(),
    }
    with open("./self_rag_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  报告已保存：./self_rag_report.json")
    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
