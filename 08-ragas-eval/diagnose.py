"""
诊断对比实验：好配置 vs 差配置
演示 RAGAS 如何量化定位 RAG 系统的问题
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_pipeline import RAGPipeline, load_documents

TESTSET_PATH = "./data/manual_testset.json"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")


def create_ragas_llm():
    llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0.1)
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_rag_and_evaluate(pipeline: RAGPipeline, testset: list, llm, embeddings, label: str):
    """运行 RAG + 评估，返回结果"""
    print(f"\n{'='*50}")
    print(f"  配置: {label}")
    print(f"{'='*50}")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(testset, 1):
        q = item["question"]
        print(f"  查询 [{i}/{len(testset)}]: {q[:35]}...")
        result = pipeline.query(q)
        questions.append(q)
        answers.append(result["answer"])
        contexts_list.append([ctx.page_content for ctx in result["contexts"]])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print(f"\n  运行 RAGAS 评估...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {}
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric_name in metric_names:
        vals = [s.get(metric_name, np.nan) for s in result.scores if metric_name in s]
        if vals:
            scores[metric_name] = float(np.nanmean(vals))
    return scores


def print_comparison(good_scores: dict, bad_scores: dict):
    """打印对比表格"""
    print("\n" + "=" * 70)
    print(" 诊断对比：好配置 vs 差配置")
    print("=" * 70)
    print(f"\n  {'指标':<22} {'好配置':>10} {'差配置':>10} {'差异':>10} {'诊断':>12}")
    print("  " + "-" * 68)

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        g = good_scores.get(metric, 0)
        b = bad_scores.get(metric, 0)
        diff = g - b
        diagnosis = "✓ 正常" if diff < 0.15 else ("⚠ 警告" if diff < 0.3 else "✗ 严重")
        print(f"  {metric:<22} {g:>10.3f} {b:>10.3f} {diff:>+10.3f} {diagnosis:>12}")

    g_avg = sum(good_scores.values()) / len(good_scores)
    b_avg = sum(bad_scores.values()) / len(bad_scores)
    diff_avg = g_avg - b_avg
    print(f"  {'-'*68}")
    print(f"  {'平均得分':<22} {g_avg:>10.3f} {b_avg:>10.3f} {diff_avg:>+10.3f}")
    print("=" * 70)

    print("\n  📋 诊断结论:")
    if bad_scores.get("context_recall", 1) < good_scores.get("context_recall", 1) - 0.15:
        print("     → Context Recall 显著下降：检索阶段有问题（chunk 太大 / top-k 太少）")
    if bad_scores.get("context_precision", 1) < good_scores.get("context_precision", 1) - 0.15:
        print("     → Context Precision 显著下降：检索结果中混入了噪声")
    if bad_scores.get("faithfulness", 1) < good_scores.get("faithfulness", 1) - 0.15:
        print("     → Faithfulness 显著下降：答案出现了幻觉，上下文不足")
    if bad_scores.get("answer_relevancy", 1) < good_scores.get("answer_relevancy", 1) - 0.15:
        print("     → Answer Relevancy 显著下降：答案偏离了问题")


def main():
    print("=" * 70)
    print(" RAG 诊断实验：对比不同配置的评估结果")
    print("=" * 70)

    testset = json.load(open(TESTSET_PATH, "r", encoding="utf-8"))
    print(f"\n测试集: {len(testset)} 条问答对")

    llm = create_ragas_llm()
    embeddings = create_ragas_embeddings()

    # 好配置
    print("\n【好配置】chunk_size=512, overlap=50, top_k=4")
    good_pipeline = RAGPipeline(chunk_size=512, chunk_overlap=50, top_k=4, persist_dir="./chroma_db_good")
    good_pipeline.build_index(force_rebuild=True)
    good_scores = run_rag_and_evaluate(good_pipeline, testset, llm, embeddings, "好配置")

    # 差配置：chunk 太小导致上下文碎片化，top_k 太少导致召回不足
    print("\n【差配置】chunk_size=128, overlap=0, top_k=2")
    bad_pipeline = RAGPipeline(chunk_size=128, chunk_overlap=0, top_k=2, persist_dir="./chroma_db_bad")
    bad_pipeline.build_index(force_rebuild=True)
    bad_scores = run_rag_and_evaluate(bad_pipeline, testset, llm, embeddings, "差配置")

    # 对比
    print_comparison(good_scores, bad_scores)

    # 保存对比结果
    comparison = {
        "good_config": {"chunk_size": 512, "chunk_overlap": 50, "top_k": 4, "scores": good_scores},
        "bad_config": {"chunk_size": 128, "chunk_overlap": 0, "top_k": 2, "scores": bad_scores},
    }
    with open("./diagnose_report.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print("\n💾 诊断报告已保存至: ./diagnose_report.json")


if __name__ == "__main__":
    main()
