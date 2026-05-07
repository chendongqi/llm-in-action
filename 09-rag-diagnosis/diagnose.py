"""
RAG 效果诊断实验：故意制造 3 种典型问题，用 RAGAS 定位根因

诊断决策树：
  回答质量差
      ↓
  context_recall 低？
      ├─ 是 → 检索问题（Chunk 太小 / Top-K 不够）
      └─ 否 → faithfulness 低？
                  ├─ 是 → 生成问题（Prompt 引导幻觉）
                  └─ 否 → answer_relevancy 低？
                              └─ 是 → 答案偏题（Prompt 不聚焦）
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_pipeline import RAGPipeline

TESTSET_PATH = "../08-ragas-eval/data/manual_testset.json"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


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


def run_scenario(pipeline: RAGPipeline, testset: list, ragas_llm, ragas_emb, label: str) -> dict:
    """运行单个场景：执行 RAG 查询 + RAGAS 评估，返回各指标均值"""
    print(f"\n{'─'*60}")
    print(f"  场景: {label}")
    print(f"{'─'*60}")

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(testset, 1):
        q = item["question"]
        print(f"  [{i}/{len(testset)}] {q[:40]}...")
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

    print("  运行 RAGAS 评估...")
    result = evaluate(dataset=dataset, metrics=METRICS, llm=ragas_llm, embeddings=ragas_emb)

    scores = {}
    for name in METRIC_NAMES:
        vals = [s.get(name, np.nan) for s in result.scores if name in s]
        if vals:
            scores[name] = float(np.nanmean(vals))

    print(f"  结果: { {k: f'{v:.3f}' for k, v in scores.items()} }")
    return scores


def print_comparison(scenarios: list[tuple[str, dict, str]]):
    """
    打印多场景对比表
    scenarios: [(label, scores, problem_type), ...]
    """
    print("\n" + "=" * 80)
    print("  RAG 诊断对比报告")
    print("=" * 80)

    # 表头
    col_w = 24
    header = f"  {'指标':<20}"
    for label, _, _ in scenarios:
        header += f" {label:>{col_w}}"
    print(header)
    print("  " + "─" * (20 + (col_w + 1) * len(scenarios)))

    for name in METRIC_NAMES:
        row = f"  {name:<20}"
        vals = [s.get(name, 0) for _, s, _ in scenarios]
        baseline = vals[0]
        for i, v in enumerate(vals):
            marker = ""
            if i > 0:
                drop = baseline - v
                if drop > 0.2:
                    marker = " ← ✗"
                elif drop > 0.1:
                    marker = " ← ⚠"
            row += f" {v:>{col_w-len(marker)}.3f}{marker}"
        print(row)

    print("  " + "─" * (20 + (col_w + 1) * len(scenarios)))

    # 平均分
    row = f"  {'平均得分':<20}"
    for _, scores, _ in scenarios:
        avg = np.nanmean(list(scores.values()))
        row += f" {avg:>{col_w}.3f}"
    print(row)
    print("=" * 80)


def diagnostic_decision_tree(baseline: dict, problem_scores: list[tuple[str, dict, str]]):
    """
    基于 RAGAS 指标，按决策树逻辑输出诊断结论
    """
    print("\n" + "=" * 80)
    print("  诊断决策树分析")
    print("=" * 80)

    threshold = 0.12  # 超过此幅度认为显著下降

    for label, scores, problem_type in problem_scores:
        print(f"\n  【{label}】（预期问题类型：{problem_type}）")

        recall_drop = baseline.get("context_recall", 0) - scores.get("context_recall", 0)
        faith_drop = baseline.get("faithfulness", 0) - scores.get("faithfulness", 0)
        relev_drop = baseline.get("answer_relevancy", 0) - scores.get("answer_relevancy", 0)
        prec_drop = baseline.get("context_precision", 0) - scores.get("context_precision", 0)

        print(f"    步骤 1 → 检查 context_recall：下降 {recall_drop:+.3f}", end="")
        if recall_drop > threshold:
            print(f"（显著下降）")
            print(f"    ✗ 诊断：检索阶段有问题")
            if scores.get("context_recall", 1) < 0.5:
                print(f"       → 重要内容没被检索到，检查 chunk_size 和 top_k")
                print(f"       → 当前 top_k 可能太小，或 chunk 太碎导致语义不完整")
        else:
            print(f"（正常）")
            print(f"    步骤 2 → 检查 faithfulness：下降 {faith_drop:+.3f}", end="")
            if faith_drop > threshold:
                print(f"（显著下降）")
                print(f"    ✗ 诊断：生成阶段出现幻觉")
                print(f"       → 答案包含了上下文中没有的内容")
                print(f"       → 修复建议：优化 Prompt，明确要求只基于参考资料回答")
            else:
                print(f"（正常）")
                print(f"    步骤 3 → 检查 answer_relevancy：下降 {relev_drop:+.3f}", end="")
                if relev_drop > threshold:
                    print(f"（显著下降）")
                    print(f"    ✗ 诊断：答案偏题，未直接回答用户问题")
                    print(f"       → Prompt 格式要求导致答案冗长或结构固化")
                    print(f"       → 修复建议：简化 Prompt，去除强制格式约束")
                else:
                    print(f"（正常）")
                    print(f"    ✓ 诊断：该场景未检测到明显问题")

        if prec_drop > threshold:
            print(f"    ⚠ 附加发现：context_precision 下降 {prec_drop:+.3f}，检索结果中混入了噪声")

    print("\n" + "=" * 80)


def save_report(scenarios: list[tuple[str, dict, str]], output_path: str = "./diagnose_report.json"):
    report = []
    for label, scores, problem_type in scenarios:
        report.append({
            "scenario": label,
            "expected_problem": problem_type,
            "scores": scores,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存：{output_path}")


def main():
    print("=" * 80)
    print("  RAG 效果诊断：制造 3 种典型问题，用 RAGAS 定位根因")
    print("=" * 80)

    testset = json.load(open(TESTSET_PATH, "r", encoding="utf-8"))
    print(f"\n  测试集：{len(testset)} 条问答对")

    ragas_llm = create_ragas_llm()
    ragas_emb = create_ragas_embeddings()

    # ── 基准配置 ────────────────────────────────────────────────────────────
    print("\n\n【基准配置】chunk_size=512, top_k=4, 正常 Prompt")
    baseline_pipeline = RAGPipeline(
        chunk_size=512, chunk_overlap=50, top_k=4,
        prompt_type="baseline",
        persist_dir="./chroma_baseline",
    )
    baseline_pipeline.build_index(force_rebuild=True)
    baseline_scores = run_scenario(baseline_pipeline, testset, ragas_llm, ragas_emb, "基准配置")

    # ── 问题一：chunk 过小 + top_k 不足 → context_recall 低 ───────────────
    print("\n\n【问题一】chunk_size=64, top_k=1（检索召回不足）")
    print("  预期：文档被切成极小碎片，top_k=1 只取 1 条，大量信息丢失 → context_recall 低")
    p1_pipeline = RAGPipeline(
        chunk_size=64, chunk_overlap=0, top_k=1,
        prompt_type="baseline",
        persist_dir="./chroma_p1",
    )
    p1_pipeline.build_index(force_rebuild=True)
    p1_scores = run_scenario(p1_pipeline, testset, ragas_llm, ragas_emb, "问题一：检索召回不足")

    # ── 问题二：幻觉 Prompt → faithfulness 低 ─────────────────────────────
    print("\n\n【问题二】正常检索 + 幻觉诱导 Prompt（生成幻觉）")
    print("  预期：Prompt 鼓励模型超出上下文发挥 → faithfulness 低")
    p2_pipeline = RAGPipeline(
        chunk_size=512, chunk_overlap=50, top_k=4,
        prompt_type="hallucination",
        persist_dir="./chroma_baseline",  # 复用已有索引
    )
    p2_pipeline.build_index(force_rebuild=False)
    p2_scores = run_scenario(p2_pipeline, testset, ragas_llm, ragas_emb, "问题二：生成幻觉")

    # ── 问题三：强制学术综述格式 → answer_relevancy 低 ─────────────────────
    print("\n\n【问题三】正常检索 + 偏题学术综述 Prompt（答案偏题）")
    print("  预期：Prompt 强制输出学术综述格式，答案冗长不聚焦 → answer_relevancy 低")
    p3_pipeline = RAGPipeline(
        chunk_size=512, chunk_overlap=50, top_k=4,
        prompt_type="offtopic",
        persist_dir="./chroma_baseline",  # 复用已有索引
    )
    p3_pipeline.build_index(force_rebuild=False)
    p3_scores = run_scenario(p3_pipeline, testset, ragas_llm, ragas_emb, "问题三：答案偏题")

    # ── 汇总对比 ─────────────────────────────────────────────────────────────
    scenarios = [
        ("基准配置", baseline_scores, "无"),
        ("问题一：检索召回不足", p1_scores, "context_recall 低"),
        ("问题二：生成幻觉", p2_scores, "faithfulness 低"),
        ("问题三：答案偏题", p3_scores, "answer_relevancy 低"),
    ]

    print_comparison(scenarios)
    diagnostic_decision_tree(baseline_scores, scenarios[1:])
    save_report(scenarios)

    print("\n  ✅ 诊断实验完成！")


if __name__ == "__main__":
    main()
