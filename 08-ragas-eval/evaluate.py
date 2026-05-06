"""
RAGAS 评估主脚本
使用 4 个核心指标评估 RAG Pipeline 质量
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

from rag_pipeline import RAGPipeline, load_documents, DATA_PATH

# ─── 配置 ──────────────────────────────────────────────────────────────
TESTSET_PATH = "./data/manual_testset.json"
OUTPUT_PATH = "./evaluation_report.json"
CHROMA_PATH = "./chroma_db"

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def load_testset(path: str) -> list[dict]:
    """加载测试集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_ragas_llm():
    """创建 RAGAS 所需的 LLM Wrapper"""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.1,
    )
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings():
    """创建 RAGAS 所需的 Embeddings Wrapper"""
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def prepare_dataset(pipeline: RAGPipeline, testset: list[dict]) -> Dataset:
    """运行 RAG 并准备 RAGAS 数据集"""
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    print("\n[1/3] 执行 RAG 查询，收集 answers 和 contexts...")
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
    return dataset


def run_evaluation(dataset: Dataset, llm, embeddings):
    """运行 RAGAS 评估"""
    print("\n[2/3] 运行 RAGAS 评估（4 个核心指标）...")
    print("     - Faithfulness（忠实度）")
    print("     - Answer Relevancy（答案相关性）")
    print("     - Context Precision（上下文精确度）")
    print("     - Context Recall（上下文召回率）")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    return result


def print_report(result, dataset: Dataset):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print(" RAGAS 评估报告")
    print("=" * 60)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {}
    for metric_name in metric_names:
        vals = [s.get(metric_name, np.nan) for s in result.scores if metric_name in s]
        if vals:
            scores[metric_name] = np.nanmean(vals)

    # 总体平均分
    print("\n📊 总体得分:")
    for name, score in scores.items():
        if np.isnan(score):
            bar = "N/A"
            print(f"   {name:22s}: {bar}")
        else:
            bar = "█" * int(score * 20)
            print(f"   {name:22s}: {score:.3f} {bar}")

    avg = np.nanmean(list(scores.values())) if scores else 0
    print(f"\n   {'平均得分':22s}: {avg:.3f}")

    # 逐题明细
    print("\n📋 逐题明细:")
    header = f"{'#':>3} {'Question':<30} {'Faith':>6} {'AnsRel':>6} {'CtxPre':>6} {'CtxRec':>6}"
    print("   " + header)
    print("   " + "-" * 64)

    for i in range(len(dataset)):
        q = dataset[i]["question"][:28]
        f_val = result.scores[i].get("faithfulness", np.nan) if i < len(result.scores) else np.nan
        a_val = result.scores[i].get("answer_relevancy", np.nan) if i < len(result.scores) else np.nan
        p_val = result.scores[i].get("context_precision", np.nan) if i < len(result.scores) else np.nan
        r_val = result.scores[i].get("context_recall", np.nan) if i < len(result.scores) else np.nan
        fv = f"{f_val:>6.2f}" if not np.isnan(f_val) else "  N/A"
        av = f"{a_val:>6.2f}" if not np.isnan(a_val) else "  N/A"
        pv = f"{p_val:>6.2f}" if not np.isnan(p_val) else "  N/A"
        rv = f"{r_val:>6.2f}" if not np.isnan(r_val) else "  N/A"
        print(f"   {i+1:>3} {q:<30} {fv} {av} {pv} {rv}")

    # 最差指标提示
    valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
    if valid_scores:
        min_metric = min(valid_scores.items(), key=lambda x: x[1])
        print(f"\n⚠️  最差指标: {min_metric[0]} ({min_metric[1]:.3f})")
    print("=" * 60)


def save_report(result, dataset: Dataset, output_path: str):
    """保存评估报告到 JSON"""
    report = {
        "summary": {},
        "per_question": [],
    }

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric_name in metric_names:
        vals = [s.get(metric_name, np.nan) for s in result.scores if metric_name in s]
        if vals:
            report["summary"][metric_name] = float(np.nanmean(vals))

    for i in range(len(dataset)):
        item = {
            "question": dataset[i]["question"],
            "answer": dataset[i]["answer"][:200],
            "ground_truth": dataset[i]["ground_truth"][:200],
        }
        for metric_name in metric_names:
            if i < len(result.scores) and metric_name in result.scores[i]:
                v = result.scores[i][metric_name]
                item[metric_name] = float(v) if not np.isnan(v) else None
        report["per_question"].append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 评估报告已保存至: {output_path}")


def main():
    print("=" * 60)
    print(" RAG 系统评估（RAGAS）")
    print("=" * 60)

    # 1. 加载测试集
    testset = load_testset(TESTSET_PATH)
    print(f"\n[0/3] 加载测试集: {len(testset)} 条")

    # 2. 构建 RAG Pipeline
    print("\n构建 RAG Pipeline...")
    pipeline = RAGPipeline(
        chunk_size=512,
        chunk_overlap=50,
        top_k=4,
        persist_dir=CHROMA_PATH,
    )
    pipeline.build_index(force_rebuild=True)

    # 3. 准备数据集
    dataset = prepare_dataset(pipeline, testset)

    # 4. 创建 RAGAS LLM/Embeddings
    llm = create_ragas_llm()
    embeddings = create_ragas_embeddings()

    # 5. 运行评估
    result = run_evaluation(dataset, llm, embeddings)

    # 6. 打印并保存报告
    print_report(result, dataset)
    save_report(result, dataset, OUTPUT_PATH)

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
