"""
HippoRAG 评测脚本

流程：
1. 加载 31 个文档，批量 index（记录耗时）
2. 对 testset.jsonl 里 89 道题逐一调用 rag_qa（每题单独计时）
3. 评分：边界拒答率 + P90 延迟 + Jaccard 答案匹配 + 索引构建时间

前置条件：
- 停止 QAnything Docker（释放 VRAM：QAnything ~8GB，NV-Embed-v2 ~8GB，RTX 3060 12GB 不够同时跑）
- 安装：pip install hipporag（独立环境或 llm_base 中已解决冲突）
- EMBEDDING_MODEL_PATH 指向本地 NV-Embed-v2 目录

用法：
  # 完整流程（建库 + 评测）
  CUDA_VISIBLE_DEVICES=0 python run_eval.py

  # 跳过建库（已有索引，数据保存在 data/outputs/）
  python run_eval.py --skip-index
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# ── 路径配置 ──────────────────────────────────────────────────────────────────

DOCS_DIR = SCRIPT_DIR.parent / "kb-00-testset/data/raw_docs"
TESTSET_PATH = SCRIPT_DIR.parent / "kb-00-testset/data/output/testset.jsonl"
SAVE_DIR = str(SCRIPT_DIR / "data/outputs")
RESULTS_DIR = SCRIPT_DIR / "data/results"

# ── 环境变量 ──────────────────────────────────────────────────────────────────

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4-flash")
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/mnt/hdd/Database/01_Reference/OpenSource/LLM/12_Model/NV-Embed-v2"
)

# ── NV-Embed-v2 离线补丁 ────────────────────────────────────────────────────
# modeling_nvembed.py 在 __init__ 里调 AutoTokenizer.from_pretrained("nvidia/NV-Embed-v2")
# 导致 HF hub 联网检查失败。在 transformers 导入之前 monkey-patch，把
# "nvidia/NV-Embed-v2" 重定向到本地模型目录。
import transformers.models.auto.tokenization_auto as _tok_auto
_orig_tok_from_pretrained = _tok_auto.AutoTokenizer.from_pretrained.__func__

@classmethod
def _patched_tok_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    _NV_EMBED_ALIASES = {"nvidia/NV-Embed-v2", "nvidia/nv-embed-v2"}
    if str(pretrained_model_name_or_path) in _NV_EMBED_ALIASES:
        pretrained_model_name_or_path = EMBEDDING_MODEL_PATH
        kwargs["local_files_only"] = True
    return _orig_tok_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

_tok_auto.AutoTokenizer.from_pretrained = _patched_tok_from_pretrained

# ── 文档加载 ──────────────────────────────────────────────────────────────────

def load_docs() -> list[str]:
    """读取所有 Markdown 文档，每个文件作为一个 passage。"""
    docs = []
    for f in sorted(DOCS_DIR.glob("*.md")):
        content = f.read_text(encoding="utf-8").strip()
        if content:
            # 在文档开头加文件名作为标题，帮助 HippoRAG 关联实体
            docs.append(f"# {f.stem}\n\n{content}")
    print(f"  加载 {len(docs)} 个文档")
    return docs


# ── 知识库构建 ────────────────────────────────────────────────────────────────

def build_index(hipporag, docs: list[str]) -> float:
    """运行 hipporag.index()，返回耗时秒数。"""
    print(f"\n== 开始构建 HippoRAG 记忆图谱 ==")
    print(f"  文档数量: {len(docs)}")
    print(f"  LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  Embedding: {EMBEDDING_MODEL_PATH}")
    print("  (OpenIE 实体提取 + 图构建，预计 10-20 分钟)")

    t0 = time.perf_counter()
    hipporag.index(docs=docs)
    elapsed = time.perf_counter() - t0

    print(f"\n记忆图谱构建完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return elapsed


# ── 查询 ───────────────────────────────────────────────────────────────────────

def query_single(hipporag, question: str) -> tuple[str, float]:
    """
    对单个问题调用 rag_qa，返回 (answer, latency_ms)。
    HippoRAG 的 rag_qa 支持批量，但为了精确计时每题单独调用。
    """
    t0 = time.perf_counter()
    _, response_messages, _ = hipporag.rag_qa(queries=[question])
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = response_messages[0] if response_messages else ""
    return answer, latency_ms


def query_all(hipporag, testset: list[dict]) -> list[dict]:
    """对所有题目逐一发问，记录答案和延迟。"""
    results = []
    for i, item in enumerate(testset):
        question = item["question"]
        print(f"  [{i+1}/{len(testset)}] {question[:60]}...", end=" ", flush=True)

        try:
            answer, latency_ms = query_single(hipporag, question)
            print(f"✓ {latency_ms:.0f}ms")
        except Exception as e:
            answer = ""
            latency_ms = 0.0
            print(f"✗ {e}")

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "question_type": item.get("question_type", "unknown"),
            "source_docs": item.get("source_docs", []),
            "latency_ms": latency_ms,
        })

    return results


# ── 评分 ───────────────────────────────────────────────────────────────────────

def compute_boundary_refusal_rate(results: list[dict]) -> float:
    REFUSAL_KEYWORDS = [
        "don't have", "do not have", "no information", "not contain",
        "not available", "cannot find", "not mentioned", "doesn't contain",
        "does not contain", "no relevant", "not found", "cannot provide",
        "unable to find", "抱歉", "无法回答", "没有相关",
        "no specific information", "not provided", "not included",
        "doesn't mention", "does not mention", "no data",
    ]
    boundary = [r for r in results if r["question_type"] == "boundary"]
    if not boundary:
        return 0.0
    refused = sum(
        1 for r in boundary
        if any(kw in r["answer"].lower() for kw in REFUSAL_KEYWORDS)
    )
    return refused / len(boundary)


def compute_p90_latency(results: list[dict]) -> float:
    latencies = sorted(r["latency_ms"] for r in results)
    idx = int(len(latencies) * 0.9)
    return latencies[min(idx, len(latencies) - 1)]


def compute_answer_match(results: list[dict]) -> dict:
    by_type = {}
    for qt in ["single_hop", "multi_hop", "boundary"]:
        items = [r for r in results if r["question_type"] == qt]
        if not items:
            continue
        matches = []
        for r in items:
            gt_words = set(r["ground_truth"].lower().split())
            ans_words = set(r["answer"].lower().split())
            if not gt_words:
                continue
            overlap = len(gt_words & ans_words) / len(gt_words | ans_words)
            matches.append(overlap)
        by_type[qt] = round(sum(matches) / len(matches), 3) if matches else 0.0
    return by_type


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(skip_index: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / "data/outputs").mkdir(parents=True, exist_ok=True)

    print("== HippoRAG 评测 ==")
    print(f"  文档目录   : {DOCS_DIR}")
    print(f"  测试集     : {TESTSET_PATH}")
    print(f"  LLM        : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  Embedding  : {EMBEDDING_MODEL_PATH}")
    print()

    # 设置 OPENAI_API_KEY（HippoRAG 内部使用 openai 客户端）
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY

    # 初始化 HippoRAG
    print("== 初始化 HippoRAG ==")
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    global_config = BaseConfig(
        embedding_max_seq_len=512,   # 降低显存需求（默认 2048）
        embedding_batch_size=1,      # 一次一条，避免 OOM
        embedding_model_dtype="bfloat16",  # 强制 bf16，避免 auto 选到 float32
    )

    hipporag = HippoRAG(
        global_config=global_config,
        save_dir=SAVE_DIR,
        llm_model_name=LLM_MODEL,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL_PATH,
    )

    # 建库
    index_time_s = 0.0
    docs = load_docs()

    if skip_index:
        # 检查是否已有索引（save_dir 下有子目录）
        outputs = list(Path(SAVE_DIR).iterdir()) if Path(SAVE_DIR).exists() else []
        if outputs:
            print(f"跳过建库（已有索引: {len(outputs)} 个子目录）")
        else:
            print("未找到索引，强制重新建库...")
            index_time_s = build_index(hipporag, docs)
    else:
        index_time_s = build_index(hipporag, docs)

    # 加载测试集
    testset = [json.loads(l) for l in TESTSET_PATH.read_text().splitlines() if l.strip()]
    print(f"\n== 开始检索（{len(testset)} 题）==")
    results = query_all(hipporag, testset)

    # 保存答案
    answers_path = RESULTS_DIR / "answers.jsonl"
    with open(answers_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 计算指标
    boundary_rate = compute_boundary_refusal_rate(results)
    p90 = compute_p90_latency(results)
    match_by_type = compute_answer_match(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    report = {
        "framework": "HippoRAG",
        "version": "2.0.0",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL_PATH,
        "total_questions": len(results),
        "index_time_s": round(index_time_s, 1),
        "index_time_min": round(index_time_s / 60, 1),
        "boundary_refusal_rate": round(boundary_rate, 3),
        "p90_latency_ms": round(p90, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "answer_match_by_type": match_by_type,
        "note": "answer_match uses keyword overlap (Jaccard), not LLM judge",
    }

    report_path = RESULTS_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n== 评测结果 ==")
    print(f"  索引构建时间 : {index_time_s:.0f}s ({index_time_s/60:.1f} min)")
    print(f"  边界拒答率   : {boundary_rate:.1%}")
    print(f"  P90 延迟     : {p90:.0f} ms")
    print(f"  平均延迟     : {avg_latency:.0f} ms")
    print(f"  答案匹配率   :")
    for qt, score in match_by_type.items():
        print(f"    {qt}: {score:.3f}")
    print(f"\n  完整报告: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true", help="跳过建库，直接评测")
    args = parser.parse_args()
    main(skip_index=args.skip_index)
