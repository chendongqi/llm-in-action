"""
LightRAG 评测脚本

1. 建库：把 kb-00-testset/data/raw_docs 里的文档索引进 LightRAG
2. 检索：对 testset.jsonl 里的 89 道题逐一检索（mix 模式）
3. 评分：用 RAGAS 计算四项指标 + 自定义边界拒答率 + P90 延迟

输出：
  data/results/answers.jsonl      - 每题的检索结果和生成答案
  data/results/eval_report.json   - 汇总指标
"""

import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

DOCS_DIR = SCRIPT_DIR.parent / "kb-00-testset/data/raw_docs"
TESTSET_PATH = SCRIPT_DIR.parent / "kb-00-testset/data/output/testset.jsonl"
STORAGE_DIR = SCRIPT_DIR / "data/rag_storage"
RESULTS_DIR = SCRIPT_DIR / "data/results"

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# ── LightRAG 初始化 ───────────────────────────────────────────────────────────

async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )

async def _raw_embed(texts: list[str]):
    """直接调 OpenAI embeddings 接口，支持任意维度（绕过 lightrag openai_embed 的 1536 硬编码）"""
    import numpy as np
    client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in resp.data], dtype=np.float32)

async def build_rag() -> LightRAG:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(STORAGE_DIR),
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_raw_embed,
        ),
    )
    await rag.initialize_storages()
    return rag

# ── 建库 ──────────────────────────────────────────────────────────────────────

async def index_docs(rag: LightRAG):
    """把 raw_docs 里的文档插入 LightRAG"""
    md_files = sorted(DOCS_DIR.glob("*.md"))
    print(f"建库：{len(md_files)} 个文档")

    for i, f in enumerate(md_files):
        content = f.read_text(encoding="utf-8").strip()
        if len(content) < 200:
            continue
        print(f"  [{i+1}/{len(md_files)}] {f.name}...", end=" ", flush=True)
        await rag.ainsert(content)
        print("✓")

    print("建库完成")

# ── 检索 ──────────────────────────────────────────────────────────────────────

async def query_all(rag: LightRAG, testset: list[dict]) -> list[dict]:
    """对所有题目检索，记录答案和耗时"""
    results = []
    for i, item in enumerate(testset):
        question = item["question"]
        print(f"  [{i+1}/{len(testset)}] {question[:60]}...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            answer = await rag.aquery(
                question,
                param=QueryParam(mode="mix"),
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"✓ {latency_ms:.0f}ms")
        except Exception as e:
            answer = ""
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"✗ {e}")

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "contexts": [],          # LightRAG 不直接暴露 retrieved chunks，留空
            "question_type": item.get("question_type", "unknown"),
            "source_docs": item.get("source_docs", []),
            "latency_ms": latency_ms,
        })

    return results

# ── 评分 ──────────────────────────────────────────────────────────────────────

def compute_boundary_refusal_rate(results: list[dict]) -> float:
    """边界题拒答率：answer 中包含拒答关键词的比例"""
    REFUSAL_KEYWORDS = [
        "don't have", "do not have", "no information", "not contain",
        "not available", "cannot find", "not mentioned", "doesn't contain",
        "does not contain", "no relevant", "not found",
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
    """P90 检索延迟（ms）"""
    latencies = sorted(r["latency_ms"] for r in results)
    idx = int(len(latencies) * 0.9)
    return latencies[min(idx, len(latencies) - 1)]


def compute_simple_answer_match(results: list[dict]) -> dict:
    """简单答案匹配率（不需要 LLM judge）——关键词重叠"""
    by_type = {}
    for qt in ["single_hop", "multi_hop", "boundary"]:
        items = [r for r in results if r["question_type"] == qt]
        if not items:
            continue
        matches = []
        for r in items:
            gt_words = set(r["ground_truth"].lower().split())
            ans_words = set(r["answer"].lower().split())
            # Jaccard 相似度
            if not gt_words:
                continue
            overlap = len(gt_words & ans_words) / len(gt_words | ans_words)
            matches.append(overlap)
        by_type[qt] = round(sum(matches) / len(matches), 3) if matches else 0.0
    return by_type

# ── 主流程 ────────────────────────────────────────────────────────────────────

async def main(skip_index: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== LightRAG 评测 ==")
    print(f"  文档目录 : {DOCS_DIR}")
    print(f"  测试集   : {TESTSET_PATH}")
    print(f"  LLM      : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  Embedding: {EMBEDDING_MODEL}")
    print()

    rag = await build_rag()

    # 建库（如果 storage 目录已存在且非空则跳过）
    storage_files = list(STORAGE_DIR.glob("*"))
    if skip_index or (storage_files and any(f.suffix in (".json", ".db") for f in storage_files)):
        print("跳过建库（已有索引）")
    else:
        await index_docs(rag)

    # 加载测试集
    testset = [json.loads(l) for l in TESTSET_PATH.read_text().splitlines() if l.strip()]
    print(f"\n开始检索（{len(testset)} 题）...")
    results = await query_all(rag, testset)

    # 保存答案
    answers_path = RESULTS_DIR / "answers.jsonl"
    with open(answers_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n答案保存至: {answers_path}")

    # 计算指标
    boundary_rate = compute_boundary_refusal_rate(results)
    p90 = compute_p90_latency(results)
    match_by_type = compute_simple_answer_match(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    report = {
        "framework": "LightRAG",
        "version": "1.5.6",
        "query_mode": "mix",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "total_questions": len(results),
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
    print(f"  边界拒答率  : {boundary_rate:.1%}")
    print(f"  P90 延迟    : {p90:.0f} ms")
    print(f"  平均延迟    : {avg_latency:.0f} ms")
    print(f"  答案匹配率  :")
    for qt, score in match_by_type.items():
        print(f"    {qt}: {score:.3f}")
    print(f"\n完整报告: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true", help="跳过建库，直接评测")
    args = parser.parse_args()
    asyncio.run(main(skip_index=args.skip_index))
