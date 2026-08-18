"""
GraphRAG 评测脚本

流程：
1. 初始化工作区（生成 prompts、复制文档）
2. graphrag index 构建知识图谱（记录耗时）
3. 对 testset.jsonl 里 89 道题用 local 模式逐一查询
4. 评分：边界拒答率 + P90 延迟 + Jaccard 答案匹配 + 索引构建时间

输出：
  data/results/answers.jsonl   - 每题的答案和延迟
  data/results/eval_report.json - 汇总指标

用法：
  # 完整流程（建库 + 评测）
  conda activate llm_base
  cd kb-03-graphrag-eval
  python run_eval.py

  # 跳过建库（已有索引）
  python run_eval.py --skip-index
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# ── 路径配置 ──────────────────────────────────────────────────────────────────

DOCS_DIR = SCRIPT_DIR.parent / "kb-00-testset/data/raw_docs"
TESTSET_PATH = SCRIPT_DIR.parent / "kb-00-testset/data/output/testset.jsonl"
WORKSPACE_DIR = SCRIPT_DIR / "data/workspace"
RESULTS_DIR = SCRIPT_DIR / "data/results"

# ── 环境变量 ──────────────────────────────────────────────────────────────────

LLM_API_KEY = os.environ["GRAPHRAG_LLM_API_KEY"]
LLM_BASE_URL = os.environ["GRAPHRAG_LLM_BASE_URL"]
LLM_MODEL = os.environ.get("GRAPHRAG_LLM_MODEL", "glm-4-flash")

# ── 初始化工作区 ───────────────────────────────────────────────────────────────

def init_workspace():
    """创建工作区目录结构，生成 prompts，复制配置和文档。"""
    input_dir = WORKSPACE_DIR / "input"
    prompts_dir = WORKSPACE_DIR / "prompts"
    input_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "output").mkdir(exist_ok=True)
    (WORKSPACE_DIR / "logs").mkdir(exist_ok=True)
    (WORKSPACE_DIR / "cache").mkdir(exist_ok=True)
    (WORKSPACE_DIR / "lancedb").mkdir(exist_ok=True)

    # 复制 settings.yaml
    settings_src = SCRIPT_DIR / "settings.yaml"
    settings_dst = WORKSPACE_DIR / "settings.yaml"
    if not settings_dst.exists():
        shutil.copy(settings_src, settings_dst)
        print(f"  settings.yaml → {settings_dst}")

    # 生成 prompts（从 graphrag 内置默认值）
    _write_default_prompts(prompts_dir)

    # 复制文档到 input/
    md_files = list(DOCS_DIR.glob("*.md"))
    existing = set(f.name for f in input_dir.glob("*.md"))
    copied = 0
    for f in md_files:
        if f.name not in existing:
            shutil.copy(f, input_dir / f.name)
            copied += 1
    print(f"  文档: {len(md_files)} 个（新增 {copied} 个）→ {input_dir}")


def _write_default_prompts(prompts_dir: Path):
    """从 graphrag 内置模块写出默认 prompt 文件。"""
    from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
    from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
    from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
    from graphrag.prompts.index.community_report_text_units import COMMUNITY_REPORT_TEXT_PROMPT
    from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_knowledge_system_prompt import GENERAL_KNOWLEDGE_INSTRUCTION
    from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT
    from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT

    prompt_files = {
        "extract_graph.txt": GRAPH_EXTRACTION_PROMPT,
        "summarize_descriptions.txt": SUMMARIZE_PROMPT,
        "community_report_graph.txt": COMMUNITY_REPORT_PROMPT,
        "community_report_text.txt": COMMUNITY_REPORT_TEXT_PROMPT,  # noqa: F821
        "local_search_system_prompt.txt": LOCAL_SEARCH_SYSTEM_PROMPT,
        "global_search_map_system_prompt.txt": MAP_SYSTEM_PROMPT,
        "global_search_reduce_system_prompt.txt": REDUCE_SYSTEM_PROMPT,
        "global_search_knowledge_system_prompt.txt": GENERAL_KNOWLEDGE_INSTRUCTION,
        "drift_search_system_prompt.txt": DRIFT_LOCAL_SYSTEM_PROMPT,
        "drift_search_reduce_prompt.txt": DRIFT_REDUCE_PROMPT,
        "basic_search_system_prompt.txt": BASIC_SEARCH_SYSTEM_PROMPT,
    }

    written = 0
    for filename, content in prompt_files.items():
        path = prompts_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            written += 1
    print(f"  prompts: {written} 个写出到 {prompts_dir}")


# ── 知识图谱构建 ───────────────────────────────────────────────────────────────

def build_index() -> float:
    """运行 graphrag index，返回耗时秒数。"""
    print("\n== 开始构建知识图谱 ==")
    print(f"  工作区: {WORKSPACE_DIR}")
    print("  (GLM-4-flash API 调用量较大，预计 30-60 分钟)")

    # 设置环境变量供 settings.yaml 的 ${VAR} 引用
    env = os.environ.copy()
    env["GRAPHRAG_LLM_API_KEY"] = LLM_API_KEY
    env["GRAPHRAG_LLM_BASE_URL"] = LLM_BASE_URL
    env["GRAPHRAG_EMBEDDING_API_KEY"] = os.environ.get("GRAPHRAG_EMBEDDING_API_KEY", "")
    env["GRAPHRAG_EMBEDDING_BASE_URL"] = os.environ.get("GRAPHRAG_EMBEDDING_BASE_URL", "")

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "graphrag", "index", "--root", str(WORKSPACE_DIR)],
        env=env,
        capture_output=False,  # 实时输出到终端，方便观察进度
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] graphrag index 失败 (returncode={result.returncode})")
        print("检查 data/workspace/logs/ 目录查看详细错误")
        sys.exit(1)

    print(f"\n知识图谱构建完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return elapsed


def check_index_exists() -> bool:
    """检查索引是否完整（需要 lancedb 向量库）。"""
    output_dir = WORKSPACE_DIR / "output"
    lancedb_dir = WORKSPACE_DIR / "lancedb"
    parquet_files = list(output_dir.rglob("*.parquet"))
    lancedb_tables = list(lancedb_dir.rglob("*.lance")) if lancedb_dir.exists() else []
    return len(parquet_files) > 0 and len(lancedb_tables) > 0


# ── 查询 ───────────────────────────────────────────────────────────────────────

def query_graphrag(question: str) -> tuple[str, float]:
    """
    调用 graphrag query --method local，返回 (answer, latency_ms)。
    local 模式：实体聚焦，适合单跳和多跳问题。
    """
    env = os.environ.copy()
    env["GRAPHRAG_LLM_API_KEY"] = LLM_API_KEY
    env["GRAPHRAG_LLM_BASE_URL"] = LLM_BASE_URL
    env["GRAPHRAG_EMBEDDING_API_KEY"] = os.environ.get("GRAPHRAG_EMBEDDING_API_KEY", "")
    env["GRAPHRAG_EMBEDDING_BASE_URL"] = os.environ.get("GRAPHRAG_EMBEDDING_BASE_URL", "")

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable, "-m", "graphrag", "query",
            "--root", str(WORKSPACE_DIR),
            "--method", "local",
            "--response-type", "Single Paragraph",
            question,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        return "", latency_ms

    # graphrag query 的输出格式：
    # SUCCESS: Global Search Response:
    # <answer text>
    output = result.stdout.strip()
    lines = output.split("\n")
    # 跳过 "SUCCESS:" 行和空行，取后续内容
    answer_lines = []
    capture = False
    for line in lines:
        if "SUCCESS" in line or "Response:" in line:
            capture = True
            continue
        if capture:
            answer_lines.append(line)
    answer = "\n".join(answer_lines).strip()

    # 如果解析失败，直接返回原始输出
    if not answer:
        answer = output

    return answer, latency_ms


def query_all(testset: list[dict], index_time_s: float) -> list[dict]:
    """对所有题目发问，记录答案和延迟。"""
    results = []
    for i, item in enumerate(testset):
        question = item["question"]
        print(f"  [{i+1}/{len(testset)}] {question[:60]}...", end=" ", flush=True)

        try:
            answer, latency_ms = query_graphrag(question)
            print(f"✓ {latency_ms:.0f}ms")
        except subprocess.TimeoutExpired:
            answer = ""
            latency_ms = 120000.0
            print("✗ timeout")
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

    print("== GraphRAG 评测 ==")
    print(f"  工作区  : {WORKSPACE_DIR}")
    print(f"  测试集  : {TESTSET_PATH}")
    print(f"  LLM     : {LLM_MODEL} @ {LLM_BASE_URL}")
    print()

    # 初始化工作区
    print("== 初始化工作区 ==")
    init_workspace()

    # 建库
    index_time_s = 0.0
    if skip_index:
        if check_index_exists():
            print("跳过建库（使用已有索引）")
        else:
            print("未找到索引，强制重新建库...")
            index_time_s = build_index()
    else:
        index_time_s = build_index()

    # 加载测试集
    testset = [json.loads(l) for l in TESTSET_PATH.read_text().splitlines() if l.strip()]
    print(f"\n== 开始检索（{len(testset)} 题）==")
    results = query_all(testset, index_time_s)

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
        "framework": "GraphRAG",
        "version": "3.1.1",
        "query_mode": "local",
        "llm_model": LLM_MODEL,
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
