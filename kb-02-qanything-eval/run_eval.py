"""
QAnything 评测脚本

1. 建库：创建知识库，上传 30 个文档，等待处理完成
2. 检索：对 testset.jsonl 里的 89 道题逐一问答
3. 评分：边界拒答率 + P90 延迟 + 关键词匹配率

输出：
  data/results/answers.jsonl   - 每题的答案和延迟
  data/results/eval_report.json - 汇总指标
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# ── 配置 ──────────────────────────────────────────────────────────────────────

DOCS_DIR = SCRIPT_DIR.parent / "kb-00-testset/data/raw_docs"
TESTSET_PATH = SCRIPT_DIR.parent / "kb-00-testset/data/output/testset.jsonl"
RESULTS_DIR = SCRIPT_DIR / "data/results"

BASE_URL = os.getenv("QANYTHING_BASE_URL", "http://localhost:8777")
USER_ID = os.getenv("QANYTHING_USER_ID", "zzp__1234")
KB_NAME = "enterprise_kb_eval"

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── API 封装 ──────────────────────────────────────────────────────────────────

def api_post(endpoint: str, data: dict = None, files=None) -> dict:
    url = f"{BASE_URL}/api/local_doc_qa/{endpoint}"
    if files:
        resp = requests.post(url, data=data, files=files, timeout=120)
    else:
        resp = requests.post(url, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def create_kb() -> str:
    """创建知识库，返回 kb_id"""
    # 先检查是否已有同名知识库
    result = api_post("list_knowledge_base", {"user_id": USER_ID})
    for kb in result.get("data", []):
        if kb["kb_name"] == KB_NAME:
            print(f"已有知识库: {kb['kb_id']}")
            return kb["kb_id"]

    result = api_post("new_knowledge_base", {
        "user_id": USER_ID,
        "kb_name": KB_NAME,
        "kb_info": "Enterprise KB evaluation dataset"
    })
    kb_id = result["data"]["kb_id"]
    print(f"创建知识库: {kb_id}")
    return kb_id


def upload_docs(kb_id: str) -> list[str]:
    """上传所有文档，返回 file_id 列表"""
    md_files = sorted(DOCS_DIR.glob("*.md"))
    file_ids = []

    # 先检查已上传的文件
    existing = api_post("list_files", {"user_id": USER_ID, "kb_id": kb_id, "page_id": 1, "page_limit": 100})
    existing_names = {d["file_name"] for d in existing.get("data", {}).get("details", [])}
    print(f"已有 {len(existing_names)} 个文件")

    for i, f in enumerate(md_files):
        if f.name in existing_names:
            print(f"  [{i+1}/{len(md_files)}] {f.name} (已存在，跳过)")
            continue
        with open(f, "rb") as fp:
            result = api_post("upload_files",
                data={"user_id": USER_ID, "kb_id": kb_id},
                files={"files": (f.name, fp, "text/markdown")}
            )
        fid = result["data"][0]["file_id"]
        file_ids.append(fid)
        print(f"  [{i+1}/{len(md_files)}] {f.name} → {fid}")
        time.sleep(0.3)  # 避免上传过快

    return file_ids


def wait_for_processing(kb_id: str, timeout: int = 600):
    """等待所有文件处理完成（status=green）"""
    print("\n等待文件处理完成...")
    start = time.time()
    while time.time() - start < timeout:
        result = api_post("list_files", {"user_id": USER_ID, "kb_id": kb_id, "page_id": 1, "page_limit": 100})
        status_count = result["data"]["status_count"]
        total = result["data"]["total"]
        green = status_count.get("green", 0)
        gray = status_count.get("gray", 0)
        red = status_count.get("red", 0)

        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] 总计:{total} 完成:{green} 等待:{gray} 失败:{red}")

        if gray == 0:
            print(f"所有文件处理完成！完成:{green} 失败:{red}")
            return True
        time.sleep(15)

    print("超时！部分文件可能未处理完成")
    return False


def query_all(kb_id: str, testset: list[dict]) -> list[dict]:
    """对所有题目发问，记录答案和延迟"""
    results = []
    for i, item in enumerate(testset):
        question = item["question"]
        print(f"  [{i+1}/{len(testset)}] {question[:60]}...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            result = api_post("local_doc_chat", {
                "user_id": USER_ID,
                "kb_ids": [kb_id],
                "question": question,
                "model": LLM_MODEL,
                "api_base": LLM_BASE_URL,
                "api_key": LLM_API_KEY,
                "max_token": 512,
                "networking": False,
                "history": [],
                "streaming": False,
            })
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = result.get("response", "").replace("data: [DONE]\n\n", "").strip()
            # history 里有完整答案
            if not answer and result.get("history"):
                answer = result["history"][-1][-1] if result["history"] else ""
            source_count = len(result.get("source_documents", []))
            print(f"✓ {latency_ms:.0f}ms (sources:{source_count})")
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = ""
            source_count = 0
            print(f"✗ {e}")

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "question_type": item.get("question_type", "unknown"),
            "source_docs": item.get("source_docs", []),
            "source_count": source_count,
            "latency_ms": latency_ms,
        })

    return results


# ── 评分 ──────────────────────────────────────────────────────────────────────

def compute_boundary_refusal_rate(results: list[dict]) -> float:
    REFUSAL_KEYWORDS = [
        "don't have", "do not have", "no information", "not contain",
        "not available", "cannot find", "not mentioned", "doesn't contain",
        "does not contain", "no relevant", "not found", "cannot provide",
        "unable to find", "抱歉", "无法回答", "没有相关",
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

    print("== QAnything 评测 ==")
    print(f"  服务地址 : {BASE_URL}")
    print(f"  文档目录 : {DOCS_DIR}")
    print(f"  测试集   : {TESTSET_PATH}")
    print(f"  LLM      : {LLM_MODEL} @ {LLM_BASE_URL}")
    print()

    # 建库 & 上传文档
    kb_id = create_kb()

    if not skip_index:
        upload_docs(kb_id)
        wait_for_processing(kb_id)
    else:
        print("跳过上传（使用已有知识库）")

    # 加载测试集
    testset = [json.loads(l) for l in TESTSET_PATH.read_text().splitlines() if l.strip()]
    print(f"\n开始检索（{len(testset)} 题）...")
    results = query_all(kb_id, testset)

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
        "framework": "QAnything",
        "version": "v2.0",
        "llm_model": LLM_MODEL,
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
    print(f"\n  完整报告: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true", help="跳过上传文档，直接评测")
    args = parser.parse_args()
    main(skip_index=args.skip_index)
