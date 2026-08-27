"""
HyperGraphRAG 评测脚本

流程：
1. 加载 31 个文档，批量 index（记录耗时）
2. 对 testset.jsonl 里 89 道题逐一查询（每题单独计时）
3. 评分：边界拒答率 + P90 延迟 + Jaccard 答案匹配 + 索引构建时间

前置条件：
  conda activate llm_base
  pip install graspologic nano-vectordb
  cd kb-04-hypergraphrag-eval
  # 将 HyperGraphRAG 源码安装（pip install -e）或确保 hypergraphrag 可 import
  pip install -e /mnt/hdd/Database/01_Reference/OpenSource/LLM/08_KB/HyperGraphRAG

用法：
  python run_eval.py              # 完整流程（建库 + 评测）
  python run_eval.py --skip-index # 跳过建库，已有索引时直接评测
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

# ── 路径配置 ──────────────────────────────────────────────────────────────────

DOCS_DIR    = SCRIPT_DIR.parent / "kb-00-testset/data/raw_docs"
TESTSET_PATH = SCRIPT_DIR.parent / "kb-00-testset/data/output/testset.jsonl"
WORKING_DIR = str(SCRIPT_DIR / "data/workspace")
RESULTS_DIR = SCRIPT_DIR / "data/results"

# ── 环境变量 ──────────────────────────────────────────────────────────────────

LLM_API_KEY     = os.environ["LLM_API_KEY"]
LLM_BASE_URL    = os.environ["LLM_BASE_URL"]
LLM_MODEL       = os.environ.get("LLM_MODEL", "glm-4-flash")
EMBEDDING_API_KEY  = os.environ["EMBEDDING_API_KEY"]
EMBEDDING_BASE_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL    = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM      = 1024

# ── LLM & Embedding 函数 ──────────────────────────────────────────────────────

async def glm_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """GLM-4-flash via OpenAI-compatible API。keyword_extraction 时用 json 格式提示而非结构化输出。"""
    kwargs.pop("hashing_kv", None)
    if keyword_extraction or "response_format" in kwargs:
        kwargs.pop("response_format", None)
        json_hint = '\nIMPORTANT: You must respond in valid JSON format only.'
        system_prompt = (system_prompt or "") + json_hint

    # 长超时 + 手动重试，应对 GLM 偶发 500/ConnectTimeout
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=180.0, max_retries=5)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    for attempt in range(6):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 5:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [LLM retry {attempt+1}/5, wait {wait}s] {type(e).__name__}: {e}")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


import re as _re
# 匹配以下模式的整行（行内替换为占位符）：
#   1. 含 key/secret/token/password/Authorization 等关键词的行
#   2. KEY=VALUE 形式的环境变量赋值行（大写或混合大写变量名）
#   3. sk-xxxxx / Bearer xxxxx 形式的凭证
_SENSITIVE_LINE_RE = _re.compile(
    r'[^\n]*(?:'
    r'api[_\-]?key|secret|token|password|Authorization'  # 关键词
    r'|sk-[A-Za-z0-9]{10,}'                              # sk-style key
    r'|Bearer\s+\S{10,}'                                 # Bearer token
    r'|[A-Z][A-Z0-9_]{3,}=[^\s\n]+'                     # ENV_VAR=value
    r')[^\n]*',
    _re.IGNORECASE,
)

def _sanitize(text: str) -> str:
    """脱敏：把含敏感内容的行替换为占位符，避免触发 SiliconFlow 内容过滤。"""
    return _SENSITIVE_LINE_RE.sub("[REDACTED]", text)


_MAX_CHARS = 700  # bge-large-en-v1.5 via SiliconFlow 约 512 tokens；密集表格每字符约 1 token，留足余量

def _truncate(text: str) -> str:
    """脱敏 + 截断到 API 限制内。"""
    text = _sanitize(text)
    if len(text) <= _MAX_CHARS:
        return text
    # 按空格截断，保持词完整
    truncated = text[:_MAX_CHARS]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


async def bge_embedding(texts: list[str]) -> np.ndarray:
    """BGE-large-en-v1.5 via SiliconFlow OpenAI-compatible API。"""
    client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL,
                         timeout=60.0, max_retries=5)
    safe_texts = [_truncate(t) for t in texts]
    for attempt in range(6):
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=safe_texts,
                encoding_format="float",
            )
            return np.array([dp.embedding for dp in response.data])
        except Exception as e:
            if attempt == 5:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [Embed retry {attempt+1}/5, wait {wait}s] {type(e).__name__}: {e}")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")

# ── 建库 ──────────────────────────────────────────────────────────────────────

def _load_index_progress() -> dict:
    p = Path(WORKING_DIR) / "index_progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"indexed": [], "elapsed_s": 0.0}

def _save_index_progress(progress: dict) -> None:
    p = Path(WORKING_DIR) / "index_progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, ensure_ascii=False, indent=2))


def build_index(rag, doc_files: list[Path]) -> float:
    """逐文档 insert，支持从上次中断处续跑。"""
    print(f"\n== 开始构建 HyperGraphRAG 知识超图 ==")
    print(f"  文档数量   : {len(doc_files)}")
    print(f"  LLM        : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  Embedding  : {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")

    progress = _load_index_progress()
    already_done = set(progress["indexed"])
    elapsed = progress["elapsed_s"]

    remaining = [f for f in doc_files if f.name not in already_done]
    if already_done:
        print(f"  续跑：已完成 {len(already_done)} 个，剩余 {len(remaining)} 个")

    for i, fpath in enumerate(remaining, 1):
        content = fpath.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  [{i}/{len(remaining)}] {fpath.name}: 空文件，跳过")
            progress["indexed"].append(fpath.name)
            _save_index_progress(progress)
            continue

        print(f"  [{i}/{len(remaining)}] {fpath.name} ({len(content)} chars)...")
        t0 = time.time()
        rag.insert([content])
        doc_elapsed = time.time() - t0
        elapsed += doc_elapsed
        progress["indexed"].append(fpath.name)
        progress["elapsed_s"] = elapsed
        _save_index_progress(progress)
        print(f"    ✓ {doc_elapsed:.0f}s，累计 {elapsed/60:.1f} min")

    print(f"\n  建库完成，总耗时 {elapsed/60:.1f} min")
    return elapsed

# ── 单题查询 ──────────────────────────────────────────────────────────────────

def query_one(rag, item: dict) -> dict:
    from hypergraphrag import QueryParam
    t0 = time.time()
    try:
        answer = rag.query(item["question"], param=QueryParam(mode="hybrid"))
    except Exception as e:
        answer = f"[ERROR] {e}"
    latency_ms = (time.time() - t0) * 1000
    return {
        "id": item.get("id", ""),
        "question": item["question"],
        "answer": answer or "",
        "expected": item.get("answer", ""),
        "question_type": item.get("question_type", ""),
        "latency_ms": latency_ms,
    }

def query_all(rag, testset: list[dict]) -> list[dict]:
    results = []
    for i, item in enumerate(testset, 1):
        q_short = item["question"][:60].replace("\n", " ")
        res = query_one(rag, item)
        ok = "✓" if "[ERROR]" not in res["answer"] else "✗"
        print(f"  [{i}/{len(testset)}] {q_short}... {ok} {res['latency_ms']:.0f}ms")
        results.append(res)
    return results

# ── 评分 ──────────────────────────────────────────────────────────────────────

BOUNDARY_KEYWORDS = [
    "don't know", "do not know", "cannot", "can't", "not sure",
    "no information", "not found", "unable", "not available",
    "outside", "beyond", "not in", "not provided", "no relevant",
    "not mentioned", "not contain", "not include", "context does not",
]

def _jaccard(a: str, b: str) -> float:
    a_tok = set(a.lower().split())
    b_tok = set(b.lower().split())
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / len(a_tok | b_tok)

def compute_boundary_refusal_rate(results: list[dict]) -> float:
    boundary = [r for r in results if r.get("question_type") == "boundary"]
    if not boundary:
        return 0.0
    refused = sum(
        1 for r in boundary
        if any(kw in r["answer"].lower() for kw in BOUNDARY_KEYWORDS)
    )
    return refused / len(boundary)

def compute_p90_latency(results: list[dict]) -> float:
    lats = sorted(r["latency_ms"] for r in results)
    idx = int(len(lats) * 0.9)
    return lats[min(idx, len(lats) - 1)]

def compute_answer_match(results: list[dict]) -> dict:
    by_type: dict[str, list[float]] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        score = _jaccard(r["answer"], r["expected"])
        by_type.setdefault(qt, []).append(score)
    return {qt: round(sum(scores) / len(scores), 3) for qt, scores in by_type.items()}

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(skip_index: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

    print("== HyperGraphRAG 评测 ==")
    print(f"  文档目录   : {DOCS_DIR}")
    print(f"  测试集     : {TESTSET_PATH}")
    print(f"  LLM        : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  Embedding  : {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")
    print()

    from hypergraphrag import HyperGraphRAG
    from hypergraphrag.utils import EmbeddingFunc

    rag = HyperGraphRAG(
        working_dir=WORKING_DIR,
        llm_model_func=glm_complete,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=1,          # GLM-4-flash rate limit
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            concurrent_limit=1,         # SiliconFlow rate limit
            func=bge_embedding,
        ),
        embedding_batch_num=4,
    )

    print("== 初始化 HyperGraphRAG ==")
    doc_files = sorted(DOCS_DIR.glob("*.md"))
    print(f"  发现 {len(doc_files)} 个文档")

    index_time_s = 0.0
    if skip_index:
        existing = list(Path(WORKING_DIR).iterdir()) if Path(WORKING_DIR).exists() else []
        json_files = [f for f in existing if f.suffix in ('.json', '.jsonl')]
        if json_files:
            # 从进度文件读取实际耗时
            prog = _load_index_progress()
            index_time_s = prog.get("elapsed_s", 0.0)
            print(f"跳过建库（已有索引文件 {len(json_files)} 个，历史耗时 {index_time_s/60:.1f} min）")
        else:
            print("未找到索引，强制重新建库...")
            index_time_s = build_index(rag, doc_files)
    else:
        index_time_s = build_index(rag, doc_files)

    testset = [json.loads(l) for l in TESTSET_PATH.read_text().splitlines() if l.strip()]
    print(f"\n== 开始检索（{len(testset)} 题）==")
    results = query_all(rag, testset)

    answers_path = RESULTS_DIR / "answers.jsonl"
    with open(answers_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    boundary_rate = compute_boundary_refusal_rate(results)
    p90 = compute_p90_latency(results)
    match_by_type = compute_answer_match(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    report = {
        "framework": "HyperGraphRAG",
        "version": "1.0.6",
        "query_mode": "hybrid",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true", help="跳过建库，直接评测")
    args = parser.parse_args()
    main(skip_index=args.skip_index)
