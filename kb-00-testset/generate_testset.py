"""
企业知识库评测数据集生成脚本

策略：直接 LLM 调用生成三类问题，不依赖 ragas transforms（后者在本地网络环境下易超时）。
文档来源：LightRAG + graphrag 官方技术文档（31 个 Markdown 文件）

问题类型分布（参考大纲设计）：
  - 单跳事实查询（50%）：答案直接在单个文档中
  - 多跳推理（30%）：需要跨文档组合推理
  - 边界测试（20%）：知识库中无答案，测拒答能力

输出：
  data/output/testset.jsonl        - 标准格式，用于 RAGAS 评测
  data/output/testset_stats.json   - 数据集统计信息
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 无论从哪个目录运行，都以脚本文件所在目录为基准
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

DOCS_DIR = SCRIPT_DIR / "data/raw_docs"
OUTPUT_DIR = SCRIPT_DIR / "data/output"

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TESTSET_SIZE = int(os.getenv("TESTSET_SIZE", "100"))

# ── Prompts ───────────────────────────────────────────────────────────────────

SINGLE_HOP_PROMPT = """You are a technical documentation expert creating evaluation questions for a RAG system.

Given the following technical document, generate {n} single-hop factual questions.
Requirements:
- Questions must be answerable from this document alone
- Questions should cover key concepts, configurations, or procedures
- Each question should have a clear, specific answer
- Vary question types: what/how/why/when/which

Document title: {title}
Document content:
{content}

Output ONLY a JSON array with this exact format, no other text:
[
  {{
    "question": "...",
    "ground_truth": "...",
    "question_type": "single_hop"
  }}
]"""

MULTI_HOP_PROMPT = """You are a technical documentation expert creating evaluation questions for a RAG system.

Given MULTIPLE technical documents, generate {n} multi-hop reasoning questions.
Requirements:
- Questions MUST require information from at least 2 different documents to answer
- Questions should explore relationships between systems, compare approaches, or trace workflows
- Answers should synthesize information across documents

Documents:
{documents}

Output ONLY a JSON array with this exact format, no other text:
[
  {{
    "question": "...",
    "ground_truth": "...",
    "question_type": "multi_hop"
  }}
]"""

BOUNDARY_PROMPT = """You are a technical documentation expert creating evaluation questions for a RAG system.

Given the following technical documents about {topic}, generate {n} questions that CANNOT be answered from the provided documents.
Requirements:
- Questions should be plausible and relevant to the topic domain
- Questions should ask about things NOT covered in the documents
- This tests the system's ability to say "I don't know"

Example boundary topics: pricing details, internal architecture decisions, future roadmap, specific bug fixes

Document topics covered: {covered_topics}

Output ONLY a JSON array with this exact format, no other text:
[
  {{
    "question": "...",
    "ground_truth": "The provided documents do not contain information about this topic.",
    "question_type": "boundary"
  }}
]"""

# ── 文档加载 ──────────────────────────────────────────────────────────────────

def load_docs(docs_dir: Path) -> list[dict]:
    """加载 Markdown 文档"""
    docs = []
    for md_file in sorted(docs_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if len(content) < 200:
            continue
        title = md_file.stem.replace("-", " ").replace("_", " ")
        docs.append({
            "title": title,
            "filename": md_file.name,
            "content": content,
        })
    print(f"加载文档: {len(docs)} 个有效文件")
    return docs

# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, prompt: str, timeout: int = 60) -> Optional[str]:
    """带超时的 LLM 调用，失败返回 None"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return None


def parse_json_response(text: Optional[str]) -> list:
    """从 LLM 响应中解析 JSON，处理常见的格式问题"""
    if not text:
        return []
    # 提取 JSON 数组部分
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return []

# ── 问题生成 ──────────────────────────────────────────────────────────────────

def generate_single_hop(client: OpenAI, docs: list[dict], target_count: int) -> list[dict]:
    """生成单跳事实问题"""
    print(f"\n[1/3] 生成单跳问题（目标 {target_count} 题）...")
    if not docs:
        print("  警告：无可用文档，跳过单跳问题生成")
        return []
    results = []
    # 每个文档生成 2-3 题，均匀分布
    questions_per_doc = max(1, target_count // len(docs) + 1)

    for i, doc in enumerate(docs):
        if len(results) >= target_count:
            break
        remaining = target_count - len(results)
        n = min(questions_per_doc, remaining)

        content_preview = doc["content"][:3000]  # 限制长度避免超出 context
        prompt = SINGLE_HOP_PROMPT.format(
            n=n,
            title=doc["title"],
            content=content_preview,
        )

        print(f"  [{i+1}/{len(docs)}] {doc['filename']} → 生成 {n} 题...", end=" ", flush=True)
        raw = call_llm(client, prompt)
        items = parse_json_response(raw)

        for item in items:
            if item.get("question") and item.get("ground_truth"):
                item["source_docs"] = [doc["filename"]]
                results.append(item)

        print(f"✓ {len(items)} 题")
        time.sleep(0.5)  # 避免频控

    print(f"  单跳问题合计: {len(results)} 题")
    return results[:target_count]


def generate_multi_hop(client: OpenAI, docs: list[dict], target_count: int) -> list[dict]:
    """生成多跳推理问题（每次取 3-4 个相关文档）"""
    print(f"\n[2/3] 生成多跳问题（目标 {target_count} 题）...")
    results = []

    # 按主题分组：LightRAG docs vs graphrag docs
    lightrag_docs = [d for d in docs if not d["filename"].startswith("graphrag_")]
    graphrag_docs = [d for d in docs if d["filename"].startswith("graphrag_")]

    # 组合策略：同系统内跨文档 + 跨系统比较
    doc_groups = []
    # LightRAG 内部跨文档
    for i in range(0, len(lightrag_docs) - 2, 3):
        doc_groups.append(lightrag_docs[i:i+3])
    # 跨系统对比
    if lightrag_docs and graphrag_docs:
        doc_groups.append(lightrag_docs[:2] + graphrag_docs[:2])
        doc_groups.append(lightrag_docs[2:4] + graphrag_docs[1:3])

    questions_per_group = max(1, target_count // len(doc_groups) + 1)

    for i, group in enumerate(doc_groups):
        if len(results) >= target_count:
            break
        remaining = target_count - len(results)
        n = min(questions_per_group, remaining)

        doc_texts = "\n\n---\n\n".join(
            f"Document: {d['title']}\n{d['content'][:1500]}" for d in group
        )
        prompt = MULTI_HOP_PROMPT.format(n=n, documents=doc_texts)

        titles = [d["filename"] for d in group]
        print(f"  [{i+1}/{len(doc_groups)}] {titles} → 生成 {n} 题...", end=" ", flush=True)
        raw = call_llm(client, prompt, timeout=90)
        items = parse_json_response(raw)

        for item in items:
            if item.get("question") and item.get("ground_truth"):
                item["source_docs"] = [d["filename"] for d in group]
                results.append(item)

        print(f"✓ {len(items)} 题")
        time.sleep(0.5)

    print(f"  多跳问题合计: {len(results)} 题")
    return results[:target_count]


def generate_boundary(client: OpenAI, docs: list[dict], target_count: int) -> list[dict]:
    """生成边界测试问题（知识库中无答案）"""
    print(f"\n[3/3] 生成边界问题（目标 {target_count} 题）...")

    covered_topics = ", ".join(set(
        d["filename"].replace(".md", "").replace("graphrag_", "").replace("-", " ")
        for d in docs[:10]
    ))

    prompt = BOUNDARY_PROMPT.format(
        topic="RAG systems and knowledge base frameworks",
        n=target_count,
        covered_topics=covered_topics,
    )

    raw = call_llm(client, prompt, timeout=60)
    items = parse_json_response(raw)

    results = []
    for item in items:
        if item.get("question") and item.get("ground_truth"):
            item["source_docs"] = []
            results.append(item)

    print(f"  边界问题合计: {len(results)} 题")
    return results[:target_count]

# ── 主流程 ────────────────────────────────────────────────────────────────────

def generate(testset_size: int = TESTSET_SIZE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 按比例分配：单跳50% + 多跳30% + 边界20%
    n_single = int(testset_size * 0.50)
    n_multi = int(testset_size * 0.30)
    n_boundary = testset_size - n_single - n_multi

    print("== 企业知识库评测数据集生成 ==")
    print(f"  文档目录 : {DOCS_DIR}")
    print(f"  LLM      : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  目标数量 : {testset_size} 题")
    print(f"  分布     : 单跳{n_single} + 多跳{n_multi} + 边界{n_boundary}")

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    docs = load_docs(DOCS_DIR)

    single_hop = generate_single_hop(client, docs, n_single)
    multi_hop = generate_multi_hop(client, docs, n_multi)
    boundary = generate_boundary(client, docs, n_boundary)

    all_records = single_hop + multi_hop + boundary

    # ── 保存 JSONL ─────────────────────────────────────────────────────────────
    jsonl_path = OUTPUT_DIR / "testset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 保存统计 ───────────────────────────────────────────────────────────────
    stats = {
        "total": len(all_records),
        "single_hop": len(single_hop),
        "multi_hop": len(multi_hop),
        "boundary": len(boundary),
        "source_docs_count": len(docs),
        "llm_model": LLM_MODEL,
    }
    stats_path = OUTPUT_DIR / "testset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n== 生成完成 ==")
    print(f"  总计     : {len(all_records)} 题")
    print(f"  单跳     : {len(single_hop)} 题")
    print(f"  多跳     : {len(multi_hop)} 题")
    print(f"  边界     : {len(boundary)} 题")
    print(f"  输出     : {jsonl_path}")

    return all_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成 RAG 评测数据集")
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help="生成题目数量（默认100）")
    args = parser.parse_args()

    generate(testset_size=args.size)
