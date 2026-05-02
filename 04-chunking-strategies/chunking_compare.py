"""
RAG 系列第四篇：4 种分块策略对比实验
分别用固定大小、递归字符、语义分块、文档结构分块处理同一份 Markdown 文档，
输出统计对比和可视化结果。

用法：
    python chunking_compare.py

输出：
    - 终端：统计表格 + 每种策略的前 3 个块示例
    - 文件：report.md（完整对比报告）
"""

import os
import re
from pathlib import Path
from collections import Counter
from statistics import mean, median

from dotenv import load_dotenv
load_dotenv()

# ─── LangChain 分块器 ─────────────────────────────────────────────────
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

# SemanticChunker 是可选的（需要 Embedding API）
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from typing import List

    class FilteredSemanticChunker(SemanticChunker):
        """过滤空字符串，避免 SiliconFlow Embedding API 返回 400"""
        def _get_single_sentences_list(self, text: str) -> List[str]:
            import re
            sentences = re.split(self.sentence_split_regex, text)
            return [s for s in sentences if s.strip()]

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# ─── 配置 ──────────────────────────────────────────────────────────────
DATA_PATH = "./data/sample-tech-doc.md"
REPORT_PATH = "./report.md"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_markdown(path: str) -> str:
    """加载 Markdown 文件内容"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_markdown_as_documents(path: str) -> list[Document]:
    """把 Markdown 按行加载为 Document 列表（保留元数据）"""
    text = load_markdown(path)
    return [Document(page_content=text, metadata={"source": path})]


def print_separator(title: str):
    """打印带标题的分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_chunks(chunks: list[Document]) -> dict:
    """统计块的基本信息"""
    lengths = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(lengths),
        "avg_len": round(mean(lengths), 1) if lengths else 0,
        "median_len": round(median(lengths), 1) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "min_len": min(lengths) if lengths else 0,
        "lengths": lengths,
    }


def print_stats(name: str, stats: dict):
    """打印统计信息"""
    print(f"\n📊 [{name}]")
    print(f"   块数:        {stats['count']}")
    print(f"   总字符数:    {stats['total_chars']}")
    print(f"   平均长度:    {stats['avg_len']} 字符")
    print(f"   中位数长度:  {stats['median_len']} 字符")
    print(f"   最大长度:    {stats['max_len']} 字符")
    print(f"   最小长度:    {stats['min_len']} 字符")


def print_chunk_preview(chunks: list[Document], n: int = 3):
    """打印前 n 个块的预览"""
    for i, chunk in enumerate(chunks[:n], 1):
        preview = chunk.page_content.replace("\n", " ")[:120]
        meta = chunk.metadata
        headers = meta.get("Header", "")
        if headers:
            print(f"\n   [{i}] 📌 标题层级: {headers}")
        else:
            print(f"\n   [{i}]")
        print(f"       长度: {len(chunk.page_content)} 字符")
        print(f"       内容: {preview}...")


def ascii_histogram(lengths: list[int], bins: int = 10, width: int = 40):
    """用 ASCII 字符画直方图"""
    if not lengths:
        return "(无数据)"

    min_v, max_v = min(lengths), max(lengths)
    if min_v == max_v:
        return "(所有块长度相同)"

    bin_size = (max_v - min_v) / bins
    counts = [0] * bins
    for l in lengths:
        idx = min(int((l - min_v) / bin_size), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        start = int(min_v + i * bin_size)
        end = int(min_v + (i + 1) * bin_size)
        bar = "█" * int(c / max_count * width)
        lines.append(f"  {start:>5}-{end:<5} | {bar} {c}")

    return "\n".join(lines)


def _preprocess_for_semantic(documents: list[Document], max_sentence_len: int = 180) -> list[Document]:
    """
    SemanticChunker 会对每条句子做 Embedding，而 SiliconFlow 的 BGE 模型
    限制单条输入 < 512 tokens。这里对超长句子按逗号/分号截断，避免 API 报错。
    """
    processed = []
    for doc in documents:
        text = doc.page_content
        # 按句号/问号/叹号拆分句子，但保留标点
        sentence_delims = r'(。{1,2}|？|！|\n{2,})'
        parts = re.split(sentence_delims, text)
        # 重组句子（标点归还给前一句）
        chunks = []
        i = 0
        while i < len(parts):
            s = parts[i]
            if i + 1 < len(parts) and re.match(r'^(。{1,2}|？|！|\n{2,})$', parts[i + 1]):
                s += parts[i + 1]
                i += 2
            else:
                i += 1
            chunks.append(s)

        # 对超长句子内部按逗号/分号截断
        final_sentences = []
        for chunk in chunks:
            if len(chunk) <= max_sentence_len:
                final_sentences.append(chunk)
            else:
                # 按逗号/分号/顿号拆，保留标点
                subparts = re.split(r'(，|；|、)', chunk)
                current = ""
                for part in subparts:
                    if len(current) + len(part) <= max_sentence_len:
                        current += part
                    else:
                        if current:
                            final_sentences.append(current)
                        current = part
                if current:
                    final_sentences.append(current)

        processed.append(Document(
            page_content="".join(final_sentences),
            metadata=doc.metadata.copy()
        ))
    return processed


# ═══════════════════════════════════════════════════════════════════════
# 策略 1：固定大小分块（Fixed Size）
# ═══════════════════════════════════════════════════════════════════════
def strategy_fixed_size(documents: list[Document]) -> list[Document]:
    """
    固定大小分块：按固定字符数切分，不考虑语义边界。
    像用剪刀按固定长度剪纸条，简单直接，但可能切断句子。
    """
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separator="\n",
    )
    return splitter.split_documents(documents)


# ═══════════════════════════════════════════════════════════════════════
# 策略 2：递归字符分块（Recursive Character）
# ═══════════════════════════════════════════════════════════════════════
def strategy_recursive(documents: list[Document]) -> list[Document]:
    """
    递归字符分块：按优先级尝试多种分隔符——段落、换行、句子、单词。
    尽量在语义边界处切断，是最常用的通用策略。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ═══════════════════════════════════════════════════════════════════════
# 策略 3：语义分块（Semantic Chunking）
# ═══════════════════════════════════════════════════════════════════════
def strategy_semantic(documents: list[Document]) -> list[Document] | None:
    """
    语义分块：按句子切分后，计算相邻句子的语义相似度。
    相似度低于阈值的地方就切开——确保每个块内部语义连贯。
    需要 Embedding 模型支持。
    """
    if not SEMANTIC_AVAILABLE:
        return None

    api_key = os.getenv("EMBEDDING_API_KEY", "")
    if not api_key:
        print("   ⚠️ 未设置 EMBEDDING_API_KEY，跳过语义分块")
        return None

    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key=api_key,
        base_url=os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1"),
        chunk_size=32,
    )

    # 预处理：截断超长句子，避免 Embedding API 单条长度超限
    processed_docs = _preprocess_for_semantic(documents)

    # 关键：传入中文句子拆分正则，否则 SemanticChunker 默认只按英文标点切分
    splitter = FilteredSemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
        buffer_size=0,  # 避免组合句子后超过 SiliconFlow Embedding API 的 512 token 限制
        sentence_split_regex=r"(?<=[。！？.?!])\s+",
    )
    return splitter.split_documents(processed_docs)


# ═══════════════════════════════════════════════════════════════════════
# 策略 4：文档结构分块（按 Markdown 标题层级）
# ═══════════════════════════════════════════════════════════════════════
def strategy_document_structure(text: str) -> list[Document]:
    """
    文档结构分块：按 Markdown 标题层级切分。
    每个块以标题开头，包含该标题下的所有内容，直到下一个同级或更高级标题。
    优点：保留文档结构，检索到的块自带上下文（知道属于哪个章节）。
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=False,
    )
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════════════
# 生成对比报告
# ═══════════════════════════════════════════════════════════════════════
def generate_report(results: dict) -> str:
    """生成 Markdown 格式的对比报告"""
    lines = [
        "# RAG 分块策略对比报告\n",
        "> 对比对象：同一份 Markdown 技术文档（约 12000 字符）\n",
        "> 对比维度：块数、长度分布、语义完整性、结构保留度\n",
        "---\n",
    ]

    # 统计总表
    lines.append("## 一、统计对比总表\n")
    lines.append("| 策略 | 块数 | 平均长度 | 中位数 | 最大 | 最小 |\n")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|\n")
    for name, data in results.items():
        if data is None:
            lines.append(f"| {name} | — | — | — | — | — |\n")
            continue
        s = data["stats"]
        lines.append(
            f"| {name} | {s['count']} | {s['avg_len']} | {s['median_len']} | "
            f"{s['max_len']} | {s['min_len']} |\n"
        )
    lines.append("\n")

    # 每种策略的详细分析
    for name, data in results.items():
        if data is None:
            continue
        lines.append(f"## 二、{name}\n")
        lines.append(f"**块数**：{data['stats']['count']}  |  **平均长度**：{data['stats']['avg_len']} 字符\n")
        lines.append("\n**长度分布直方图**：\n")
        lines.append("```\n")
        lines.append(ascii_histogram(data["stats"]["lengths"]))
        lines.append("\n```\n")
        lines.append("\n**前 3 个块示例**：\n")
        for i, chunk in enumerate(data["chunks"][:3], 1):
            preview = chunk.page_content.replace("\n", " ")[:200]
            headers = chunk.metadata.get("Header", "")
            if headers:
                lines.append(f"\n### 块 {i}（标题: {headers}）\n")
            else:
                lines.append(f"\n### 块 {i}\n")
            lines.append(f"```\n{preview}...\n```\n")
        lines.append("\n---\n")

    # 策略推荐
    lines.append("## 三、策略选择建议\n")
    lines.append("""
| 场景 | 推荐策略 | 理由 |
|:---|:---|:---|
| 通用技术文档 | **递归字符分块** | 兼顾语义完整性和块大小均匀性，最稳妥 |
| 结构化文档（Markdown/论文） | **文档结构分块** | 保留标题层级，检索结果自带章节上下文 |
| 需要精确语义边界的场景 | **语义分块** | 块内语义高度一致，适合专业术语密集的文档 |
| 对分块速度要求极高 | **固定大小分块** | 实现最简单，计算开销最小 |

> **实际建议**：先用递归字符分块跑通 baseline，然后根据检索质量决定是否升级到语义分块或文档结构分块。
""")

    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RAG 系列第四篇：4 种分块策略对比实验")
    print("=" * 60)

    # 加载文档
    text = load_markdown(DATA_PATH)
    documents = load_markdown_as_documents(DATA_PATH)
    print(f"\n📄 加载文档: {DATA_PATH}")
    print(f"   总字符数: {len(text)}")
    print(f"   总行数:   {text.count(chr(10))}")

    results = {}

    # ── 策略 1：固定大小 ──────────────────────────────────────────────
    print_separator("策略 1：固定大小分块（Fixed Size）")
    print("   原理：按固定字符数硬切，像剪刀剪纸条，不考虑语义边界")
    chunks = strategy_fixed_size(documents)
    stats = analyze_chunks(chunks)
    print_stats("固定大小分块", stats)
    print_chunk_preview(chunks)
    print(f"\n   📈 长度分布：\n{ascii_histogram(stats['lengths'])}")
    results["固定大小分块"] = {"chunks": chunks, "stats": stats}

    # ── 策略 2：递归字符 ──────────────────────────────────────────────
    print_separator("策略 2：递归字符分块（Recursive Character）")
    print("   原理：按优先级尝试段落→换行→句子→单词，尽量在语义边界切断")
    chunks = strategy_recursive(documents)
    stats = analyze_chunks(chunks)
    print_stats("递归字符分块", stats)
    print_chunk_preview(chunks)
    print(f"\n   📈 长度分布：\n{ascii_histogram(stats['lengths'])}")
    results["递归字符分块"] = {"chunks": chunks, "stats": stats}

    # ── 策略 3：语义分块 ──────────────────────────────────────────────
    print_separator("策略 3：语义分块（Semantic Chunking）")
    print("   原理：计算相邻句子的语义相似度，低相似度处切开")
    if SEMANTIC_AVAILABLE:
        try:
            chunks = strategy_semantic(documents)
            if chunks:
                stats = analyze_chunks(chunks)
                print_stats("语义分块", stats)
                print_chunk_preview(chunks)
                print(f"\n   📈 长度分布：\n{ascii_histogram(stats['lengths'])}")
                results["语义分块"] = {"chunks": chunks, "stats": stats}
            else:
                print("   ⚠️ 语义分块未执行（可能需要设置 EMBEDDING_API_KEY）")
                results["语义分块"] = None
        except Exception as e:
            print(f"   ⚠️ 语义分块失败: {e}")
            results["语义分块"] = None
    else:
        print("   ⚠️ SemanticChunker 不可用（请安装 langchain-experimental）")
        results["语义分块"] = None

    # ── 策略 4：文档结构 ──────────────────────────────────────────────
    print_separator("策略 4：文档结构分块（Markdown Header）")
    print("   原理：按 Markdown 标题层级切分，每个块是一个章节")
    chunks = strategy_document_structure(text)
    stats = analyze_chunks(chunks)
    print_stats("文档结构分块", stats)
    print_chunk_preview(chunks)
    print(f"\n   📈 长度分布：\n{ascii_histogram(stats['lengths'])}")
    results["文档结构分块"] = {"chunks": chunks, "stats": stats}

    # ── 生成报告 ──────────────────────────────────────────────────────
    print_separator("生成对比报告")
    report = generate_report(results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 报告已保存: {REPORT_PATH}")

    # ── 总结 ──────────────────────────────────────────────────────────
    print_separator("实验总结")
    print("""
📌 四种策略的核心差异：

   ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
   │     维度        │ 固定大小 │ 递归字符 │ 语义分块 │ 文档结构 │
   ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
   │ 语义完整性      │   ⭐⭐   │  ⭐⭐⭐  │ ⭐⭐⭐⭐⭐ │  ⭐⭐⭐  │
   │ 块大小均匀性    │ ⭐⭐⭐⭐⭐ │  ⭐⭐⭐  │  ⭐⭐⭐  │  ⭐⭐   │
   │ 计算开销        │ ⭐⭐⭐⭐⭐ │  ⭐⭐⭐  │   ⭐⭐   │  ⭐⭐⭐  │
   │ 结构保留度      │   ⭐⭐   │  ⭐⭐⭐  │  ⭐⭐⭐  │ ⭐⭐⭐⭐⭐ │
   │ 实现复杂度      │ ⭐⭐⭐⭐⭐ │  ⭐⭐⭐  │   ⭐⭐   │  ⭐⭐⭐  │
   └─────────────────┴──────────┴──────────┴──────────┴──────────┘

💡 选型建议：
   • 先跑通 → 递归字符分块（最稳妥的 baseline）
   • 要精度 → 语义分块（适合专业术语密集的文档）
   • 有结构 → 文档结构分块（Markdown/论文最佳）
   • 求速度 → 固定大小分块（实现最简单）
""")


if __name__ == "__main__":
    main()
