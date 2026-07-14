"""
AI Eval Article 04 — RAGAS Four-Metric RAG Evaluation

Demonstrates RAGAS evaluation across two knowledge base versions:
  Version A (raw):     documents ingested as-is, minimal preprocessing
  Version B (cleaned): documents with knowledge distillation applied
                       (structured Q&A extraction, noise removal, chunking optimization)

RAGAS Four Metrics:
  Faithfulness        → Does the answer stay within the retrieved context?
                        Low = hallucination risk
  Answer Relevancy    → Does the answer actually address the question?
                        Low = off-topic or vague
  Context Precision   → Of retrieved chunks, what fraction is relevant?
                        Low = retrieval is polluted with noise
  Context Recall      → Does the context contain all info needed to answer?
                        Low = retrieval is missing key information

Knowledge domain: Python development best practices
Test set: 10 questions with ground truth answers

Run:
    conda activate llm_base
    pip install ragas langchain-openai langchain-community chromadb
    python demo_ragas_eval.py
"""

from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset

load_dotenv()

WIDTH = 70

# ── LLM and Embeddings ────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.0,
)

embedder = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    openai_api_key=os.environ.get("EMBEDDING_API_KEY", ""),  # type: ignore[arg-type]
    openai_api_base=os.environ.get("EMBEDDING_API_BASE", ""),  # type: ignore[call-arg]
    check_embedding_ctx_length=False,
)

# ── Knowledge base documents ──────────────────────────────────────────────────

# Version A: raw documents (typical enterprise docs — verbose, mixed content)
RAW_DOCS = [
    Document(
        page_content="""
Python 代码规范指南 v2.3 — 2024年更新版
本文档由工程平台团队维护，适用于所有 Python 项目。

1. 命名规范
变量和函数使用 snake_case，类使用 PascalCase，常量使用 UPPER_SNAKE_CASE。
请注意旧版本文档（v1.x）中有不同的命名规范，以本文档为准。

具体规则：
- 变量名应该有意义，避免使用单字母（除了循环计数器 i, j, k）
- 函数名应该是动词开头，描述函数的行为
- 类名应该是名词，描述类的职责
- 私有变量和方法使用下划线前缀：_private_method
- 模块级私有使用双下划线：__module_private

2. 代码风格
遵循 PEP 8 标准。行宽限制 88 字符（Black 默认）。
使用 4 个空格缩进，不使用 Tab。

3. 文档字符串
所有公共函数必须有 docstring，使用 Google 风格。

附录：本规范的变更历史见 CHANGELOG.md。
联系人：platform-team@company.com
最后更新：2024-03-15
""",
        metadata={"source": "coding-standards.md", "section": "naming"}
    ),
    Document(
        page_content="""
Python 异常处理最佳实践

会议纪要 2024-01-10：工程委员会讨论了异常处理规范，达成以下共识。

一、捕获具体异常，不要裸露的 except
错误做法：
try:
    do_something()
except:  # 捕获所有，包括 SystemExit 和 KeyboardInterrupt
    pass

正确做法：
try:
    do_something()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise

二、自定义异常应该继承适当的基类
class DatabaseConnectionError(ConnectionError):
    pass

三、不要吃掉异常（silent exception）
错误做法：
try:
    result = dangerous_operation()
except Exception:
    result = None  # 用户不知道发生了什么

四、异常链（Exception Chaining）
raise NewException("message") from original_exception

五、上下文管理器用于资源清理
with open("file.txt") as f:
    data = f.read()
# 文件自动关闭，即使发生异常
""",
        metadata={"source": "exception-handling.md", "section": "best-practices"}
    ),
    Document(
        page_content="""
Python 性能优化指南

第三章：内存优化（本文档第1、2章讲基础知识，这里只展示第3章）

3.1 __slots__ 的使用
普通类为每个实例分配一个 __dict__ 字典，占用额外内存。
使用 __slots__ 可以减少内存占用：

class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

适用场景：创建大量小对象时（如数据处理中的中间对象）

3.2 生成器 vs 列表
内存密集的场景使用生成器：
# 内存友好：一次只处理一个元素
def read_large_file(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()

3.3 字符串拼接
不要在循环里用 + 拼接字符串，使用 join：
# 错误：O(n²) 内存复杂度
result = ""
for s in strings:
    result += s

# 正确：O(n)
result = "".join(strings)

作者注：本章内容来自 2023 年技术分享会，具体数据见性能测试报告 perf-report-2023.xlsx
""",
        metadata={"source": "performance-guide.md", "section": "memory"}
    ),
    Document(
        page_content="""
Python 测试最佳实践

（草稿版本，待 QA 团队审阅）

pytest 使用规范

1. 测试文件命名：test_<module_name>.py 或 <module_name>_test.py
2. 测试函数命名：test_<what_you_test>_<expected_behavior>

固定装置（Fixtures）
@pytest.fixture
def db_connection():
    conn = create_test_db()
    yield conn
    conn.close()

参数化测试
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected

Mock 使用
from unittest.mock import patch, MagicMock

@patch('module.external_api_call')
def test_with_mock(mock_api):
    mock_api.return_value = {'status': 'ok'}
    result = function_under_test()
    assert result == expected

覆盖率要求：核心业务逻辑 > 90%，工具脚本 > 70%
""",
        metadata={"source": "testing-guide.md", "section": "pytest"}
    ),
    Document(
        page_content="""
Python 日志最佳实践 — 内部培训材料

背景：2023年Q3事故复盘发现，30%的问题定位时间浪费在日志不规范上。

核心原则：
1. 使用结构化日志（JSON格式），便于日志系统解析
2. 不要使用 print()，使用 logging 模块
3. 日志级别使用规范：
   - DEBUG：详细调试信息，生产环境关闭
   - INFO：正常业务流程，关键节点
   - WARNING：可恢复的异常，不影响主流程
   - ERROR：错误，影响当前操作但程序继续运行
   - CRITICAL：严重错误，程序可能无法继续

配置示例：
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'timestamp': self.formatTime(record)
        })

logger = logging.getLogger(__name__)

禁止事项：
- 不要在日志中记录密码、token、个人信息
- 不要在循环内打 DEBUG 日志（性能影响）
""",
        metadata={"source": "logging-guide.md", "section": "best-practices"}
    ),
]

# Version B: cleaned documents (knowledge distillation applied)
# Key changes: removed meeting notes, draft markers, file references, changelogs
# Added explicit Q&A structure, cleaner chunking, focused content
CLEANED_DOCS = [
    Document(
        page_content="""Python 命名规范：
- 变量/函数：snake_case
- 类名：PascalCase
- 常量：UPPER_SNAKE_CASE
- 私有成员：下划线前缀 _private，模块私有双下划线 __private
- 函数名以动词开头描述行为，类名以名词描述职责
- 行宽限制 88 字符，4 空格缩进，遵循 PEP 8""",
        metadata={"source": "coding-standards", "type": "rule"}
    ),
    Document(
        page_content="""Python 异常处理规则：
1. 捕获具体异常类型，不用裸 except
2. 捕获后要处理或重新抛出，不要静默吃掉
3. 自定义异常继承适当基类（如 ConnectionError）
4. 异常链：raise NewException("msg") from original
5. 资源清理用 with 语句（上下文管理器）

禁止模式：
- except: pass（吃掉所有异常）
- 捕获后返回 None 而不告知调用方""",
        metadata={"source": "exception-handling", "type": "rule"}
    ),
    Document(
        page_content="""Python 内存优化方法：
1. __slots__：对大量小对象使用，用数组代替每实例 __dict__，减少内存占用
2. 生成器：处理大数据集时用 yield，一次只处理一个元素，避免全量加载
3. 字符串拼接：循环内用 "".join(list) 而非 +，后者是 O(n²) 内存复杂度""",
        metadata={"source": "performance-guide", "type": "rule"}
    ),
    Document(
        page_content="""Python 测试规范：
- 文件命名：test_<module>.py
- 函数命名：test_<功能>_<预期行为>
- Fixture 用于测试数据准备和清理，用 yield 分隔 setup/teardown
- parametrize 装饰器用于多输入多输出测试
- 外部依赖用 unittest.mock.patch 替换
- 覆盖率要求：核心业务逻辑 > 90%，工具脚本 > 70%""",
        metadata={"source": "testing-guide", "type": "rule"}
    ),
    Document(
        page_content="""Python 日志规范：
- 使用 logging 模块，不用 print()
- 生产环境用结构化 JSON 日志，便于日志系统解析
- 日志级别：DEBUG（调试）/ INFO（正常流程）/ WARNING（可恢复异常）/ ERROR（影响操作）/ CRITICAL（程序崩溃）
- 禁止在日志中记录密码、token、个人信息（PII）
- 禁止在循环内打 DEBUG 日志（性能影响）""",
        metadata={"source": "logging-guide", "type": "rule"}
    ),
]

# ── Test set with ground truth ────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "Python 中类名应该用什么命名风格？",
        "ground_truth": "类名使用 PascalCase（驼峰式），例如 MyClass。",
    },
    {
        "question": "如何正确捕获 Python 异常，应该避免什么模式？",
        "ground_truth": "应该捕获具体异常类型（如 ValueError），避免裸 except。捕获后要处理或重抛，不要静默吃掉（如 except: pass）。",
    },
    {
        "question": "Python 的 __slots__ 有什么作用，什么时候用？",
        "ground_truth": "__slots__ 用数组代替每实例的 __dict__，减少内存占用。适用于创建大量小对象的场景。",
    },
    {
        "question": "Python 循环中如何高效拼接字符串？",
        "ground_truth": "用 ''.join(list) 而不是 + 拼接，+ 在循环中是 O(n²) 内存复杂度，join 是 O(n)。",
    },
    {
        "question": "pytest 测试函数应该如何命名？",
        "ground_truth": "测试函数命名格式：test_<功能>_<预期行为>，清晰描述测试的内容和期望结果。",
    },
    {
        "question": "Python 中何时应该使用生成器而非列表？",
        "ground_truth": "处理大数据集时用生成器，yield 一次只产出一个元素，避免全量加载到内存。",
    },
    {
        "question": "Python 日志中可以记录用户密码吗？",
        "ground_truth": "不可以。禁止在日志中记录密码、token 等敏感信息和个人身份信息（PII）。",
    },
    {
        "question": "私有方法和模块私有变量在命名上有什么区别？",
        "ground_truth": "私有方法/属性用单下划线前缀 _private，模块级私有用双下划线 __private。",
    },
]


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document], collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name=collection_name,
    )


def run_rag(vectorstore: Chroma, question: str, k: int = 3) -> tuple[str, list[str]]:
    """Retrieve context and generate answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    contexts = retriever.invoke(question)
    context_texts = [doc.page_content for doc in contexts]

    prompt = f"""根据以下上下文回答问题。只使用上下文中的信息，不要添加额外内容。

上下文：
{chr(10).join(context_texts)}

问题：{question}
回答："""

    answer = str(llm.invoke(prompt).content)
    return answer, context_texts


def evaluate_rag(
    vectorstore: Chroma,
    test_cases: list[dict],
    label: str,
) -> dict:
    """Run RAG on all test cases and evaluate with RAGAS."""
    print(f"\n  Running {label} ({len(test_cases)} questions)...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, tc in enumerate(test_cases):
        answer, context = run_rag(vectorstore, tc["question"])
        questions.append(tc["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(tc["ground_truth"])
        print(f"    [{i+1}/{len(test_cases)}] {tc['question'][:40]}...")
        time.sleep(0.3)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print(f"  Computing RAGAS metrics for {label}...")
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedder)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    import statistics as _stats

    def _mean(v) -> float:
        if isinstance(v, list):
            valid = [x for x in v if x is not None and not (isinstance(x, float) and x != x)]
            return _stats.mean(valid) if valid else 0.0
        return float(v)

    scores = {
        "faithfulness": round(_mean(result["faithfulness"]), 3),
        "answer_relevancy": round(_mean(result["answer_relevancy"]), 3),
        "context_precision": round(_mean(result["context_precision"]), 3),
        "context_recall": round(_mean(result["context_recall"]), 3),
    }
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("RAGAS Four-Metric RAG Evaluation")
    print(f"Test set: {len(TEST_CASES)} questions  |  Model: glm-4-flash")
    print("=" * WIDTH)

    # Build vectorstores
    print("\nBuilding vector stores...")
    vs_raw = build_vectorstore(RAW_DOCS, "raw_docs")
    vs_clean = build_vectorstore(CLEANED_DOCS, "clean_docs")
    print("  Done.")

    # Evaluate both versions
    print(f"\n{'─' * WIDTH}")
    print("Evaluating Version A: Raw Documents")
    raw_scores = evaluate_rag(vs_raw, TEST_CASES, "Version A (raw)")

    print(f"\n{'─' * WIDTH}")
    print("Evaluating Version B: Cleaned Documents (knowledge distillation)")
    clean_scores = evaluate_rag(vs_clean, TEST_CASES, "Version B (cleaned)")

    # Results
    print(f"\n{'=' * WIDTH}")
    print("RAGAS Evaluation Results")
    print(f"{'─' * WIDTH}")
    print(f"\n  {'Metric':<24} {'Version A (raw)':>16}  {'Version B (clean)':>17}  {'Delta':>6}")
    print(f"  {'─'*24} {'─'*16}  {'─'*17}  {'─'*6}")

    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
    ]

    for label, key in metrics:
        raw_v = raw_scores[key]
        clean_v = clean_scores[key]
        delta = clean_v - raw_v
        delta_str = f"{delta:+.3f}"
        icon = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "=")
        print(f"  {label:<24} {raw_v:>16.3f}  {clean_v:>17.3f}  {delta_str:>6} {icon}")

    overall_raw = sum(raw_scores.values()) / len(raw_scores)
    overall_clean = sum(clean_scores.values()) / len(clean_scores)
    overall_delta = overall_clean - overall_raw
    print(f"  {'─'*24} {'─'*16}  {'─'*17}  {'─'*6}")
    print(f"  {'Average':<24} {overall_raw:>16.3f}  {overall_clean:>17.3f}  {overall_delta:+.3f}")

    print(f"\n  Interpretation:")
    for label, key in metrics:
        delta = clean_scores[key] - raw_scores[key]
        if abs(delta) > 0.02:
            direction = "improved" if delta > 0 else "decreased"
            print(f"    {label}: {direction} by {abs(delta):.3f} after knowledge distillation")

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
