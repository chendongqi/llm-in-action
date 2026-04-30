"""
RAG Pipeline —— LangChain 1.x + 多Provider LLM + SiliconFlow Embedding
系列文章第二篇完整可运行代码

支持的 LLM Provider（通过 .env 配置）：
  - zhipu  : 智谱 AI（默认）
  - openai : OpenAI / SiliconFlow 等 OpenAI 兼容 API
  - ollama : 本地 Ollama
  - azure  : Azure OpenAI Service

支持的 Embedding Provider：
  - openai : SiliconFlow / OpenAI 等 OpenAI 兼容 API（默认）
  - ollama : 本地 Ollama embedding

用法：
    1. 复制 .env.example 为 .env，填入真实 API Key
    2. 将 PDF 文件放入 data/ 目录
    3. 运行：python rag_pipeline.py
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ─── 加载 .env 配置 ───────────────────────────────────────────────────
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_ZHIPU_THINKING = os.getenv("LLM_ZHIPU_THINKING", "false").lower() == "true"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
DATA_DIRECTORY = "./data"


# ─── LLM 工厂 ─────────────────────────────────────────────────────────
def build_llm(temperature: float = 0) -> ChatOpenAI:
    """根据 LLM_PROVIDER 构建 LLM 实例"""

    if not LLM_API_KEY:
        raise ValueError(
            f"[Error] LLM_PROVIDER={LLM_PROVIDER} 但未设置 LLM_API_KEY。\n"
            f"请在 .env 文件中配置。"
        )

    # OpenAI 兼容模式（OpenAI / SiliconFlow / Ollama / 智谱 等）
    if LLM_PROVIDER in ("openai", "zhipu", "ollama"):
        base_url = LLM_API_BASE or _get_default_base_url(LLM_PROVIDER)
        extra_headers = _get_extra_headers(LLM_PROVIDER)

        print(f"[LLM] Provider: {LLM_PROVIDER} | Model: {LLM_MODEL} | Base: {base_url}")

        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=base_url,
            extra_headers=extra_headers or None,
            temperature=temperature,
        )

    # Azure OpenAI 模式
    elif LLM_PROVIDER == "azure":
        azure_version = os.getenv("LLM_AZURE_API_VERSION", "2024-08-01-preview")
        print(f"[LLM] Provider: Azure | Model: {LLM_MODEL} | API Version: {azure_version}")

        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            azure_endpoint=LLM_API_BASE,
            api_version=azure_version,
            temperature=temperature,
        )

    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}，可选值：zhipu, openai, ollama, azure")


def _get_default_base_url(provider: str) -> str:
    defaults = {
        "zhipu":  "https://open.bigmodel.cn/api/paas/v4",
        "openai": "https://api.openai.com/v1",
        "ollama": "http://localhost:11434/v1",
    }
    return defaults.get(provider, "")


def _get_extra_headers(provider: str) -> dict | None:
    """某些 Provider 需要额外的 HTTP Header"""
    if provider == "zhipu":
        return {
            "HTTP-Referer": "https://github.com/example/rag-pipeline",
            "X-Title":      "RAG Pipeline Demo",
        }
    return None


# ─── Embedding 工厂 ──────────────────────────────────────────────────
def build_embeddings() -> OpenAIEmbeddings:
    """根据 EMBEDDING_PROVIDER 构建 Embedding 实例"""

    if not EMBEDDING_API_KEY:
        raise ValueError(
            "[Error] 未设置 EMBEDDING_API_KEY。\n"
            "请在 .env 文件中配置（使用 SiliconFlow 或 OpenAI）。"
        )

    print(f"[Embedding] Provider: {EMBEDDING_PROVIDER} | Model: {EMBEDDING_MODEL} | Base: {EMBEDDING_API_BASE}")

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        dimensions=EMBEDDING_DIMENSIONS,
        chunk_size=32,  # SiliconFlow 限制最大批次 32
    )


# ─── 辅助函数 ─────────────────────────────────────────────────────────
def format_docs(docs: list) -> str:
    """把 Document 列表格式化成字符串，塞进 Prompt 的 {context}"""
    return "\n\n".join(doc.page_content for doc in docs)


# ─── Step 1: 加载文档 ───────────────────────────────────────────────
def load_pdfs(data_dir: str = DATA_DIRECTORY):
    """从数据目录加载所有 PDF 文件"""
    documents = []
    pdf_paths = list(Path(data_dir).glob("*.pdf"))

    if not pdf_paths:
        print(f"警告：在 '{data_dir}' 目录下没有找到 PDF 文件")
        return documents

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        documents.extend(pages)
        print(f"已加载 '{pdf_path.name}'：{len(pages)} 页")

    print(f"共加载 {len(documents)} 个文档片段")
    return documents


# ─── Step 2: 切分文档 ───────────────────────────────────────────────
def split_documents(documents, chunk_size=200, chunk_overlap=30):
    """将文档切分为有重叠的块，chunk_overlap 确保相邻块之间有上下文连续性"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个块（chunk_size={chunk_size}，overlap={chunk_overlap}）")
    return chunks


# ─── Step 3: Embedding 并存入 ChromaDB ──────────────────────────────
def build_vector_store(chunks):
    """创建 Embedding 并存入 ChromaDB"""
    embeddings = build_embeddings()

    # 重新建立索引前清除旧数据
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"已清除旧向量库：{PERSIST_DIRECTORY}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"向量库构建完成：{vector_store._collection.count()} 个向量已持久化")
    return vector_store


# ─── Step 4: 构建 Retriever ─────────────────────────────────────────
def get_retriever(vector_store, search_k=4):
    """配置相似度检索的 Retriever"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k}
    )
    return retriever


# ─── Step 5: 构建 RAG Chain（LCEL 方式）─────────────────────────────
def build_rag_chain(retriever):
    """
    构建完整 RAG Chain（langchain 1.x 兼容写法，不依赖 create_retrieval_chain）
    检索 → format_docs → 塞进 Prompt → LLM → StrOutputParser
    """
    llm = build_llm(temperature=0)

    # System Prompt，{context} 由 format_docs 填充，{question} 是用户原始问题
    system_prompt = (
        "你是一个精准的知识助手。请仅根据下方提供的参考内容回答用户问题。"
        "如果参考内容中没有答案，请明确说明——不要编造。\n\n"
        "参考内容：\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # LCEL Chain：先用 retriever 检索文档，再用 RunnablePassthrough 透传原始问题，
    # 最后 format_docs 把 Document 对象列表转成字符串，填入 {context}
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ─── Step 6: 查询 Pipeline ──────────────────────────────────────────
def query(rag_chain, question: str, retriever):
    """把问题丢进 RAG Pipeline 跑一遍，打印答案和来源"""
    print(f"\n{'='*50}")
    print(f"问题：{question}")
    print(f"{'='*50}")

    answer = rag_chain.invoke(question)
    print(f"\n答案：\n{answer}")

    # 单独检索一次来源（rag_chain 不直接暴露 retrieved docs）
    docs = retriever.invoke(question)
    print("\n检索到的来源：")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {source}（第 {page} 页）：{preview}...")

    return answer


# ─── 主程序 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("RAG Pipeline 启动")
    print(f"  LLM Provider    : {LLM_PROVIDER}")
    print(f"  LLM Model       : {LLM_MODEL}")
    print(f"  Embedding       : {EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}")
    print(f"  数据目录        : {DATA_DIRECTORY}")
    print(f"  向量库          : {PERSIST_DIRECTORY}")
    print("=" * 50)

    # 1. 加载
    docs = load_pdfs(DATA_DIRECTORY)
    if not docs:
        print(f"\n请在 {DATA_DIRECTORY}/ 目录下放入 PDF 文件后再运行")
        exit(0)

    # 2. 切分
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=30)

    # 3. Embedding & 存储
    vector_store = build_vector_store(chunks)

    # 4. 检索器
    retriever = get_retriever(vector_store, search_k=4)

    # 5. 构建 Chain（LCEL 方式，langchain 1.x 兼容）
    rag_chain = build_rag_chain(retriever)

    print("\n" + "=" * 50)
    print("RAG Pipeline 构建完成！输入问题开始问答（输入 'quit' 退出）")
    print("=" * 50)

    # 6. 交互式提问
    while True:
        try:
            user_input = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit", "q", "退出"):
            print("再见！")
            break

        if user_input:
            query(rag_chain, user_input, retriever)
