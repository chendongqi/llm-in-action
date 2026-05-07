"""
RAG Pipeline（支持自定义 Prompt）
用于第 9 篇：故意制造不同问题，观察 RAGAS 指标变化
"""

import json
import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_PATH = "../08-ragas-eval/data/knowledge_base.json"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# ── 3 种 Prompt 配置 ────────────────────────────────────────────────────────

# 基准 Prompt：正常、严格基于上下文
PROMPT_BASELINE = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请严格根据提供的参考资料回答问题。"
               "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
])

# 问题 Prompt 2：诱导幻觉——鼓励模型超出上下文发挥
PROMPT_HALLUCINATION = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的百科全书式 AI 助手。请基于你丰富的知识储备全面回答问题。"
               "下面的参考资料仅供参考，你可以在此基础上扩展更多相关知识，不必局限于参考资料内容。"
               "尽量补充背景知识和延伸信息，让回答更加丰富。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请给出全面详细的回答："),
])

# 问题 Prompt 3：诱导偏题——要求输出学术综述格式
PROMPT_OFFTOPIC = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深技术研究员，负责撰写学术综述。"
               "针对用户的问题，请按照以下固定结构回答：\n"
               "1. 技术背景与历史演进\n"
               "2. 主要技术流派与对比分析\n"
               "3. 当前挑战与未来发展趋势\n"
               "回答需要学术化，涵盖广泛，每部分至少 200 字。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请撰写综述报告："),
])

PROMPT_MAP = {
    "baseline": PROMPT_BASELINE,
    "hallucination": PROMPT_HALLUCINATION,
    "offtopic": PROMPT_OFFTOPIC,
}


def load_documents(path: str = DATA_PATH) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        content = f"标题：{item['title']}\n{item['content']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "doc_id": item["id"],
                "title": item["title"],
                "category": item["category"],
                "tags": ",".join(item["tags"]),
            }
        ))
    return docs


def build_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_API_BASE,
        chunk_size=32,
    )


def build_llm(temperature: float = 0.1):
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


class RAGPipeline:
    """可配置的 RAG Pipeline，支持自定义 Prompt"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 4,
        temperature: float = 0.1,
        persist_dir: str = "./chroma_db",
        prompt_type: str = "baseline",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.persist_dir = persist_dir
        self.prompt_type = prompt_type
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def build_index(self, docs: list[Document] = None, force_rebuild: bool = False):
        if force_rebuild and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        embeddings = build_embeddings()

        if os.path.exists(self.persist_dir) and not force_rebuild:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
        else:
            if docs is None:
                docs = load_documents()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", " ", ""],
            )
            chunks = splitter.split_documents(docs)
            print(f"  [索引] chunk_size={self.chunk_size}, overlap={self.chunk_overlap} → {len(chunks)} 个 chunk")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_dir,
            )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k},
        )

        prompt = PROMPT_MAP.get(self.prompt_type, PROMPT_BASELINE)
        llm = build_llm(self.temperature)

        def format_docs(docs):
            return "\n\n---\n\n".join([d.page_content for d in docs])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return self

    def query(self, question: str) -> dict:
        if self.chain is None:
            raise RuntimeError("请先调用 build_index() 构建索引")
        contexts = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
