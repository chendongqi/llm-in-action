"""
RAG Pipeline 实现
支持可配置的 chunk_size, chunk_overlap, top_k 等参数
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


DATA_PATH = "./data/knowledge_base.json"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术问答助手。请根据提供的参考资料回答问题。"
               "如果参考资料中没有相关信息，请明确说明。回答要简洁准确。"),
    ("human", "参考资料：\n{context}\n\n问题：{question}\n\n请回答："),
])


def load_documents(path: str) -> list[Document]:
    """加载知识库文档"""
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
    """可配置的 RAG Pipeline"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 4,
        temperature: float = 0.1,
        persist_dir: str = CHROMA_PATH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def build_index(self, docs: list[Document] = None, force_rebuild: bool = False):
        """构建或加载向量索引"""
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
                docs = load_documents(DATA_PATH)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", " ", ""],
            )
            chunks = splitter.split_documents(docs)
            print(f"[RAG] 文档切分为 {len(chunks)} 个 chunk (size={self.chunk_size}, overlap={self.chunk_overlap})")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_dir,
            )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k},
        )

        llm = build_llm(self.temperature)

        def format_docs(docs):
            return "\n\n---\n\n".join([d.page_content for d in docs])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        return self

    def query(self, question: str) -> dict:
        """执行 RAG 查询，返回答案和检索上下文"""
        if self.chain is None:
            raise RuntimeError("请先调用 build_index() 构建索引")

        contexts = self.retriever.invoke(question)
        answer = self.chain.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }

    def query_batch(self, questions: list[str]) -> list[dict]:
        """批量执行 RAG 查询"""
        results = []
        for q in questions:
            results.append(self.query(q))
        return results


if __name__ == "__main__":
    # 快速测试
    pipeline = RAGPipeline(chunk_size=512, chunk_overlap=50, top_k=4)
    pipeline.build_index(force_rebuild=True)

    test_q = "什么是 RAG 技术？"
    result = pipeline.query(test_q)
    print(f"\n问题: {test_q}")
    print(f"答案: {result['answer'][:200]}...")
    print(f"检索到 {len(result['contexts'])} 条上下文")
