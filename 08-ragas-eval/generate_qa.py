"""
使用 LLM 自动生成测试集（QA 对）
基于知识库文档，让模型生成问题和参考答案
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_pipeline import load_documents

DATA_PATH = "./data/knowledge_base.json"
OUTPUT_PATH = "./data/generated_testset.json"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个测试数据生成专家。请基于给定的技术文档，生成1个高质量的问答对。\n"
               "要求：\n"
               "1. 问题应覆盖文档的核心知识点\n"
               "2. 答案应简洁准确，基于文档内容\n"
               "3. 输出严格的 JSON 格式，不要包含其他内容\n"
               '输出格式：{"question": "...", "ground_truth": "...", "relevant_doc_ids": ["doc-xxx"]}'),
    ("human", "文档标题：{title}\n文档内容：{content}\n文档ID：{doc_id}\n\n请生成问答对："),
])


def generate_qa_for_doc(doc, llm) -> dict:
    """为单个文档生成 QA 对"""
    chain = GENERATE_PROMPT | llm | JsonOutputParser()
    try:
        result = chain.invoke({
            "title": doc.metadata["title"],
            "content": doc.page_content[:800],
            "doc_id": doc.metadata["doc_id"],
        })
        if "relevant_doc_ids" not in result:
            result["relevant_doc_ids"] = [doc.metadata["doc_id"]]
        return result
    except Exception as e:
        print(f"生成失败 [{doc.metadata['doc_id']}]: {e}")
        return None


def generate_testset(output_path: str = OUTPUT_PATH):
    """基于知识库自动生成测试集"""
    print("=" * 50)
    print("开始生成测试集（LLM-based）")
    print("=" * 50)

    docs = load_documents(DATA_PATH)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.3,
    )

    testset = []
    for doc in docs:
        print(f"生成中... {doc.metadata['doc_id']} - {doc.metadata['title']}")
        qa = generate_qa_for_doc(doc, llm)
        if qa:
            testset.append(qa)
            print(f"  ✓ {qa['question'][:50]}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 测试集生成完成，共 {len(testset)} 条，保存至 {output_path}")
    return testset


if __name__ == "__main__":
    generate_testset()
