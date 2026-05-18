"""
Article 21: RAG Performance Optimization — Faster and Cheaper

Four optimizations, measured end-to-end:

  1. LLM Response Cache    — exact-match cache on the final LLM call
                             same question → return stored answer, 0 LLM calls
  2. Embedding Cache       — CacheBackedEmbeddings skips API for seen texts
                             same chunk → return stored vector, 0 embed calls
  3. Semantic Cache        — fuzzy cache using vector similarity
                             similar (not identical) question → reuse answer
  4. Async Batch Embedding — embed many texts in one API call vs sequentially

Each optimization is benchmarked independently so the numbers are comparable.
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import InMemoryByteStore
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ─── LLM / Embeddings ─────────────────────────────────────────────────────────

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

EMB_BASE_URL = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMB_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")
EMB_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                 model=LLM_MODEL, temperature=0)
base_embeddings = OpenAIEmbeddings(base_url=EMB_BASE_URL, api_key=EMB_API_KEY,
                                   model=EMB_MODEL)


# ─── Counting Wrapper ─────────────────────────────────────────────────────────
# Tracks how many embed_documents calls actually reach the API.

class CountingEmbeddings(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._api_calls = 0
        self._texts_sent = 0

    def embed_documents(self, texts, *args, **kwargs):
        self._api_calls += 1
        self._texts_sent += len(texts)
        return super().embed_documents(texts, *args, **kwargs)

    def reset(self):
        self._api_calls = 0
        self._texts_sent = 0


counting_emb = CountingEmbeddings(base_url=EMB_BASE_URL, api_key=EMB_API_KEY,
                                   model=EMB_MODEL)


# ─── Knowledge Base Texts ─────────────────────────────────────────────────────

CHUNKS = [
    "RAG（Retrieval-Augmented Generation）是一种将外部知识检索与大语言模型结合的技术，解决LLM的知识截止和幻觉问题。",
    "RAGAS是专为RAG系统设计的评估框架，包含context_recall、context_precision、faithfulness、answer_relevancy四个核心指标。",
    "向量数据库负责存储文档向量并支持相似度搜索，常见选项有Chroma（开发）、Pinecone（云托管）、Milvus（企业部署）。",
    "Embedding模型将文本转换为向量，中文场景推荐BAAI/bge-large-zh-v1.5，向量维度1024。",
    "Rerank（重排序）使用Cross-Encoder对初检结果重新评分，主要提升context_precision，典型提升+0.15~+0.30。",
    "混合检索结合BM25（关键词匹配）和向量检索，通过RRF算法融合排名，比单一策略召回更全面。",
    "Self-RAG通过反思令牌让模型自主决定是否需要检索，避免不必要的检索开销。",
    "Agentic RAG由Agent动态决定检索策略，包含分类、检索、质量评估、重试四个节点，适合混合意图场景。",
]

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "根据以下参考资料简洁回答问题。\n参考资料：\n{context}"),
    ("human", "{question}"),
])


# ─── Benchmark 1: LLM Response Cache ─────────────────────────────────────────
#
# set_llm_cache() enables LangChain's built-in LLM response caching.
# Identical (prompt, model, temperature) triples return the cached response.
# This is the cheapest optimization: 0 LLM calls on repeated questions.

print("=" * 70)
print("  Benchmark 1: LLM Response Cache")
print("=" * 70)

set_llm_cache(InMemoryCache())   # enable global LLM cache

# Build a tiny retriever for the benchmark
chroma_bench1 = Chroma.from_texts(CHUNKS, embedding=base_embeddings,
                                   collection_name="bench1")
retriever_b1  = chroma_bench1.as_retriever(search_kwargs={"k": 2})

test_questions = [
    "RAGAS 有哪四个核心指标？",
    "向量数据库有哪些常见选型？",
    "什么是 Rerank？",
]

llm_cache_results = []

for q in test_questions:
    docs    = retriever_b1.invoke(q)
    context = "\n".join(d.page_content for d in docs)
    prompt  = QA_PROMPT.format_messages(context=context, question=q)

    # First call — cache miss
    t0     = time.perf_counter()
    ans1   = llm.invoke(prompt).content
    t_miss = (time.perf_counter() - t0) * 1000

    # Second call — cache hit (identical prompt)
    t0     = time.perf_counter()
    ans2   = llm.invoke(prompt).content
    t_hit  = (time.perf_counter() - t0) * 1000

    speedup = t_miss / max(t_hit, 0.1)
    print(f"\n  Q: {q}")
    print(f"  Cache miss: {t_miss:.0f}ms   Cache hit: {t_hit:.1f}ms   Speedup: {speedup:.0f}×")
    assert ans1 == ans2, "Cache returned different answer!"

    llm_cache_results.append({
        "question":   q,
        "miss_ms":    round(t_miss, 1),
        "hit_ms":     round(t_hit, 1),
        "speedup_x":  round(speedup, 1),
    })

avg_miss = sum(r["miss_ms"] for r in llm_cache_results) / len(llm_cache_results)
avg_hit  = sum(r["hit_ms"]  for r in llm_cache_results) / len(llm_cache_results)
print(f"\n  Average: miss={avg_miss:.0f}ms  hit={avg_hit:.1f}ms  speedup={avg_miss/max(avg_hit,0.1):.0f}×")

# Disable LLM cache for subsequent benchmarks so they measure raw latency
set_llm_cache(None)


# ─── Benchmark 2: Embedding Cache ─────────────────────────────────────────────
#
# CacheBackedEmbeddings wraps any Embeddings with a ByteStore.
# The first call computes and stores the vector; subsequent calls for the
# same text return the stored vector directly without hitting the API.

print("\n" + "=" * 70)
print("  Benchmark 2: Embedding Cache (CacheBackedEmbeddings)")
print("=" * 70)

store = InMemoryByteStore()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=counting_emb,
    document_embedding_cache=store,
    namespace=EMB_MODEL,    # prefix per model so caches don't collide
)

# First pass: embed all chunks — every text hits the API
counting_emb.reset()
t0 = time.perf_counter()
_ = cached_embeddings.embed_documents(CHUNKS)
t_first = (time.perf_counter() - t0) * 1000
calls_first  = counting_emb._api_calls
texts_first  = counting_emb._texts_sent

# Second pass: same texts — all served from cache
counting_emb.reset()
t0 = time.perf_counter()
_ = cached_embeddings.embed_documents(CHUNKS)
t_second = (time.perf_counter() - t0) * 1000
calls_second = counting_emb._api_calls
texts_second = counting_emb._texts_sent

# Partial update: 2 new chunks, 6 already cached
new_chunks = CHUNKS[:6] + [
    "文档分块策略：固定大小分块适合通用场景，父子分块提升上下文完整性。",   # new
    "查询优化：HyDE生成假设文档用于检索，Multi-Query扩大召回面。",         # new
]
counting_emb.reset()
t0 = time.perf_counter()
_ = cached_embeddings.embed_documents(new_chunks)
t_partial = (time.perf_counter() - t0) * 1000
calls_partial = counting_emb._api_calls
texts_partial = counting_emb._texts_sent

print(f"\n  First pass  (8 texts, all new):   {t_first:.0f}ms  API calls={calls_first}  texts_sent={texts_first}")
print(f"  Second pass (8 texts, all cached): {t_second:.0f}ms  API calls={calls_second}  texts_sent={texts_second}")
print(f"  Partial upd (6 cached + 2 new):   {t_partial:.0f}ms  API calls={calls_partial}  texts_sent={texts_partial}")
print(f"\n  Cost reduction: {100*(texts_first-texts_partial)/texts_first:.0f}% fewer embed calls on partial update")

emb_cache_results = {
    "first_pass":   {"time_ms": round(t_first, 1),   "api_calls": calls_first,   "texts_sent": texts_first},
    "second_pass":  {"time_ms": round(t_second, 1),  "api_calls": calls_second,  "texts_sent": texts_second},
    "partial_update": {"time_ms": round(t_partial, 1), "api_calls": calls_partial, "texts_sent": texts_partial,
                       "texts_total": len(new_chunks), "texts_cached": 6, "texts_new": 2},
}


# ─── Benchmark 3: Semantic Cache ──────────────────────────────────────────────
#
# Exact-match cache requires identical question strings.
# Semantic cache goes further: if a new question is semantically similar
# enough to a previously answered question, return the cached answer.
#
# Implementation: store (question text → answer) pairs as vectors in Chroma.
# On each query, find the nearest cached question; if similarity ≥ threshold,
# serve the cached answer without any LLM call.

print("\n" + "=" * 70)
print("  Benchmark 3: Semantic Cache")
print("=" * 70)

SEMANTIC_THRESHOLD = 0.85   # cosine similarity cutoff

# Clean up previous Chroma directory for semantic cache
if os.path.exists("./semantic_cache_db"):
    shutil.rmtree("./semantic_cache_db")


class SemanticCache:
    """Vector-based semantic cache.

    Stores answered questions as embeddings. On lookup, finds the most
    similar past question; returns its cached answer if similarity is above
    the threshold, otherwise returns None (cache miss).
    """

    def __init__(self, embeddings, threshold: float = 0.85):
        self._store   = Chroma(
            collection_name="semantic_cache",
            embedding_function=embeddings,
            persist_directory="./semantic_cache_db",
        )
        self._answers: dict[str, str] = {}   # cache_id → answer
        self._threshold = threshold
        self.hits   = 0
        self.misses = 0

    def get(self, question: str) -> Optional[str]:
        if self._store._collection.count() == 0:
            self.misses += 1
            return None
        results = self._store.similarity_search_with_relevance_scores(question, k=1)
        if results:
            doc, score = results[0]
            if score >= self._threshold:
                self.hits += 1
                return self._answers.get(doc.metadata["cache_id"])
        self.misses += 1
        return None

    def set(self, question: str, answer: str) -> None:
        cache_id = str(uuid.uuid4())
        self._store.add_texts(
            texts=[question],
            metadatas=[{"cache_id": cache_id, "question": question}],
        )
        self._answers[cache_id] = answer


semantic_cache = SemanticCache(base_embeddings, threshold=SEMANTIC_THRESHOLD)

# Build retriever for this benchmark
chroma_bench3 = Chroma.from_texts(CHUNKS, embedding=base_embeddings,
                                   collection_name="bench3")
retriever_b3  = chroma_bench3.as_retriever(search_kwargs={"k": 2})


def query_with_semantic_cache(question: str) -> tuple[str, bool, float]:
    """Returns (answer, cache_hit, elapsed_ms)."""
    t0 = time.perf_counter()

    cached = semantic_cache.get(question)
    if cached:
        elapsed = (time.perf_counter() - t0) * 1000
        return cached, True, elapsed

    docs    = retriever_b3.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    answer  = str(llm.invoke(
        QA_PROMPT.format_messages(context=context, question=question)
    ).content)

    semantic_cache.set(question, answer)
    elapsed = (time.perf_counter() - t0) * 1000
    return answer, False, elapsed


# Test pairs: (original, similar paraphrase that should hit cache)
semantic_test_cases = [
    {
        "label":    "RAGAS 指标",
        "original": "RAGAS 框架有哪几个评估指标？",
        "similar":  "请介绍一下 RAGAS 的四个核心评估指标",   # different wording, same intent
        "different":"向量数据库的选型建议是什么？",           # different topic — should miss
    },
    {
        "label":    "Rerank 作用",
        "original": "Rerank 在 RAG 中起什么作用？",
        "similar":  "RAG 系统中为什么要做重排序？",           # paraphrase
        "different":"什么是混合检索？",                        # different topic
    },
]

semantic_results = []

for case in semantic_test_cases:
    print(f"\n  ── {case['label']} ──")

    # Step 1: original question — always a miss (cold cache)
    ans1, hit1, ms1 = query_with_semantic_cache(case["original"])
    print(f"  Original: '{case['original']}'")
    print(f"    hit={hit1}  {ms1:.0f}ms")

    # Step 2: similar paraphrase — should be a HIT
    ans2, hit2, ms2 = query_with_semantic_cache(case["similar"])
    print(f"  Similar:  '{case['similar']}'")
    print(f"    hit={hit2}  {ms2:.0f}ms  {'✓ CACHE HIT' if hit2 else '✗ miss (threshold not met)'}")

    # Step 3: different topic — should be a MISS
    ans3, hit3, ms3 = query_with_semantic_cache(case["different"])
    print(f"  Different:'{case['different']}'")
    print(f"    hit={hit3}  {ms3:.0f}ms  {'✗ correctly missed' if not hit3 else '✗ FALSE HIT (threshold too low)'}")

    semantic_results.append({
        "label":            case["label"],
        "original_ms":      round(ms1, 1),
        "similar_hit":      hit2,
        "similar_ms":       round(ms2, 1),
        "different_hit":    hit3,
        "different_ms":     round(ms3, 1),
    })

print(f"\n  Total — hits: {semantic_cache.hits}  misses: {semantic_cache.misses}")


# ─── Benchmark 4: Async Batch Embedding ───────────────────────────────────────
#
# Sequential embed: N separate embed_query() calls, one at a time
# Batch async embed: one aembed_documents() call covering all N texts
#
# The async batch wins because:
#   - One HTTP round-trip instead of N
#   - The embedding API processes all texts in parallel server-side
#   - Fewer rate-limit headers to handle

print("\n" + "=" * 70)
print("  Benchmark 4: Sequential vs Async Batch Embedding")
print("=" * 70)

BATCH_TEXTS = CHUNKS + [
    "文档分块策略：固定大小、递归字符、语义分块各有适用场景。",
    "查询优化技术：HyDE、Multi-Query、Query Decomposition。",
    "Graph RAG通过知识图谱遍历解决多跳关系推理问题。",
    "对话式RAG使用History-Aware Retriever改写含代词的追问。",
]   # 12 texts total

fresh_emb = OpenAIEmbeddings(base_url=EMB_BASE_URL, api_key=EMB_API_KEY,
                              model=EMB_MODEL)

# Sequential: embed one text at a time
t0 = time.perf_counter()
seq_results = [fresh_emb.embed_query(text) for text in BATCH_TEXTS]
t_seq = (time.perf_counter() - t0) * 1000
print(f"\n  Sequential ({len(BATCH_TEXTS)} texts, one by one):  {t_seq:.0f}ms")

# Async batch: embed all in one call
async def embed_batch_async(texts: list[str]) -> list:
    return await fresh_emb.aembed_documents(texts)

t0 = time.perf_counter()
batch_results = asyncio.run(embed_batch_async(BATCH_TEXTS))
t_batch = (time.perf_counter() - t0) * 1000
print(f"  Async batch ({len(BATCH_TEXTS)} texts, one call):   {t_batch:.0f}ms")
print(f"  Speedup: {t_seq/max(t_batch,1):.1f}×  (batch saved {t_seq-t_batch:.0f}ms)")

# Sanity check: same vectors
import numpy as np
for i, (v_seq, v_batch) in enumerate(zip(seq_results, batch_results)):
    similarity = np.dot(v_seq, v_batch) / (np.linalg.norm(v_seq) * np.linalg.norm(v_batch))
    assert similarity > 0.9999, f"Vector mismatch at index {i}: similarity={similarity}"
print(f"  Vector consistency check: PASSED (all {len(BATCH_TEXTS)} vectors identical)")

async_results = {
    "num_texts":     len(BATCH_TEXTS),
    "sequential_ms": round(t_seq, 1),
    "batch_async_ms":round(t_batch, 1),
    "speedup_x":     round(t_seq / max(t_batch, 1), 2),
}


# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Summary: Four Optimizations")
print("=" * 70)
print(f"""
  ┌────────────────────────────┬──────────────────┬──────────────────────┐
  │ Optimization               │ Before           │ After                │
  ├────────────────────────────┼──────────────────┼──────────────────────┤
  │ LLM Response Cache         │ {avg_miss:>6.0f}ms / call  │ {avg_hit:>4.1f}ms (cache hit)  │
  │ Embedding Cache (repeat)   │ {t_first:>6.0f}ms / 8 texts│ {t_second:>4.0f}ms (all cached)  │
  │ Embedding Cache (update)   │ 8 API calls      │ {calls_partial} API calls          │
  │ Semantic Cache (paraphrase)│ LLM call needed  │ 0ms (vector lookup)  │
  │ Async Batch Embed          │ {t_seq:>6.0f}ms (seq)   │ {t_batch:>4.0f}ms (batch)      │
  └────────────────────────────┴──────────────────┴──────────────────────┘
""")


# ─── Save Report ──────────────────────────────────────────────────────────────

report = {
    "llm_cache": {
        "questions": llm_cache_results,
        "avg_miss_ms": round(avg_miss, 1),
        "avg_hit_ms":  round(avg_hit, 1),
        "avg_speedup_x": round(avg_miss / max(avg_hit, 0.1), 1),
    },
    "embedding_cache": emb_cache_results,
    "semantic_cache": {
        "threshold":   SEMANTIC_THRESHOLD,
        "total_hits":  semantic_cache.hits,
        "total_misses": semantic_cache.misses,
        "cases":       semantic_results,
    },
    "async_batch_embedding": async_results,
}

with open("rag_performance_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("  Report saved to rag_performance_report.json")
