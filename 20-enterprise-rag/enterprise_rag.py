"""
Article 20: Enterprise RAG Architecture

Moving RAG from a single-user demo to a production service requires three things:

  1. Multi-tenancy   — different customers/departments see only their own documents
  2. Access control  — within a tenant, users only retrieve what their role permits
  3. Service layer   — caching (skip LLM on repeated questions) + rate limiting
                       (prevent abuse / cost blow-up)

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  FastAPI /query endpoint                                     │
  │    ↓ rate limit check  (reject if user over quota)          │
  │    ↓ cache lookup      (return cached answer if hit)        │
  │    ↓ tenant routing    (select correct Qdrant collection)   │
  │    ↓ permission filter (only retrieve allowed access levels) │
  │    ↓ RAG pipeline      (retrieve → generate)                │
  │    ↓ cache write       (store answer for future hits)       │
  └──────────────────────────────────────────────────────────────┘

This script runs a demo without starting an HTTP server — the service logic
is exercised directly so results are reproducible and easy to inspect.
The FastAPI app definition is included at the bottom for reference.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, VectorParams

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
embeddings = OpenAIEmbeddings(base_url=EMB_BASE_URL, api_key=EMB_API_KEY,
                              model=EMB_MODEL)

VECTOR_DIM = 1024   # bge-large-zh-v1.5 output dimension

# ─── Permission Model ─────────────────────────────────────────────────────────
#
# Documents have an access_level field in metadata.
# Roles map to the set of access levels they can see.
# At retrieval time, the allowed access levels are injected as a Qdrant filter.

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin":     ["public", "engineering_only", "hr_only", "finance_only"],
    "engineer":  ["public", "engineering_only"],
    "hr":        ["public", "hr_only"],
    "finance":   ["public", "finance_only"],
    "employee":  ["public"],
}


def allowed_levels(role: str) -> list[str]:
    return ROLE_PERMISSIONS.get(role, ["public"])


# ─── Tenant Knowledge Bases ───────────────────────────────────────────────────
#
# Two tenants: acme_corp and globex_corp.
# Each has its own documents; collections are completely isolated.

TENANT_DOCS: dict[str, list[Document]] = {
    "acme_corp": [
        Document(
            page_content=(
                "ACME Corp 是一家成立于2010年的智能制造企业，总部位于北京，"
                "员工超过5000人，主要业务涵盖工业机器人、智能传感器和工厂自动化解决方案。"
            ),
            metadata={"source": "company-intro", "access_level": "public"},
        ),
        Document(
            page_content=(
                "ACME Corp 员工福利政策（2026版）：\n"
                "- 年假：入职第一年12天，每满一年增加2天，上限20天\n"
                "- 医疗保险：公司承担80%保费，覆盖员工及直系亲属\n"
                "- 年终奖：基于绩效评级，S级3个月，A级2个月，B级1个月"
            ),
            metadata={"source": "hr-policy", "access_level": "hr_only"},
        ),
        Document(
            page_content=(
                "ACME Corp 机器人控制系统技术规格（内部文档）：\n"
                "- 主控芯片：ARM Cortex-A72，主频1.8GHz\n"
                "- 通信协议：EtherCAT实时总线，延迟<1ms\n"
                "- 安全等级：SIL2认证，支持紧急停机响应时间<50ms\n"
                "- 编程接口：ROS2 Humble，支持Python/C++双语言SDK"
            ),
            metadata={"source": "robot-spec", "access_level": "engineering_only"},
        ),
        Document(
            page_content=(
                "ACME Corp 2025年度财务摘要（内部保密）：\n"
                "- 总营收：42亿元，同比增长23%\n"
                "- 净利润：6.8亿元，净利润率16.2%\n"
                "- 研发投入：4.1亿元，占营收9.8%\n"
                "- 海外业务占比：31%，主要来自东南亚和欧洲市场"
            ),
            metadata={"source": "financial-report", "access_level": "finance_only"},
        ),
    ],
    "globex_corp": [
        Document(
            page_content=(
                "Globex Corp 是一家专注于云计算和SaaS服务的科技公司，"
                "成立于2015年，总部位于上海，在全球12个城市设有办事处。"
            ),
            metadata={"source": "company-intro", "access_level": "public"},
        ),
        Document(
            page_content=(
                "Globex Corp 产品线：\n"
                "- GlexCloud：企业私有云平台，支持混合云部署\n"
                "- GlexAnalytics：实时数据分析平台，支持TB级数据处理\n"
                "- GlexAI：面向企业的AI能力平台，提供NLP、CV、推荐系统服务"
            ),
            metadata={"source": "products", "access_level": "public"},
        ),
    ],
}

# ─── 1. Multi-Tenancy: One Qdrant Collection per Tenant ───────────────────────

print("=" * 70)
print("  Building Tenant Collections")
print("=" * 70)

qdrant_client = QdrantClient(":memory:")   # in-memory for demo; use host= in prod

tenant_stores: dict[str, QdrantVectorStore] = {}

for tenant_id, docs in TENANT_DOCS.items():
    # Create one Qdrant collection per tenant
    qdrant_client.create_collection(
        collection_name=tenant_id,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=tenant_id,
        embedding=embeddings,
    )
    store.add_documents(docs)
    tenant_stores[tenant_id] = store
    print(f"  [{tenant_id}] indexed {len(docs)} documents")


# ─── 2. Access-Controlled Retriever ──────────────────────────────────────────

def get_retriever(tenant_id: str, role: str, k: int = 3):
    """Return a retriever scoped to a tenant and role.

    Qdrant filter restricts results to documents whose access_level
    is in the caller's allowed set — enforced at the vector DB layer.
    """
    if tenant_id not in tenant_stores:
        raise ValueError(f"Unknown tenant: {tenant_id}")

    levels = allowed_levels(role)

    # MatchAny: document's access_level must be one of the allowed values.
    # langchain-qdrant stores metadata under the "metadata" payload key.
    access_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.access_level",
                match=MatchAny(any=levels),
            )
        ]
    )
    store: QdrantVectorStore = tenant_stores[tenant_id]
    return store.as_retriever(
        search_kwargs={"k": k, "filter": access_filter}
    )


# ─── 3. Cache: Skip LLM on Repeated Questions ────────────────────────────────

@dataclass
class CacheEntry:
    answer: str
    created_at: float = field(default_factory=time.time)


class QueryCache:
    """In-memory TTL cache keyed by (tenant_id, role, question)."""

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[tuple, CacheEntry] = {}
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _key(self, tenant_id: str, role: str, question: str) -> tuple:
        return (tenant_id, role, question.strip().lower())

    def get(self, tenant_id: str, role: str, question: str) -> Optional[str]:
        entry = self._store.get(self._key(tenant_id, role, question))
        if entry and (time.time() - entry.created_at) < self._ttl:
            self.hits += 1
            return entry.answer
        self.misses += 1
        return None

    def set(self, tenant_id: str, role: str, question: str, answer: str) -> None:
        self._store[self._key(tenant_id, role, question)] = CacheEntry(answer=answer)


cache = QueryCache(ttl_seconds=300)


# ─── 4. Rate Limiter: Per-User Sliding Window ─────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter: max_requests per window_seconds per user."""

    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._log: dict[str, list[float]] = defaultdict(list)

    def allow(self, user_id: str) -> bool:
        now = time.time()
        # Remove timestamps outside the current window
        self._log[user_id] = [t for t in self._log[user_id]
                               if now - t < self._window]
        if len(self._log[user_id]) >= self._max:
            return False
        self._log[user_id].append(now)
        return True

    def remaining(self, user_id: str) -> int:
        now = time.time()
        recent = [t for t in self._log[user_id] if now - t < self._window]
        return max(0, self._max - len(recent))


rate_limiter = RateLimiter(max_requests=5, window_seconds=60)


# ─── 5. Core Query Function ───────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个企业知识助手。根据以下参考资料回答问题，保持简洁准确。\n"
     "如果参考资料中没有足够信息，请明确说明。\n"
     "参考资料：\n{context}"),
    ("human", "{question}"),
])


@dataclass
class QueryResult:
    answer: str
    sources: list[str]
    cache_hit: bool
    rate_limited: bool
    docs_retrieved: int
    elapsed_ms: float


def query(
    tenant_id: str,
    user_id: str,
    role: str,
    question: str,
) -> QueryResult:
    """Full enterprise RAG pipeline: rate limit → cache → retrieve → generate."""

    t0 = time.perf_counter()

    # Step 1: Rate limiting
    if not rate_limiter.allow(user_id):
        return QueryResult(
            answer="请求过于频繁，请稍后重试。",
            sources=[],
            cache_hit=False,
            rate_limited=True,
            docs_retrieved=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # Step 2: Cache lookup
    cached = cache.get(tenant_id, role, question)
    if cached:
        return QueryResult(
            answer=cached,
            sources=["[from cache]"],
            cache_hit=True,
            rate_limited=False,
            docs_retrieved=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # Step 3: Tenant routing + permission-filtered retrieval
    try:
        retriever = get_retriever(tenant_id, role)
    except ValueError as e:
        return QueryResult(
            answer=f"租户不存在：{e}",
            sources=[],
            cache_hit=False,
            rate_limited=False,
            docs_retrieved=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    docs = retriever.invoke(question)

    if not docs:
        answer = "根据您的权限，未找到相关资料。"
        sources = []
    else:
        context = "\n\n".join(d.page_content for d in docs)
        answer = str(llm.invoke(
            QA_PROMPT.format_messages(context=context, question=question)
        ).content)
        sources = [d.metadata.get("source", "unknown") for d in docs]

    # Step 4: Write to cache
    cache.set(tenant_id, role, question, answer)

    return QueryResult(
        answer=answer,
        sources=sources,
        cache_hit=False,
        rate_limited=False,
        docs_retrieved=len(docs),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


# ─── Demo Scenarios ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Demo: Enterprise RAG Scenarios")
print("=" * 70)

results_log = []

def run_scenario(label, tenant_id, user_id, role, question):
    r = query(tenant_id, user_id, role, question)
    print(f"\n  ── {label} ──")
    print(f"  Tenant: {tenant_id}  User: {user_id} ({role})")
    print(f"  Q: {question}")
    print(f"  A: {r.answer[:150]}...")
    flags = []
    if r.cache_hit:     flags.append("CACHE HIT")
    if r.rate_limited:  flags.append("RATE LIMITED")
    if flags:           print(f"  ⚡ {' | '.join(flags)}")
    print(f"  Sources: {r.sources}  |  docs_retrieved={r.docs_retrieved}  |  {r.elapsed_ms:.0f}ms")
    results_log.append({
        "scenario": label,
        "tenant": tenant_id,
        "user": user_id,
        "role": role,
        "question": question,
        "answer_preview": r.answer[:200],
        "sources": r.sources,
        "cache_hit": r.cache_hit,
        "rate_limited": r.rate_limited,
        "docs_retrieved": r.docs_retrieved,
        "elapsed_ms": round(r.elapsed_ms, 1),
    })
    return r


# ── Scenario A: Normal retrieval ──────────────────────────────────────────────
r_a1 = run_scenario(
    "A1: Engineer accesses public info",
    tenant_id="acme_corp", user_id="alice", role="engineer",
    question="ACME Corp 是一家什么类型的公司？",
)

r_a2 = run_scenario(
    "A2: Engineer accesses engineering doc",
    tenant_id="acme_corp", user_id="alice", role="engineer",
    question="ACME Corp 机器人控制系统用什么通信协议？",
)

# ── Scenario B: Access control enforcement ────────────────────────────────────
r_b1 = run_scenario(
    "B1: Engineer blocked from HR doc",
    tenant_id="acme_corp", user_id="alice", role="engineer",
    question="ACME Corp 的年终奖政策是什么？",   # hr_only — engineer cannot see this
)

r_b2 = run_scenario(
    "B2: HR blocked from finance doc",
    tenant_id="acme_corp", user_id="bob", role="hr",
    question="ACME Corp 2025年净利润是多少？",   # finance_only — HR cannot see this
)

r_b3 = run_scenario(
    "B3: HR can see HR doc",
    tenant_id="acme_corp", user_id="bob", role="hr",
    question="ACME Corp 的年假天数如何计算？",   # hr_only — HR can see this
)

# ── Scenario C: Tenant isolation ──────────────────────────────────────────────
r_c1 = run_scenario(
    "C1: Globex user cannot access Acme docs",
    tenant_id="globex_corp", user_id="charlie", role="admin",
    question="ACME Corp 的员工有多少人？",       # exists only in acme_corp collection
)

r_c2 = run_scenario(
    "C2: Globex user sees their own docs",
    tenant_id="globex_corp", user_id="charlie", role="admin",
    question="Globex Corp 有哪些产品线？",
)

# ── Scenario D: Cache behavior ────────────────────────────────────────────────
print("\n  ── D: Cache behavior ──")
q_cache = "ACME Corp 是一家什么类型的公司？"   # same as A1

r_d1 = query("acme_corp", "alice", "engineer", q_cache)   # A1 already cached this
print(f"  First repeat: cache_hit={r_d1.cache_hit}  elapsed={r_d1.elapsed_ms:.0f}ms")
assert r_d1.cache_hit, "Expected cache hit"

results_log.append({
    "scenario": "D: Cache hit on repeated question",
    "tenant": "acme_corp", "user": "alice", "role": "engineer",
    "question": q_cache,
    "cache_hit": r_d1.cache_hit,
    "elapsed_ms": round(r_d1.elapsed_ms, 1),
})

# ── Scenario E: Rate limiting ─────────────────────────────────────────────────
print("\n  ── E: Rate limiting (max 5 req/min) ──")
# Reset rate limiter for a fresh user so we can hit the limit cleanly
test_user = "rate_test_user"
allowed_count = 0
blocked_count = 0
for i in range(7):
    r = query("acme_corp", test_user, "employee", f"测试问题{i}")
    if r.rate_limited:
        blocked_count += 1
        print(f"  Request {i+1}: RATE LIMITED")
    else:
        allowed_count += 1
        print(f"  Request {i+1}: allowed  (elapsed={r.elapsed_ms:.0f}ms)")

results_log.append({
    "scenario": "E: Rate limiting",
    "user": test_user,
    "allowed": allowed_count,
    "blocked": blocked_count,
    "rate_limit_config": "5 req/60s",
})

# ─── Cache + Rate Limit Stats ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Summary")
print("=" * 70)
print(f"  Cache hits:   {cache.hits}")
print(f"  Cache misses: {cache.misses}")
print(f"  Hit rate:     {100*cache.hits/(cache.hits+cache.misses):.0f}%")
print(f"  Requests allowed:  {allowed_count}")
print(f"  Requests blocked:  {blocked_count}")

# ─── Save Report ──────────────────────────────────────────────────────────────

report = {
    "scenarios": results_log,
    "cache_stats": {
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate_pct": round(100 * cache.hits / (cache.hits + cache.misses), 1),
    },
    "rate_limit_stats": {
        "allowed": allowed_count,
        "blocked": blocked_count,
        "config": "5 req/60s",
    },
    "access_control_summary": {
        "engineer_sees_engineering_doc": r_a2.docs_retrieved > 0,
        "engineer_blocked_from_hr": r_b1.docs_retrieved == 0,
        "hr_blocked_from_finance": r_b2.docs_retrieved == 0,
        "hr_sees_hr_doc": r_b3.docs_retrieved > 0,
        "cross_tenant_returns_no_results": r_c1.docs_retrieved == 0,
    },
}

with open("enterprise_rag_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n  Report saved to enterprise_rag_report.json")

# ─── FastAPI App (reference — not started in this demo) ───────────────────────
# Uncomment and run `uvicorn enterprise_rag:app --port 8080` to expose as HTTP API.

"""
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="Enterprise RAG Service")


class QueryRequest(BaseModel):
    tenant_id: str
    question: str


@app.post("/query")
async def query_endpoint(
    req: QueryRequest,
    x_user_id: str = Header(...),
    x_user_role: str = Header(...),
):
    result = query(
        tenant_id=req.tenant_id,
        user_id=x_user_id,
        role=x_user_role,
        question=req.question,
    )
    if result.rate_limited:
        raise HTTPException(status_code=429, detail="Too many requests")
    return {
        "answer":      result.answer,
        "sources":     result.sources,
        "cache_hit":   result.cache_hit,
        "elapsed_ms":  result.elapsed_ms,
    }
"""
