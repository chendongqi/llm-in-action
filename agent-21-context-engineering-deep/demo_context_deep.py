"""
Context Engineering Deep Dive

Demonstrates three context management strategies for long-running agents,
with quantified trade-offs in token usage, latency, and recall quality.

Three strategies:
  1. Naive        — pass all history, no management
  2. Sliding Window — keep last N messages (truncation)
  3. Rolling Summary — LLM-compressed summary + recent window

Run:
    conda activate dev_base
    python demo_context_deep.py
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ["LLM_API_KEY"],  # type: ignore[arg-type]
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.1,
)

SYSTEM_PROMPT = (
    "You are a project management assistant for a software engineering team. "
    "Help the team recall and reason about decisions, technical choices, "
    "and action items discussed in previous meetings."
)

WIDTH = 70


# ── Token estimation ─────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Conservative estimate: ~3 chars per token for Chinese+English mix."""
    return max(1, len(text) // 3)


def count_messages_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        total += count_tokens(content) + 4  # per-message overhead
    return total


# ── Synthetic conversation (30 turns, key decisions spread across history) ───

def make_history() -> list:
    """
    30-turn project discussion.
    Decisions are intentionally scattered: early, middle, and recent turns.
    Test queries will target EARLY decisions to stress context recall.
    """
    turns = [
        # ── Turns 1-10: early, foundational decisions ─────────────────────
        (
            "What database should we use for the new analytics service?",
            "Decision: PostgreSQL with TimescaleDB extension. Reasons: ACID "
            "compliance, mature ecosystem, and TimescaleDB handles time-series "
            "queries with automatic partitioning. Owner: David (DB Lead).",
        ),
        (
            "What's the caching strategy?",
            "Decision: Redis Cluster for high availability. TTL policy: "
            "1 hour for user session data, 5 minutes for real-time analytics "
            "dashboards. Maximum memory: 16 GB per node.",
        ),
        (
            "Who owns the database migration process?",
            "Decision: Sarah (Backend Lead) owns all database migrations. "
            "Requirement: at least 2 senior-engineer approvals before any "
            "production migration runs. Use Flyway for versioned migrations.",
        ),
        (
            "What deployment platform are we targeting?",
            "Decision: Kubernetes on AWS EKS. Use Helm charts for deployment "
            "management. Production: 3-node cluster (m5.xlarge). "
            "Staging: 1-node cluster. CD via ArgoCD.",
        ),
        (
            "What's the API versioning strategy?",
            "Decision: URL path versioning (/api/v1/, /api/v2/). Maintain "
            "backwards compatibility for 2 major versions. Deprecation notice "
            "period: 6 months. Breaking changes require RFC approval.",
        ),
        (
            "How should we handle authentication?",
            "Decision: JWT tokens. Access token TTL: 24 hours for regular "
            "users, 1 hour for admin operations. Refresh tokens in Redis "
            "with 30-day TTL. Signing: HMAC-SHA256.",
        ),
        (
            "What's the logging and observability stack?",
            "Decision: Structured JSON logging with correlation IDs. "
            "Pipeline: Fluentd → ELK (Elasticsearch + Logstash + Kibana). "
            "Retention: 30 days hot, 1 year cold in S3 Glacier.",
        ),
        (
            "How do we implement rate limiting?",
            "Decision: Token bucket algorithm. Limits: 100 req/min for "
            "standard users, 1000 req/min for premium tier. "
            "State stored in Redis for distributed enforcement.",
        ),
        (
            "What's the CI/CD pipeline setup?",
            "Decision: GitHub Actions for CI, ArgoCD for CD. Gates: all "
            "tests pass, 80% code coverage, Snyk security scan clean. "
            "Deployment strategy: blue-green with 5-minute health check.",
        ),
        (
            "gRPC or REST for internal services?",
            "Decision: REST for all external-facing APIs; gRPC + Protocol "
            "Buffers for internal service-to-service communication. "
            "Rationale: REST is easier to consume externally, gRPC is faster "
            "for high-frequency internal calls.",
        ),
        # ── Turns 11-20: middle decisions ─────────────────────────────────
        (
            "What's the data retention policy?",
            "Policy: User PII — 3 years active, 7 years archived. "
            "Analytics events — 2 years active. Application logs — 30 days. "
            "GDPR right-to-deletion must be fulfilled within 30 days.",
        ),
        (
            "How do we handle database backups?",
            "Decision: Daily full backups at 02:00 UTC, hourly incremental. "
            "Storage: S3 with 90-day retention. Restore test: monthly. "
            "RTO target: 4 hours. RPO target: 1 hour.",
        ),
        (
            "What monitoring and alerting should we configure?",
            "Stack: Prometheus + Grafana for metrics, PagerDuty for on-call. "
            "SLO targets: 99.9% availability, P95 latency < 500 ms, "
            "error rate < 0.1%. Alert fatigue rule: no P3 alerts on weekends.",
        ),
        (
            "How do we structure the microservices?",
            "Domain-Driven Design: User Service, Analytics Service, "
            "Notification Service, API Gateway. Each service owns its own "
            "data store (no shared databases). Async communication via "
            "Kafka events for cross-service workflows.",
        ),
        (
            "What's the feature flag strategy?",
            "Decision: LaunchDarkly. Rollout plan: 1% → 10% → 50% → 100%. "
            "Kill switch required for all production features. "
            "Internal beta flags separate from external rollout flags.",
        ),
        (
            "Let's finalize sprint planning.",
            "Decision: 2-week sprints. Story point scale: Fibonacci "
            "(1, 2, 3, 5, 8, 13). Velocity target: 40 points/sprint for "
            "the 6-person team. Sprint review every other Friday at 3 PM.",
        ),
        (
            "What's the code review policy?",
            "Policy: All PRs require 2 approvals. Security-sensitive code "
            "additionally requires security team sign-off. Performance-critical "
            "paths need benchmark data attached. SLA: 24-hour review turnaround.",
        ),
        (
            "How do we handle incidents?",
            "Severity definitions: SEV-1 (production down) — immediate "
            "response, 15-min status updates, post-mortem required within 5 "
            "business days. SEV-2 — 1-hour response. SEV-3 — next business day.",
        ),
        (
            "What testing strategy are we adopting?",
            "Testing pyramid: 70% unit, 20% integration, 10% E2E. "
            "Required coverage: 80% for business logic modules. "
            "Property-based testing for all data transformation code (Hypothesis).",
        ),
        (
            "What are the documentation requirements?",
            "All public APIs: OpenAPI 3.0 specs required. Architecture "
            "decisions: recorded as ADRs in /docs/adr/. Runbooks for every "
            "production operation. Wiki updated within 1 sprint of feature launch.",
        ),
        # ── Turns 21-30: recent decisions ─────────────────────────────────
        (
            "What's the security scanning approach?",
            "SAST: SonarQube on every PR. DAST: OWASP ZAP weekly on staging. "
            "Dependency scanning: Snyk daily. Penetration test: quarterly. "
            "Critical CVEs must be patched within 24 hours.",
        ),
        (
            "How should we handle configuration management?",
            "Secrets: AWS Parameter Store (SSM). App config: Kubernetes "
            "ConfigMaps per environment. Zero secrets in git. "
            "Credential rotation: every 90 days, automated via AWS Secrets Manager.",
        ),
        (
            "What's the error handling standard?",
            "Use Result<T, E> pattern for domain errors (no raw exceptions "
            "in business logic). HTTP status codes per RFC 7231. "
            "Every error response includes correlation_id. "
            "All 5xx errors trigger automatic PagerDuty alert.",
        ),
        (
            "Let's finalize the distributed tracing setup.",
            "Decision: OpenTelemetry for instrumentation (language-agnostic). "
            "Backend: Jaeger. Sampling: 100% of errors, 10% of successful "
            "requests. Trace IDs propagated in X-Trace-ID header.",
        ),
        (
            "What naming conventions are we enforcing?",
            "Python: snake_case. JavaScript/TypeScript: camelCase. "
            "REST endpoints: plural nouns (/users, /orders). "
            "Database tables: snake_case singular. "
            "Event names: SCREAMING_SNAKE_CASE past tense (USER_CREATED).",
        ),
        (
            "How should we approach database query optimization?",
            "Mandatory EXPLAIN ANALYZE for any query touching > 10k rows. "
            "Index review required in every migration PR. "
            "Query timeout: 30 seconds hard limit. "
            "N+1 detection via Bullet (Rails) or SQLAlchemy instrumentation.",
        ),
        (
            "What's the service mesh decision?",
            "Decision: Istio on EKS. Mutual TLS between all services by "
            "default. Traffic policies: circuit breaker (50% error rate "
            "threshold), retry (3 attempts, 250ms backoff).",
        ),
        (
            "How do we handle async job processing?",
            "Decision: Celery + Redis as broker for Python services. "
            "Job priority: 5 levels (CRITICAL to LOW). Dead-letter queue "
            "for failed jobs with 3-retry policy. Dashboard: Flower.",
        ),
        (
            "What's the container image strategy?",
            "Base images: distroless for production, slim for development. "
            "Image scanning: Trivy in CI on every build. "
            "Registry: AWS ECR. Tag policy: semantic version + git SHA. "
            "Max image age in ECR: 90 days for untagged images.",
        ),
        (
            "Final check: what's the on-call rotation?",
            "Weekly rotation, 5-person pool. Primary on-call owns SEV-1 "
            "response. Secondary shadows for the first month. "
            "On-call compensation: 0.5x day rate per week. "
            "Post-incident 48-hour recovery period guaranteed.",
        ),
    ]

    messages = []
    for human, ai in turns:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


# ── Quality measurement ───────────────────────────────────────────────────────

def recall_score(response: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the response (case-insensitive)."""
    text = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in text)
    return found / len(keywords)


# ── Strategy 1: Naive ────────────────────────────────────────────────────────

class StrategyResult(NamedTuple):
    response: str
    tokens: int
    latency: float
    recall: float


def run_naive(history: list, query: str, keywords: list[str]) -> StrategyResult:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)]
    tokens = count_messages_tokens(msgs)
    t0 = time.time()
    text = str(llm.invoke(msgs).content)
    return StrategyResult(text, tokens, time.time() - t0, recall_score(text, keywords))


# ── Strategy 2: Sliding Window ────────────────────────────────────────────────

def run_sliding_window(
    history: list, query: str, keywords: list[str], window: int = 12
) -> StrategyResult:
    recent = history[-window:]
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + recent + [HumanMessage(content=query)]
    tokens = count_messages_tokens(msgs)
    t0 = time.time()
    text = str(llm.invoke(msgs).content)
    return StrategyResult(text, tokens, time.time() - t0, recall_score(text, keywords))


# ── Strategy 3: Rolling Summary ───────────────────────────────────────────────

def summarize(messages: list) -> str:
    """Compress a block of conversation into bullet-point decisions."""
    text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )
    prompt = (
        "Compress the following project discussion into concise bullet points.\n"
        "Preserve: every decision made, owner names, technical choices, exact numbers.\n"
        "Remove: conversational filler, redundancy.\n\n"
        f"Conversation:\n{text}\n\n"
        "Bullet-point summary:"
    )
    return str(llm.invoke([HumanMessage(content=prompt)]).content)


def run_rolling_summary(
    history: list,
    query: str,
    keywords: list[str],
    recent_window: int = 8,
    cached_summary: str | None = None,
) -> tuple[StrategyResult, str]:
    """
    Returns (StrategyResult, summary_text).
    Pass cached_summary to avoid recomputing for subsequent queries.
    """
    if len(history) > recent_window:
        old = history[:-recent_window]
        recent = history[-recent_window:]
        summary = cached_summary if cached_summary is not None else summarize(old)
    else:
        recent = history
        summary = ""

    sys = SYSTEM_PROMPT
    if summary:
        sys += f"\n\n## Earlier Meeting Notes (Summary)\n{summary}"

    msgs = [SystemMessage(content=sys)] + recent + [HumanMessage(content=query)]
    tokens = count_messages_tokens(msgs)
    t0 = time.time()
    text = str(llm.invoke(msgs).content)
    return StrategyResult(text, tokens, time.time() - t0, recall_score(text, keywords)), summary


# ── Main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Context Engineering Deep Dive")
    print("Strategy comparison: Naive vs Sliding Window vs Rolling Summary")
    print("=" * WIDTH)

    history = make_history()
    n_turns = len(history) // 2
    full_tokens = count_messages_tokens([SystemMessage(content=SYSTEM_PROMPT)] + history)
    print(f"\nHistory: {n_turns} turns  |  Full context: ~{full_tokens:,} estimated tokens")

    # ── Build rolling summary once (reuse across queries) ───────────────────
    print("\nBuilding rolling summary of early conversation... ", end="", flush=True)
    t_sum = time.time()
    cached_summary = summarize(history[:-8])
    print(f"done ({time.time() - t_sum:.1f}s)")

    # ── Test queries (all targeting EARLY decisions to stress recall) ────────
    test_cases = [
        {
            "label": "DB decision (turn 1)",
            "query": "What database did we choose for the analytics service? "
                     "Who owns it, and why did we pick that database?",
            "keywords": ["postgresql", "timescaledb", "david", "acid", "time-series"],
        },
        {
            "label": "Cache config (turn 2)",
            "query": "What's our caching technology and TTL configuration?",
            "keywords": ["redis", "cluster", "1 hour", "5 minute", "16"],
        },
        {
            "label": "Migration ownership (turn 3)",
            "query": "Who is responsible for database migrations, and what approvals are needed?",
            "keywords": ["sarah", "backend lead", "2", "senior", "flyway"],
        },
        {
            "label": "Deployment platform (turn 4)",
            "query": "What deployment platform and cluster configuration did we decide on?",
            "keywords": ["kubernetes", "eks", "helm", "argocd", "3-node"],
        },
    ]

    all_results: list[dict] = []

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * WIDTH}")
        print(f"Query {i}/{len(test_cases)}  [{tc['label']}]")
        print(f"Q: {tc['query']}")
        print(f"{'─' * WIDTH}")

        row: dict = {"label": tc["label"]}

        # Strategy 1
        print("  [1] Naive           ", end="", flush=True)
        r1 = run_naive(history, tc["query"], tc["keywords"])
        print(f"tokens={r1.tokens:>5,}  latency={r1.latency:.1f}s  recall={r1.recall:.0%}")
        row["naive"] = r1

        # Strategy 2
        print("  [2] Sliding Window  ", end="", flush=True)
        r2 = run_sliding_window(history, tc["query"], tc["keywords"], window=12)
        print(f"tokens={r2.tokens:>5,}  latency={r2.latency:.1f}s  recall={r2.recall:.0%}")
        row["sliding"] = r2

        # Strategy 3 (reuse cached summary)
        print("  [3] Rolling Summary ", end="", flush=True)
        r3, _ = run_rolling_summary(history, tc["query"], tc["keywords"],
                                    recent_window=8, cached_summary=cached_summary)
        print(f"tokens={r3.tokens:>5,}  latency={r3.latency:.1f}s  recall={r3.recall:.0%}")
        row["rolling"] = r3

        all_results.append(row)

    # ── Aggregate table ───────────────────────────────────────────────────────
    print("\n" + "=" * WIDTH)
    print("Aggregate Results (avg across all queries)")
    print("=" * WIDTH)

    strategies = ["naive", "sliding", "rolling"]
    labels     = ["Naive (full history)", "Sliding Window (last 12)", "Rolling Summary"]

    avg: dict[str, dict] = {}
    for s in strategies:
        avg[s] = {
            "tokens":  sum(r[s].tokens  for r in all_results) / len(all_results),
            "latency": sum(r[s].latency for r in all_results) / len(all_results),
            "recall":  sum(r[s].recall  for r in all_results) / len(all_results),
        }

    print(f"\n  {'Strategy':<28} {'Avg Tokens':>10} {'Avg Latency':>12} {'Avg Recall':>11}")
    print("  " + "─" * 65)
    for s, label in zip(strategies, labels):
        a = avg[s]
        print(f"  {label:<28} {a['tokens']:>10,.0f} {a['latency']:>11.1f}s {a['recall']:>10.0%}")

    print(f"\n  Token reduction vs Naive:")
    for s, label in zip(strategies[1:], labels[1:]):
        pct = (1 - avg[s]["tokens"] / avg["naive"]["tokens"]) * 100
        print(f"    {label}: -{pct:.0f}%")

    # ── Per-query recall detail ───────────────────────────────────────────────
    print(f"\n  Recall by query:")
    print(f"  {'Query':<30} {'Naive':>7} {'Sliding':>9} {'Rolling':>9}")
    print("  " + "─" * 58)
    for row in all_results:
        print(
            f"  {row['label']:<30} "
            f"{row['naive'].recall:>7.0%} "
            f"{row['sliding'].recall:>9.0%} "
            f"{row['rolling'].recall:>9.0%}"
        )

    # ── Insights ──────────────────────────────────────────────────────────────
    best_recall    = max(strategies, key=lambda s: avg[s]["recall"])
    most_efficient = min(strategies, key=lambda s: avg[s]["tokens"])
    best_balanced  = max(
        strategies,
        key=lambda s: avg[s]["recall"] / (avg[s]["tokens"] / avg["naive"]["tokens"])
    )

    print(f"\n  Key insights:")
    print(f"    Highest recall:     {labels[strategies.index(best_recall)]}")
    print(f"    Most token-efficient: {labels[strategies.index(most_efficient)]}")
    print(f"    Best quality/cost:  {labels[strategies.index(best_balanced)]}")
    print("=" * WIDTH + "\n")

    # ── Show rolling summary ──────────────────────────────────────────────────
    print("Rolling Summary (built from turns 1–22):")
    print("─" * WIDTH)
    print(cached_summary)
    print("─" * WIDTH + "\n")


if __name__ == "__main__":
    main()
