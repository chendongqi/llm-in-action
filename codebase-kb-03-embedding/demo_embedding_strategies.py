"""
Codebase KB Article 03 — Code Embedding Strategy Comparison

Compares three embedding strategies for code retrieval:

  Strategy A: Raw code
    Embed the function body as-is.
    Problem: the model sees a wall of code tokens with little natural language.

  Strategy B: Signature + docstring only (comment-enhanced)
    Prepend function name, parameters, and docstring before embedding.
    Hypothesis: richer natural language context improves semantic matching.

  Strategy C: Hybrid (code + comment weighted average)
    Embed code and comment text separately, then combine with weights.
    Hypothesis: preserves both implementation detail and semantic intent.

Evaluation:
  - 30 Python functions across 5 modules (sample_codebase.py)
  - 12 natural language queries with ground-truth relevant functions
  - Metric: Recall@3 and Recall@5 (fraction of relevant functions in top-k)

All three strategies use the same embedding model (BAAI/bge-large-zh-v1.5
via SiliconFlow) to isolate input format effects from model effects.

Run:
    conda activate llm_base
    python demo_embedding_strategies.py
"""

from __future__ import annotations

import ast
import math
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

WIDTH = 70

# ── Embedding model ───────────────────────────────────────────────────────────

embedder = OpenAIEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
    openai_api_key=os.environ.get("EMBEDDING_API_KEY", ""),  # type: ignore[arg-type]
    openai_api_base=os.environ.get("EMBEDDING_API_BASE", ""),  # type: ignore[call-arg]
    check_embedding_ctx_length=False,
)


# ── Parse sample codebase ─────────────────────────────────────────────────────

def parse_functions(source_file: Path) -> list[dict]:
    """Extract all functions with name, docstring, signature, and body."""
    source = source_file.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        docstring = ast.get_docstring(node) or ""
        # Reconstruct signature from args
        args = [a.arg for a in node.args.args]
        # Extract body (skip docstring line)
        start = node.lineno - 1
        end = node.end_lineno
        body_lines = lines[start:end]
        body = "\n".join(body_lines)

        functions.append({
            "name": node.name,
            "signature": f"def {node.name}({', '.join(args)})",
            "docstring": docstring,
            "body": body,
            "module": _infer_module(node.lineno, source),
        })
    return functions


def _infer_module(lineno: int, source: str) -> str:
    """Guess module name from the nearest preceding comment header."""
    lines = source.splitlines()
    for i in range(lineno - 2, -1, -1):
        line = lines[i]
        if "── Module:" in line:
            return line.split("Module:")[-1].strip().rstrip(" ─")
    return "unknown"


# ── Three input strategies ────────────────────────────────────────────────────

def strategy_a_raw_code(func: dict) -> str:
    """Strategy A: embed the raw function body."""
    return func["body"]


def strategy_b_comment_enhanced(func: dict) -> str:
    """Strategy B: signature + docstring only (natural language heavy)."""
    parts = [func["signature"]]
    if func["docstring"]:
        parts.append(f'"""{func["docstring"]}"""')
    return "\n".join(parts)


def strategy_c_hybrid(func: dict) -> str:
    """Strategy C: signature + docstring + code body (full context)."""
    parts = []
    if func["docstring"]:
        parts.append(f"# {func['docstring']}")
    parts.append(func["body"])
    return "\n".join(parts)


STRATEGIES = {
    "A_raw_code": strategy_a_raw_code,
    "B_comment_enhanced": strategy_b_comment_enhanced,
    "C_hybrid": strategy_c_hybrid,
}


# ── Evaluation queries with ground truth ──────────────────────────────────────

QUERIES = [
    {
        "query": "verify user identity and check JWT token validity",
        "relevant": ["validate_jwt_token", "verify_password"],
    },
    {
        "query": "encrypt and store user password securely",
        "relevant": ["hash_password", "verify_password"],
    },
    {
        "query": "generate JWT access token for authenticated user",
        "relevant": ["generate_access_token", "refresh_token_rotation"],
    },
    {
        "query": "check if user has permission to perform an action",
        "relevant": ["check_permission"],
    },
    {
        "query": "store and retrieve data from Redis cache",
        "relevant": ["get_cached_value", "set_cache", "cache_user_session"],
    },
    {
        "query": "limit how many times a user can call an API",
        "relevant": ["rate_limit_check"],
    },
    {
        "query": "execute SQL query safely against the database",
        "relevant": ["execute_query", "bulk_insert", "paginate_query"],
    },
    {
        "query": "process payment and create Stripe charge",
        "relevant": ["create_payment_intent", "calculate_order_total"],
    },
    {
        "query": "issue refund to customer",
        "relevant": ["process_refund"],
    },
    {
        "query": "send email notification to user",
        "relevant": ["send_email", "batch_notify_users"],
    },
    {
        "query": "send mobile push notification",
        "relevant": ["send_push_notification"],
    },
    {
        "query": "delete a record without permanently removing it from database",
        "relevant": ["soft_delete_record"],
    },
]


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant) if relevant else 0.0


def evaluate_strategy(
    strategy_name: str,
    strategy_fn,
    functions: list[dict],
    queries: list[dict],
) -> dict:
    print(f"\n  Building index for {strategy_name}...")

    # Build index: embed all functions with this strategy
    texts = [strategy_fn(f) for f in functions]
    func_names = [f["name"] for f in functions]

    # Batch embed (with rate limit handling)
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        time.sleep(0.3)

    # Evaluate each query
    recall3_scores = []
    recall5_scores = []

    for q in queries:
        q_emb = embedder.embed_query(q["query"])
        time.sleep(0.2)

        # Rank by cosine similarity
        scored = sorted(
            zip(func_names, embeddings),
            key=lambda x: cosine_sim(q_emb, x[1]),
            reverse=True,
        )
        ranked_names = [name for name, _ in scored]

        r3 = recall_at_k(ranked_names, q["relevant"], k=3)
        r5 = recall_at_k(ranked_names, q["relevant"], k=5)
        recall3_scores.append(r3)
        recall5_scores.append(r5)

    return {
        "strategy": strategy_name,
        "recall@3": round(statistics.mean(recall3_scores), 3),
        "recall@5": round(statistics.mean(recall5_scores), 3),
        "per_query": [
            {
                "query": q["query"][:50],
                "relevant": q["relevant"],
                "recall@3": round(r3, 2),
                "recall@5": round(r5, 2),
            }
            for q, r3, r5 in zip(queries, recall3_scores, recall5_scores)
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    source_file = Path(__file__).parent / "sample_codebase.py"

    print("\n" + "=" * WIDTH)
    print("Code Embedding Strategy Comparison")
    print(f"Model: {os.environ.get('EMBEDDING_MODEL', 'bge-large-zh-v1.5')}")
    print("=" * WIDTH)

    # Parse codebase
    functions = parse_functions(source_file)
    print(f"\nParsed {len(functions)} functions from sample_codebase.py")
    print(f"Modules: {sorted(set(f['module'] for f in functions))}")
    print(f"Queries: {len(QUERIES)}")

    # Evaluate all strategies
    results = []
    for name, fn in STRATEGIES.items():
        result = evaluate_strategy(name, fn, functions, QUERIES)
        results.append(result)
        print(f"  Done: Recall@3={result['recall@3']:.3f}  Recall@5={result['recall@5']:.3f}")

    # Summary table
    print(f"\n{'=' * WIDTH}")
    print("Results Summary")
    print(f"{'─' * WIDTH}")
    print(f"\n  {'Strategy':<28} {'Recall@3':>10}  {'Recall@5':>10}  {'R@5 vs A':>10}")
    print(f"  {'─'*28} {'─'*10}  {'─'*10}  {'─'*10}")

    baseline_r5 = results[0]["recall@5"]
    for r in results:
        delta = r["recall@5"] - baseline_r5
        delta_str = f"{delta:+.3f}" if delta != 0 else "  base"
        print(f"  {r['strategy']:<28} {r['recall@3']:>10.3f}  {r['recall@5']:>10.3f}  {delta_str:>10}")

    # Per-query detail for biggest difference
    print(f"\n  Per-query breakdown (Recall@5):")
    print(f"  {'Query':<50} {'A':>6}  {'B':>6}  {'C':>6}")
    print(f"  {'─'*50} {'─'*6}  {'─'*6}  {'─'*6}")
    for i, q in enumerate(results[0]["per_query"]):
        a = results[0]["per_query"][i]["recall@5"]
        b = results[1]["per_query"][i]["recall@5"]
        c = results[2]["per_query"][i]["recall@5"]
        # Highlight rows where strategies differ
        marker = " ←" if len({a, b, c}) > 1 else ""
        print(f"  {q['query']:<50} {a:>6.2f}  {b:>6.2f}  {c:>6.2f}{marker}")

    print(f"\n{'=' * WIDTH}\n")


if __name__ == "__main__":
    main()
