"""
Codebase KB Article 07 — Hybrid Search: BM25 + Vector

After Articles 03-06, Q8 ("process payment and create Stripe charge")
consistently fails pure vector retrieval: calculate_order_total's semantic
distance from "Stripe charge" is too large for any embedding trick to bridge.

This article tests hybrid search (BM25 + vector) using Reciprocal Rank
Fusion (RRF) to combine two independent ranked lists.

Three approaches compared:
  A_vector:   Pure vector (AST function-level, raw code). Baseline.
  B_bm25:     Pure BM25 keyword search (TF-IDF term matching).
  C_hybrid:   RRF fusion of A and B.

BM25 is expected to rescue Q8: "calculate_order_total" body contains
"order", "discount", "tax" — and its docstring says "Sum item prices,
apply discount, compute tax." The query "process payment" contains
"payment" which is absent from calculate_order_total, but the word
"order" in the function name itself bridges the gap at keyword level.

Actually let's verify empirically, not assume.

Run:
    conda activate llm_base
    pip install rank_bm25
    python demo_hybrid_search.py
"""

from __future__ import annotations

import ast
import math
import os
import re
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

MAX_CHARS = 1200


def embed_one(text: str) -> list[float]:
    clean = text[:MAX_CHARS].encode("ascii", errors="ignore").decode("ascii")
    return embedder.embed_query(clean)


# ── Parse codebase ────────────────────────────────────────────────────────────

def parse_functions(source: str) -> list[dict]:
    tree = ast.parse(source)
    lines = source.splitlines()
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        text = "\n".join(lines[start:end]).strip()
        docstring = ast.get_docstring(node) or ""
        functions.append({
            "name": node.name,
            "text": text,
            "docstring": docstring,
        })
    return functions


# ── Tokenizer for BM25 ───────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Split code into tokens for BM25.
    Splits on non-alphanumeric, lowercases, splits camelCase / snake_case.
    """
    # First split on non-word chars
    raw_tokens = re.split(r"[^a-zA-Z0-9_]", text)
    tokens = []
    for tok in raw_tokens:
        if not tok:
            continue
        # Split snake_case
        parts = tok.split("_")
        for part in parts:
            if not part:
                continue
            # Split camelCase: insertBefore → [insert, Before]
            sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
            tokens.extend(s.lower() for s in sub if len(s) > 1)
    return tokens


# ── Evaluation queries ────────────────────────────────────────────────────────

QUERIES = [
    {"query": "verify user identity and check JWT token validity",
     "relevant_names": ["validate_jwt_token", "verify_password"]},
    {"query": "encrypt and store user password securely",
     "relevant_names": ["hash_password", "verify_password"]},
    {"query": "generate JWT access token for authenticated user",
     "relevant_names": ["generate_access_token", "refresh_token_rotation"]},
    {"query": "check if user has permission to perform an action",
     "relevant_names": ["check_permission"]},
    {"query": "store and retrieve data from Redis cache",
     "relevant_names": ["get_cached_value", "set_cache", "cache_user_session"]},
    {"query": "limit how many times a user can call an API",
     "relevant_names": ["rate_limit_check"]},
    {"query": "execute SQL query safely against the database",
     "relevant_names": ["execute_query", "bulk_insert", "paginate_query"]},
    {"query": "process payment and create Stripe charge",
     "relevant_names": ["create_payment_intent", "calculate_order_total"]},
    {"query": "issue refund to customer",
     "relevant_names": ["process_refund"]},
    {"query": "send email notification to user",
     "relevant_names": ["send_email", "batch_notify_users"]},
    {"query": "send mobile push notification",
     "relevant_names": ["send_push_notification"]},
    {"query": "delete a record without permanently removing it from database",
     "relevant_names": ["soft_delete_record"]},
]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def vector_rank(query_emb: list[float], indexed: list[tuple[str, list[float]]]) -> list[str]:
    scored = sorted(indexed, key=lambda x: cosine_sim(query_emb, x[1]), reverse=True)
    return [name for name, _ in scored]


def bm25_rank(query: str, bm25_index, func_names: list[str]) -> list[str]:
    from rank_bm25 import BM25Okapi
    q_tokens = tokenize(query)
    scores = bm25_index.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [func_names[i] for i in ranked]


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. k=60 is the standard constant."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, name in enumerate(ranked):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda n: scores[n], reverse=True)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for name in relevant if name in top_k)
    return hits / len(relevant) if relevant else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from rank_bm25 import BM25Okapi

    source_file = Path(__file__).parent / "sample_codebase.py"
    source = source_file.read_text()

    print("\n" + "=" * WIDTH)
    print("Hybrid Search: BM25 + Vector (RRF Fusion)")
    print(f"Source: {len(source.splitlines())} lines  |  Model: bge-large-zh-v1.5")
    print("=" * WIDTH)

    functions = parse_functions(source)
    func_names = [f["name"] for f in functions]
    print(f"\n  Parsed {len(functions)} functions")

    # Show tokenization for Q8 query and key functions
    print(f"\n  {'─'*WIDTH}")
    print("  BM25 tokenization preview")
    print(f"  {'─'*WIDTH}")
    q8_query = "process payment and create Stripe charge"
    print(f"\n  Query: '{q8_query}'")
    print(f"  Tokens: {tokenize(q8_query)}")
    for fname in ["calculate_order_total", "create_payment_intent", "process_checkout"]:
        func = next(f for f in functions if f["name"] == fname)
        toks = tokenize(func["text"])
        # Show which query tokens appear in func tokens
        q_toks = set(tokenize(q8_query))
        overlap = q_toks & set(toks)
        print(f"\n  {fname}:")
        print(f"    Token overlap with query: {overlap}")
        print(f"    Total tokens: {len(toks)}")

    # Build BM25 index
    corpus = [tokenize(f["text"]) for f in functions]
    bm25 = BM25Okapi(corpus)

    # Build vector index
    print(f"\n  {'─'*WIDTH}")
    print("  Building vector index...")
    vector_indexed: list[tuple[str, list[float]]] = []
    for func in functions:
        emb = embed_one(func["text"])
        vector_indexed.append((func["name"], emb))
        time.sleep(0.3)
    print(f"  Done.")

    # Evaluate all three approaches
    results: dict[str, dict] = {
        "A_vector": {"r3": [], "r5": [], "pq": []},
        "B_bm25":   {"r3": [], "r5": [], "pq": []},
        "C_hybrid": {"r3": [], "r5": [], "pq": []},
    }

    print(f"\n  {'─'*WIDTH}")
    print("  Evaluating...")

    for q in QUERIES:
        q_emb = embedder.embed_query(q["query"])
        time.sleep(0.2)

        vec_ranked  = vector_rank(q_emb, vector_indexed)
        bm25_ranked = bm25_rank(q["query"], bm25, func_names)
        hyb_ranked  = rrf_fuse([vec_ranked, bm25_ranked])

        for approach, ranked in [("A_vector", vec_ranked),
                                  ("B_bm25",   bm25_ranked),
                                  ("C_hybrid", hyb_ranked)]:
            r3 = recall_at_k(ranked, q["relevant_names"], k=3)
            r5 = recall_at_k(ranked, q["relevant_names"], k=5)
            results[approach]["r3"].append(r3)
            results[approach]["r5"].append(r5)
            results[approach]["pq"].append({
                "query": q["query"][:50],
                "r3": r3, "r5": r5,
                "top5": ranked[:5],
            })

    # Summary table
    print(f"\n{'='*WIDTH}")
    print("Results Summary")
    print(f"{'─'*WIDTH}")
    print(f"\n  {'Approach':<25} {'R@3':>7}  {'R@5':>7}  {'vs Vector':>10}")
    print(f"  {'─'*25} {'─'*7}  {'─'*7}  {'─'*10}")
    vec_r5 = statistics.mean(results["A_vector"]["r5"])
    for name, res in results.items():
        mean_r3 = statistics.mean(res["r3"])
        mean_r5 = statistics.mean(res["r5"])
        delta = mean_r5 - vec_r5
        delta_str = f"{delta:+.3f}" if abs(delta) > 1e-6 else "  base"
        print(f"  {name:<25} {mean_r3:>7.3f}  {mean_r5:>7.3f}  {delta_str:>10}")

    # Per-query divergence
    print(f"\n  Per-query Recall@5 (divergent cases):")
    print(f"  {'Query':<50} {'Vec':>6}  {'BM25':>6}  {'Hyb':>6}")
    print(f"  {'─'*50} {'─'*6}  {'─'*6}  {'─'*6}")
    pq_vec  = results["A_vector"]["pq"]
    pq_bm25 = results["B_bm25"]["pq"]
    pq_hyb  = results["C_hybrid"]["pq"]
    for i in range(len(QUERIES)):
        v = pq_vec[i]["r5"]
        b = pq_bm25[i]["r5"]
        h = pq_hyb[i]["r5"]
        marker = " ←" if len({v, b, h}) > 1 else ""
        print(f"  {pq_vec[i]['query']:<50} {v:>6.2f}  {b:>6.2f}  {h:>6.2f}{marker}")

    # Q8 deep dive
    print(f"\n  {'─'*WIDTH}")
    print("  Q8 deep dive: 'process payment and create Stripe charge'")
    print(f"  {'─'*WIDTH}")
    q8_query_str = "process payment and create Stripe charge"
    q8_tokens = tokenize(q8_query_str)
    print(f"\n  Query tokens: {q8_tokens}")
    for fname in ["calculate_order_total", "create_payment_intent"]:
        func = next(f for f in functions if f["name"] == fname)
        toks = set(tokenize(func["text"]))
        overlap = set(q8_tokens) & toks
        print(f"  {fname}: token overlap = {overlap}")
    print()
    for approach in ["A_vector", "B_bm25", "C_hybrid"]:
        print(f"  {approach:<20} top-5: {results[approach]['pq'][7]['top5']}")

    print(f"\n{'='*WIDTH}\n")


if __name__ == "__main__":
    main()
