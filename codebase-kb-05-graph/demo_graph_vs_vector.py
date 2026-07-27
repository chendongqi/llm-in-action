"""
Codebase KB Article 05 — Vector Retrieval vs Knowledge Graph

Compares two retrieval approaches on the same codebase:

  Approach A: Pure vector retrieval
    Embed every function (AST-level chunking, raw code).
    Retrieve top-k by cosine similarity to the query.
    Baseline: Recall@5 = 0.958 (from Article 04).

  Approach B: Graph-augmented retrieval
    Step 1 — vector search: same as A, but take top-k / 2 seed functions.
    Step 2 — graph expansion: follow CALLS/CALLED_BY edges one hop,
              add neighbours to the candidate set.
    Step 3 — re-rank the expanded set by vector score, return top-k.

Goal: show that graph expansion recovers functions that are
semantically distant from the query but structurally related
(e.g. calculate_order_total is called by create_payment_intent,
so it should surface for "process payment and create Stripe charge").

The call graph is built purely with the ast module — no external deps.

Run:
    conda activate llm_base
    python demo_graph_vs_vector.py
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

MAX_CHARS = 1200


def embed_one(text: str) -> list[float]:
    clean = text[:MAX_CHARS].encode("ascii", errors="ignore").decode("ascii")
    return embedder.embed_query(clean)


# ── AST parsing: extract functions + call graph ───────────────────────────────

def parse_codebase(source: str) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Returns:
        functions  — list of {name, text, docstring}
        call_graph — {caller_name: [callee_name, ...]}
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    functions: list[dict] = []
    call_graph: dict[str, list[str]] = {}

    # First pass: collect all function names for call resolution
    all_func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_func_names.add(node.name)

    # Second pass: extract functions and their outgoing calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        text = "\n".join(lines[start:end]).strip()
        docstring = ast.get_docstring(node) or ""

        # Find all function calls within this function body
        callees: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Direct call: foo()
                if isinstance(child.func, ast.Name):
                    if child.func.id in all_func_names and child.func.id != node.name:
                        callees.append(child.func.id)
                # Attribute call: obj.foo() — skip, usually library calls

        call_graph[node.name] = list(dict.fromkeys(callees))  # dedupe, preserve order

        functions.append({
            "name": node.name,
            "text": text,
            "docstring": docstring,
        })

    return functions, call_graph


def build_reverse_graph(call_graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build CALLED_BY (reverse) edges from CALLS edges."""
    reverse: dict[str, list[str]] = {name: [] for name in call_graph}
    for caller, callees in call_graph.items():
        for callee in callees:
            if callee in reverse:
                reverse[callee].append(caller)
    return reverse


# ── Evaluation queries (same as Articles 03 & 04) ────────────────────────────

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


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(retrieved_names: list[str], relevant_names: list[str], k: int) -> float:
    top_k = retrieved_names[:k]
    hits = sum(1 for name in relevant_names if name in top_k)
    return hits / len(relevant_names) if relevant_names else 0.0


def vector_retrieve(
    query_emb: list[float],
    indexed: list[tuple[dict, list[float]]],
    k: int,
) -> list[str]:
    scored = sorted(indexed, key=lambda x: cosine_sim(query_emb, x[1]), reverse=True)
    return [func["name"] for func, _ in scored[:k]]


def expand_graph(
    seed_names: set[str],
    call_graph: dict[str, list[str]],
    called_by: dict[str, list[str]],
    hops: int = 2,
) -> set[str]:
    """BFS along CALLS and CALLED_BY edges for up to `hops` steps."""
    frontier = set(seed_names)
    visited = set(seed_names)
    for _ in range(hops):
        next_frontier: set[str] = set()
        for name in frontier:
            next_frontier.update(call_graph.get(name, []))
            next_frontier.update(called_by.get(name, []))
        new_nodes = next_frontier - visited
        visited.update(new_nodes)
        frontier = new_nodes
        if not frontier:
            break
    return visited


def graph_retrieve(
    query_emb: list[float],
    indexed: list[tuple[dict, list[float]]],
    call_graph: dict[str, list[str]],
    called_by: dict[str, list[str]],
    k: int,
    seed_k: int,
    hops: int = 2,
) -> list[str]:
    """
    1. Vector search: take top seed_k as seeds.
    2. BFS graph expansion for `hops` steps along CALLS/CALLED_BY.
    3. Re-rank expanded candidates by vector score, return top-k.
    """
    scored = sorted(indexed, key=lambda x: cosine_sim(query_emb, x[1]), reverse=True)
    seed_names = {func["name"] for func, _ in scored[:seed_k]}

    candidates = expand_graph(seed_names, call_graph, called_by, hops=hops)

    name_to_score = {func["name"]: cosine_sim(query_emb, emb) for func, emb in indexed}
    ranked = sorted(candidates, key=lambda n: name_to_score.get(n, 0.0), reverse=True)
    return ranked[:k]


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(functions: list[dict]) -> list[tuple[dict, list[float]]]:
    result = []
    for func in functions:
        emb = embed_one(func["text"])
        result.append((func, emb))
        time.sleep(0.3)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    source_file = Path(__file__).parent / "sample_codebase.py"
    source = source_file.read_text()

    print("\n" + "=" * WIDTH)
    print("Vector Retrieval vs Knowledge Graph")
    print(f"Source: {len(source.splitlines())} lines  |  Model: bge-large-zh-v1.5")
    print("=" * WIDTH)

    # Parse
    functions, call_graph = parse_codebase(source)
    called_by = build_reverse_graph(call_graph)

    print(f"\n  Parsed {len(functions)} functions")
    print(f"  Call graph edges: {sum(len(v) for v in call_graph.values())}")

    # Show call graph for payment module
    print(f"\n  {'─'*WIDTH}")
    print("  Call graph (payment module)")
    print(f"  {'─'*WIDTH}")
    payment_funcs = ["create_payment_intent", "process_refund",
                     "calculate_order_total", "verify_webhook_signature",
                     "get_payment_history"]
    for fname in payment_funcs:
        calls = call_graph.get(fname, [])
        callers = called_by.get(fname, [])
        calls_str = ", ".join(calls) if calls else "(none)"
        callers_str = ", ".join(callers) if callers else "(none)"
        print(f"  {fname}")
        print(f"    CALLS    → {calls_str}")
        print(f"    CALLED_BY← {callers_str}")

    # Build embedding index (shared between both approaches)
    print(f"\n  {'─'*WIDTH}")
    print("  Building embedding index...")
    indexed = build_index(functions)
    print(f"  Done. {len(indexed)} functions indexed.")

    # Evaluate both approaches
    print(f"\n  {'─'*WIDTH}")
    print("  Evaluating...")

    vec_r3, vec_r5 = [], []
    gra_r3, gra_r5 = [], []
    per_query = []

    for q in QUERIES:
        q_emb = embedder.embed_query(q["query"])
        time.sleep(0.2)

        # Approach A: pure vector, top-5
        vec_names = vector_retrieve(q_emb, indexed, k=5)
        v3 = recall_at_k(vec_names, q["relevant_names"], k=3)
        v5 = recall_at_k(vec_names, q["relevant_names"], k=5)
        vec_r3.append(v3)
        vec_r5.append(v5)

        # Approach B: graph-augmented, seed_k=3, final top-5
        gra_names = graph_retrieve(q_emb, indexed, call_graph, called_by, k=5, seed_k=3)
        g3 = recall_at_k(gra_names, q["relevant_names"], k=3)
        g5 = recall_at_k(gra_names, q["relevant_names"], k=5)
        gra_r3.append(g3)
        gra_r5.append(g5)

        per_query.append({
            "query": q["query"][:50],
            "relevant": q["relevant_names"],
            "vec_r5": v5,
            "gra_r5": g5,
            "vec_top5": vec_names,
            "gra_top5": gra_names,
        })

    # Summary
    print(f"\n{'='*WIDTH}")
    print("Results Summary")
    print(f"{'─'*WIDTH}")
    print(f"\n  {'Approach':<35} {'R@3':>7}  {'R@5':>7}")
    print(f"  {'─'*35} {'─'*7}  {'─'*7}")
    print(f"  {'A_vector_only':<35} {statistics.mean(vec_r3):>7.3f}  {statistics.mean(vec_r5):>7.3f}")
    print(f"  {'B_graph_augmented (seed=3)':<35} {statistics.mean(gra_r3):>7.3f}  {statistics.mean(gra_r5):>7.3f}")

    # Per-query divergence
    print(f"\n  Per-query Recall@5 (divergent cases):")
    print(f"  {'Query':<50} {'Vector':>7}  {'Graph':>7}")
    print(f"  {'─'*50} {'─'*7}  {'─'*7}")
    for pq in per_query:
        marker = " ←" if pq["vec_r5"] != pq["gra_r5"] else ""
        print(f"  {pq['query']:<50} {pq['vec_r5']:>7.2f}  {pq['gra_r5']:>7.2f}{marker}")

    # Detailed analysis of the key divergent query
    print(f"\n  {'─'*WIDTH}")
    print("  Deep dive: 'process payment and create Stripe charge'")
    print(f"  {'─'*WIDTH}")
    q8 = per_query[7]
    print(f"\n  Relevant: {q8['relevant']}")
    print(f"\n  Vector top-5:  {q8['vec_top5']}")
    print(f"  Graph top-5:   {q8['gra_top5']}")

    # Show expansion path for create_payment_intent
    print(f"\n  Graph expansion trace:")
    seeds = q8['vec_top5'][:3]
    print(f"  Seeds (vector top-3): {seeds}")
    expanded = set(seeds)
    for seed in seeds:
        calls_out = call_graph.get(seed, [])
        calls_in = called_by.get(seed, [])
        if calls_out or calls_in:
            print(f"  {seed}  CALLS→{calls_out}  CALLED_BY←{calls_in}")
            expanded.update(calls_out)
            expanded.update(calls_in)
    print(f"  Candidates after expansion: {sorted(expanded)}")

    print(f"\n{'='*WIDTH}\n")


if __name__ == "__main__":
    main()
