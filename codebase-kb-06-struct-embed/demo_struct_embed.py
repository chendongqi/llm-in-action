"""
Codebase KB Article 06 — Structure-Aware Embedding

Hypothesis from Article 05: instead of graph traversal at retrieval time,
encode structural information (called_by / calls) directly into the
embedding text. This lets the vector model "learn" the call graph at
index time, without expanding the candidate set at query time.

Three variants compared (same 12 queries, same model):

  A_raw_code:
    Embed the raw function body only.
    Baseline: Recall@5 = 0.958 (Articles 03–05 consistent).

  B_struct_prefix:
    Prepend called_by / calls as a comment before the function body.
    Example for calculate_order_total:
        # called_by: process_checkout
        # calls: (none)
        def calculate_order_total(items, discount_code=None):
            ...
    Hypothesis: the embedding now includes "process_checkout" in context,
    so a query mentioning Stripe/payment (which also mentions process_checkout)
    should pull calculate_order_total closer in vector space.

  C_struct_doc:
    Add called_by / calls into the docstring instead of a comment prefix.
    Example:
        def calculate_order_total(items, discount_code=None):
            \"\"\"Sum item prices ... Called by: process_checkout. Calls: (none).\"\"\"
            ...

Run:
    conda activate llm_base
    python demo_struct_embed.py
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


# ── Parse codebase: functions + call graph ────────────────────────────────────

def parse_codebase(source: str) -> tuple[list[dict], dict[str, list[str]], dict[str, list[str]]]:
    """
    Returns:
        functions  — list of {name, text, docstring, line_start, line_end}
        call_graph — {caller: [callees]}  (CALLS edges)
        called_by  — {callee: [callers]}  (CALLED_BY edges)
    """
    tree = ast.parse(source)
    lines = source.splitlines()

    all_func_names: set[str] = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    functions: list[dict] = []
    call_graph: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        text = "\n".join(lines[start:end]).strip()
        docstring = ast.get_docstring(node) or ""

        callees: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in all_func_names and child.func.id != node.name:
                    callees.append(child.func.id)

        call_graph[node.name] = list(dict.fromkeys(callees))
        functions.append({
            "name": node.name,
            "text": text,
            "docstring": docstring,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
        })

    # Build reverse graph
    called_by: dict[str, list[str]] = {f["name"]: [] for f in functions}
    for caller, callees in call_graph.items():
        for callee in callees:
            if callee in called_by:
                called_by[callee].append(caller)

    return functions, call_graph, called_by


# ── Three embedding strategies ────────────────────────────────────────────────

def strategy_a_raw(func: dict, _call_graph: dict, _called_by: dict) -> str:
    """Strategy A: raw function body only."""
    return func["text"]


def strategy_b_struct_prefix(func: dict, call_graph: dict, called_by: dict) -> str:
    """Strategy B: prepend called_by / calls as comment lines before the body."""
    callers = called_by.get(func["name"], [])
    callees = call_graph.get(func["name"], [])

    prefix_lines = []
    if callers:
        prefix_lines.append(f"# called_by: {', '.join(callers)}")
    if callees:
        prefix_lines.append(f"# calls: {', '.join(callees)}")

    if prefix_lines:
        return "\n".join(prefix_lines) + "\n" + func["text"]
    return func["text"]


def strategy_c_struct_doc(func: dict, call_graph: dict, called_by: dict) -> str:
    """Strategy C: inject called_by / calls into the docstring."""
    callers = called_by.get(func["name"], [])
    callees = call_graph.get(func["name"], [])

    if not callers and not callees:
        return func["text"]

    struct_note = []
    if callers:
        struct_note.append(f"Called by: {', '.join(callers)}.")
    if callees:
        struct_note.append(f"Calls: {', '.join(callees)}.")

    struct_str = " ".join(struct_note)

    # Inject into existing docstring, or add a new one
    text = func["text"]
    if func["docstring"]:
        old_doc = f'"""{func["docstring"]}"""'
        new_doc = f'"""{func["docstring"]} {struct_str}"""'
        text = text.replace(old_doc, new_doc, 1)
    else:
        # Insert docstring after the def line
        lines = text.splitlines()
        def_line_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("def ")), 0
        )
        lines.insert(def_line_idx + 1, f'    """{struct_str}"""')
        text = "\n".join(lines)

    return text


STRATEGIES = {
    "A_raw_code": strategy_a_raw,
    "B_struct_prefix": strategy_b_struct_prefix,
    "C_struct_doc": strategy_c_struct_doc,
}


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


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for name in relevant if name in top_k)
    return hits / len(relevant) if relevant else 0.0


def build_index(
    functions: list[dict],
    strategy_fn,
    call_graph: dict,
    called_by: dict,
) -> list[tuple[str, list[float]]]:
    result = []
    for func in functions:
        text = strategy_fn(func, call_graph, called_by)
        emb = embed_one(text)
        result.append((func["name"], emb))
        time.sleep(0.3)
    return result


def evaluate(
    strategy_name: str,
    indexed: list[tuple[str, list[float]]],
) -> dict:
    r3_scores, r5_scores = [], []
    per_query = []

    for q in QUERIES:
        q_emb = embedder.embed_query(q["query"])
        time.sleep(0.2)

        scored = sorted(indexed, key=lambda x: cosine_sim(q_emb, x[1]), reverse=True)
        top5 = [name for name, _ in scored[:5]]

        r3 = recall_at_k(top5, q["relevant_names"], k=3)
        r5 = recall_at_k(top5, q["relevant_names"], k=5)
        r3_scores.append(r3)
        r5_scores.append(r5)
        per_query.append({
            "query": q["query"][:50],
            "relevant": q["relevant_names"],
            "r3": r3,
            "r5": r5,
            "top5": top5,
        })

    return {
        "name": strategy_name,
        "recall@3": round(statistics.mean(r3_scores), 3),
        "recall@5": round(statistics.mean(r5_scores), 3),
        "per_query": per_query,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    source_file = Path(__file__).parent / "sample_codebase.py"
    source = source_file.read_text()

    print("\n" + "=" * WIDTH)
    print("Structure-Aware Embedding Comparison")
    print(f"Source: {len(source.splitlines())} lines  |  Model: bge-large-zh-v1.5")
    print("=" * WIDTH)

    functions, call_graph, called_by = parse_codebase(source)
    print(f"\n  Parsed {len(functions)} functions")

    # Show what the struct prefix looks like for the key function
    print(f"\n  {'─'*WIDTH}")
    print("  Embedding text preview: calculate_order_total")
    print(f"  {'─'*WIDTH}")
    calc = next(f for f in functions if f["name"] == "calculate_order_total")
    print("\n  Strategy A (raw):")
    print("  " + calc["text"][:200].replace("\n", "\n  "))
    print("\n  Strategy B (struct prefix):")
    b_text = strategy_b_struct_prefix(calc, call_graph, called_by)
    print("  " + b_text[:250].replace("\n", "\n  "))
    print("\n  Strategy C (struct doc):")
    c_text = strategy_c_struct_doc(calc, call_graph, called_by)
    print("  " + c_text[:300].replace("\n", "\n  "))

    # Evaluate all three
    results = []
    for name, fn in STRATEGIES.items():
        print(f"\n  {'─'*WIDTH}")
        print(f"  Building index: {name} ...")
        indexed = build_index(functions, fn, call_graph, called_by)
        result = evaluate(name, indexed)
        results.append(result)
        print(f"  → Recall@3={result['recall@3']:.3f}  Recall@5={result['recall@5']:.3f}")

    # Summary
    print(f"\n{'='*WIDTH}")
    print("Results Summary")
    print(f"{'─'*WIDTH}")
    print(f"\n  {'Strategy':<25} {'R@3':>7}  {'R@5':>7}  {'vs A':>7}")
    print(f"  {'─'*25} {'─'*7}  {'─'*7}  {'─'*7}")
    baseline_r5 = results[0]["recall@5"]
    for r in results:
        delta = r["recall@5"] - baseline_r5
        delta_str = f"{delta:+.3f}" if delta != 0 else "  base"
        print(f"  {r['name']:<25} {r['recall@3']:>7.3f}  {r['recall@5']:>7.3f}  {delta_str:>7}")

    # Per-query divergence
    print(f"\n  Per-query Recall@5:")
    print(f"  {'Query':<50} {'A':>6}  {'B':>6}  {'C':>6}")
    print(f"  {'─'*50} {'─'*6}  {'─'*6}  {'─'*6}")
    for i in range(len(QUERIES)):
        a = results[0]["per_query"][i]["r5"]
        b = results[1]["per_query"][i]["r5"]
        c = results[2]["per_query"][i]["r5"]
        marker = " ←" if len({a, b, c}) > 1 else ""
        print(f"  {results[0]['per_query'][i]['query']:<50} {a:>6.2f}  {b:>6.2f}  {c:>6.2f}{marker}")

    # Deep dive on Q8
    print(f"\n  {'─'*WIDTH}")
    print("  Q8 deep dive: 'process payment and create Stripe charge'")
    print(f"  {'─'*WIDTH}")
    for r in results:
        q8 = r["per_query"][7]
        print(f"  {r['name']:<25}  top-5: {q8['top5']}")

    print(f"\n{'='*WIDTH}\n")


if __name__ == "__main__":
    main()
