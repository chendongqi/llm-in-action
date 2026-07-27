"""
Codebase KB Article 04 — Code Chunking Strategy Comparison

Compares three chunking strategies for code retrieval:

  Strategy 1: Fixed-line chunking
    Split code every N lines (common default in document RAG tools).
    Problem: functions get split in the middle, destroying semantic units.

  Strategy 2: File-level chunking
    One chunk per file.
    Problem: large files exceed context limits; retrieval precision suffers.

  Strategy 3: AST function-level chunking
    One chunk per function, boundaries determined by the AST.
    Hypothesis: preserves semantic units, better precision.

Each strategy is evaluated on the same 12 queries from Article 03.
Metric: Recall@3 and Recall@5.

Additional experiment: show exactly what gets broken when a function
is split by fixed-line chunking.

All strategies use the same embedding approach: Strategy A (raw code)
from Article 03, which scored Recall@5=0.958.

Run:
    conda activate llm_base
    python demo_chunking_strategies.py
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


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_fixed_lines(source: str, chunk_size: int = 20, overlap: int = 3) -> list[dict]:
    """Split source code every N lines with optional overlap."""
    lines = source.splitlines()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(lines), step):
        chunk_lines = lines[i: i + chunk_size]
        text = "\n".join(chunk_lines).strip()
        if text:
            chunks.append({
                "id": f"fixed_line_{i}",
                "text": text,
                "line_start": i + 1,
                "line_end": min(i + chunk_size, len(lines)),
                "type": "fixed_lines",
            })
    return chunks


def chunk_file_level(source: str) -> list[dict]:
    """One chunk for the entire file."""
    return [{
        "id": "whole_file",
        "text": source.strip(),
        "line_start": 1,
        "line_end": len(source.splitlines()),
        "type": "file_level",
    }]


def chunk_ast_functions(source: str) -> list[dict]:
    """One chunk per function, boundaries from the AST."""
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        func_lines = lines[start:end]
        text = "\n".join(func_lines).strip()
        docstring = ast.get_docstring(node) or ""
        chunks.append({
            "id": node.name,
            "text": text,
            "name": node.name,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "docstring": docstring,
            "type": "ast_function",
        })
    return chunks


# ── Evaluation queries (same as Article 03) ───────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(retrieved_chunks: list[dict], relevant_names: list[str], k: int) -> float:
    """
    For AST chunks: chunk id = function name, direct match.
    For fixed-line / file chunks: check if any relevant function name appears in the text.
    """
    top_k = retrieved_chunks[:k]
    hits = 0
    for name in relevant_names:
        for chunk in top_k:
            # AST chunk: exact id match
            if chunk.get("id") == name:
                hits += 1
                break
            # Fixed/file chunk: check text contains function definition
            if f"def {name}(" in chunk.get("text", ""):
                hits += 1
                break
    return hits / len(relevant_names) if relevant_names else 0.0


MAX_CHARS = 1200  # SiliconFlow bge model safe character limit

def embed_one(text: str) -> list[float]:
    """Embed a single text; strip non-ASCII chars that SiliconFlow rejects."""
    clean = text[:MAX_CHARS].encode("ascii", errors="ignore").decode("ascii")
    return embedder.embed_query(clean)


def embed_chunks(chunks: list[dict]) -> list[tuple[dict, list[float]]]:
    """Embed all chunks one by one, return (chunk, embedding) pairs."""
    result = []
    for chunk in chunks:
        emb = embed_one(chunk["text"])
        result.append((chunk, emb))
        time.sleep(0.3)
    return result


def retrieve(query: str, indexed: list[tuple[dict, list[float]]], k: int = 5) -> list[dict]:
    q_emb = embedder.embed_query(query)
    time.sleep(0.2)
    scored = sorted(indexed, key=lambda x: cosine_sim(q_emb, x[1]), reverse=True)
    return [chunk for chunk, _ in scored[:k]]


def evaluate_strategy(name: str, chunks: list[dict]) -> dict:
    print(f"\n  [{name}] {len(chunks)} chunks  avg_len={sum(len(c['text']) for c in chunks)//len(chunks)} chars")
    indexed = embed_chunks(chunks)

    r3_scores, r5_scores = [], []
    per_query = []
    for q in QUERIES:
        top5 = retrieve(q["query"], indexed, k=5)
        r3 = recall_at_k(top5, q["relevant_names"], k=3)
        r5 = recall_at_k(top5, q["relevant_names"], k=5)
        r3_scores.append(r3)
        r5_scores.append(r5)
        per_query.append({"query": q["query"][:50], "r3": r3, "r5": r5,
                          "relevant": q["relevant_names"]})

    return {
        "name": name,
        "num_chunks": len(chunks),
        "avg_chunk_len": sum(len(c["text"]) for c in chunks) // len(chunks),
        "recall@3": round(statistics.mean(r3_scores), 3),
        "recall@5": round(statistics.mean(r5_scores), 3),
        "per_query": per_query,
    }


# ── Demonstrate broken function ───────────────────────────────────────────────

def show_broken_function(source: str) -> None:
    """Show an example of a function split across two fixed-line chunks."""
    print(f"\n  {'─'*WIDTH}")
    print("  Fixed-line chunking: function split demonstration")
    print(f"  {'─'*WIDTH}")

    chunks = chunk_fixed_lines(source, chunk_size=20, overlap=3)

    # Find a chunk boundary that cuts through a function
    ast_funcs = chunk_ast_functions(source)

    for func in ast_funcs:
        func_start = func["line_start"]
        func_end = func["line_end"]
        func_len = func_end - func_start + 1

        # Find which fixed chunks overlap with this function
        overlapping = [
            c for c in chunks
            if c["line_start"] <= func_end and c["line_end"] >= func_start
        ]

        if len(overlapping) >= 2:
            print(f"\n  Function '{func['name']}' (lines {func_start}-{func_end}, {func_len} lines)")
            print(f"  Splits across {len(overlapping)} chunks:")
            for c in overlapping:
                overlap_start = max(c["line_start"], func_start)
                overlap_end = min(c["line_end"], func_end)
                overlap_lines = overlap_end - overlap_start + 1
                print(f"    Chunk lines {c['line_start']}-{c['line_end']}: "
                      f"contains lines {overlap_start}-{overlap_end} "
                      f"({overlap_lines}/{func_len} lines of function)")
            print(f"\n  Problem: Neither chunk contains the complete function.")
            print(f"  A query about '{func['name']}' will retrieve a partial implementation.")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    source_file = Path(__file__).parent / "sample_codebase.py"
    source = source_file.read_text()

    print("\n" + "=" * WIDTH)
    print("Code Chunking Strategy Comparison")
    print(f"Source: {len(source.splitlines())} lines  |  Model: bge-large-zh-v1.5")
    print("=" * WIDTH)

    # Show broken function first (no API calls needed)
    show_broken_function(source)

    # Build chunks for each strategy
    strategies = [
        ("1_fixed_lines_20", chunk_fixed_lines(source, chunk_size=20, overlap=3)),
        ("2_file_level", chunk_file_level(source)),
        ("3_ast_function", chunk_ast_functions(source)),
    ]

    print(f"\n{'─'*WIDTH}")
    print("Evaluating all strategies...")

    results = []
    for name, chunks in strategies:
        result = evaluate_strategy(name, chunks)
        results.append(result)
        print(f"  → Recall@3={result['recall@3']:.3f}  Recall@5={result['recall@5']:.3f}")

    # Summary
    print(f"\n{'='*WIDTH}")
    print("Results Summary")
    print(f"{'─'*WIDTH}")
    print(f"\n  {'Strategy':<28} {'Chunks':>7}  {'AvgLen':>7}  {'R@3':>7}  {'R@5':>7}  {'vs AST R@5':>10}")
    print(f"  {'─'*28} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")

    ast_r5 = next(r["recall@5"] for r in results if "ast" in r["name"])
    for r in results:
        delta = r["recall@5"] - ast_r5
        delta_str = f"{delta:+.3f}" if delta != 0 else "  base"
        print(f"  {r['name']:<28} {r['num_chunks']:>7}  {r['avg_chunk_len']:>7}  "
              f"{r['recall@3']:>7.3f}  {r['recall@5']:>7.3f}  {delta_str:>10}")

    # Per-query detail: show where strategies diverge
    print(f"\n  Per-query Recall@5 (divergent cases):")
    print(f"  {'Query':<50} {'Fixed':>6}  {'File':>6}  {'AST':>6}")
    print(f"  {'─'*50} {'─'*6}  {'─'*6}  {'─'*6}")

    fixed_pq = results[0]["per_query"]
    file_pq = results[1]["per_query"]
    ast_pq = results[2]["per_query"]

    for i in range(len(QUERIES)):
        f_r5 = fixed_pq[i]["r5"]
        fi_r5 = file_pq[i]["r5"]
        a_r5 = ast_pq[i]["r5"]
        if len({f_r5, fi_r5, a_r5}) > 1:
            marker = " ←"
        else:
            marker = ""
        print(f"  {fixed_pq[i]['query']:<50} {f_r5:>6.2f}  {fi_r5:>6.2f}  {a_r5:>6.2f}{marker}")

    print(f"\n{'='*WIDTH}\n")


if __name__ == "__main__":
    main()
