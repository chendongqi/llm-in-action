"""
Article 24: Code RAG — Build an AI That Understands Your Codebase

Demonstrates three capabilities:
  1. AST-based code unit extraction  (functions, methods, call graph)
  2. Semantic code search             (embedding + vector retrieval)
  3. Call chain traversal             (BFS over the call graph)
  4. LLM-assisted codebase Q&A        (retrieved context + generation)

Target repo: https://github.com/chendongqi/llm-in-action
             (analysed from local parent directory)
"""

import ast
import json
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

REPO_URL     = "https://github.com/chendongqi/llm-in-action"
REPO_DIR     = Path(__file__).parent.parent       # local copy of the repo
CHROMA_DIR   = "./code_rag_db"
REPORT_PATH  = "./code_rag_report.json"

LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = "glm-4-flash"

EMB_BASE_URL = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")
EMB_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")
EMB_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class CodeUnit:
    name:         str
    kind:         str          # "function" | "method"
    file:         str          # relative path within REPO_DIR
    start_line:   int
    end_line:     int
    source:       str
    docstring:    str
    parent_class: str          # empty for top-level functions
    decorators:   list[str]
    calls:        list[str]    # names of functions/methods invoked

# ── Step 1 : AST extraction ────────────────────────────────────────────────

class _FuncExtractor(ast.NodeVisitor):
    """Walks a single Python file and collects CodeUnit records."""

    def __init__(self, source: str, rel_path: str):
        self._lines       = source.splitlines()
        self._rel_path    = rel_path
        self._class_stack: list[str] = []
        self.units:        list[CodeUnit] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _extract_calls(self, node) -> list[str]:
        calls: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        return sorted(calls)

    def _visit_func(self, node):
        src = "\n".join(self._lines[node.lineno - 1 : node.end_lineno])
        dec_names = []
        for d in node.decorator_list:
            try:
                dec_names.append(ast.unparse(d))
            except Exception:
                dec_names.append("?")
        unit = CodeUnit(
            name         = node.name,
            kind         = "method" if self._class_stack else "function",
            file         = self._rel_path,
            start_line   = node.lineno,
            end_line     = node.end_lineno,
            source       = src,
            docstring    = ast.get_docstring(node) or "",
            parent_class = self._class_stack[-1] if self._class_stack else "",
            decorators   = dec_names,
            calls        = self._extract_calls(node),
        )
        self.units.append(unit)
        self.generic_visit(node)

    visit_FunctionDef      = _visit_func
    visit_AsyncFunctionDef = _visit_func


_SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "site-packages"}

def extract_repo(repo_dir: Path) -> list[CodeUnit]:
    units: list[CodeUnit] = []
    for path in sorted(repo_dir.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(repo_dir))
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source, filename=str(path))
            ext    = _FuncExtractor(source, rel)
            ext.visit(tree)
            units.extend(ext.units)
        except SyntaxError:
            pass
    return units

# ── Step 2 : Call graph ────────────────────────────────────────────────────

class CallGraph:
    """
    Bidirectional call graph over CodeUnit names.
    Only edges between units that exist in the repo are kept (no builtins).
    """

    def __init__(self, units: list[CodeUnit]):
        self.callees: dict[str, set[str]] = defaultdict(set)   # caller  → called
        self.callers: dict[str, set[str]] = defaultdict(set)   # callee  → caller

        known = {u.name for u in units}
        for u in units:
            for callee in u.calls:
                if callee in known:
                    self.callees[u.name].add(callee)
                    self.callers[callee].add(u.name)

    def downstream(self, name: str, depth: int = 4) -> list[str]:
        """All functions transitively called by `name`."""
        return self._bfs(name, self.callees, depth)

    def upstream(self, name: str, depth: int = 4) -> list[str]:
        """All functions that transitively call `name`."""
        return self._bfs(name, self.callers, depth)

    def shortest_path(self, start: str, end: str, max_depth: int = 6) -> Optional[list[str]]:
        """Shortest call chain from start → end, or None if unreachable."""
        queue: deque[list[str]] = deque([[start]])
        visited: set[str] = {start}
        while queue:
            path = queue.popleft()
            if path[-1] == end:
                return path
            if len(path) > max_depth:
                continue
            for nxt in self.callees.get(path[-1], set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    @staticmethod
    def _bfs(start: str, adj: dict[str, set[str]], depth: int) -> list[str]:
        result, visited = [], {start}
        q: deque[tuple[str, int]] = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            if d > depth:
                continue
            if node != start:
                result.append(node)
            for nbr in adj.get(node, set()):
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, d + 1))
        return result

# ── Step 3 : Vector store ──────────────────────────────────────────────────

def build_vectorstore(units: list[CodeUnit], embeddings) -> Chroma:
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    docs = []
    for u in units:
        full_name = f"{u.parent_class}.{u.name}" if u.parent_class else u.name
        # Embedding content: function name + docstring only (short, semantic).
        # Stays well within the 512-token limit.
        # Full source is stored in metadata for LLM Q&A context.
        sig_line = u.source.splitlines()[0] if u.source else ""
        embed_content = f"{full_name}: {u.docstring or sig_line}"[:400]

        docs.append(Document(
            page_content = embed_content,
            metadata = {
                "name":         u.name,
                "kind":         u.kind,
                "file":         u.file,
                "start_line":   u.start_line,
                "parent_class": u.parent_class,
                # Full source stored here for Q&A context (not embedded)
                "source_code":  u.source[:2000],
            },
        ))

    print(f"[vector] Embedding {len(docs)} code units …")
    BATCH = 32
    t0 = time.time()
    vs = Chroma.from_documents(docs[:BATCH], embeddings, persist_directory=CHROMA_DIR)
    for i in range(BATCH, len(docs), BATCH):
        vs.add_documents(docs[i : i + BATCH])
        print(f"[vector]   {min(i + BATCH, len(docs))}/{len(docs)}")
    print(f"[vector] Done in {time.time() - t0:.1f}s")
    return vs

# ── Step 4 : Query interfaces ──────────────────────────────────────────────

def semantic_search(query: str, vs: Chroma, k: int = 5) -> list[dict]:
    hits = vs.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "score":      round(score, 3),
            "name":       doc.metadata["name"],
            "file":       doc.metadata["file"],
            "start_line": doc.metadata["start_line"],
            "kind":       doc.metadata["kind"],
            "snippet":    doc.page_content[:180].replace("\n", " "),
        }
        for doc, score in hits
    ]


_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior code reviewer. Answer questions about the codebase "
     "based on the provided code snippets. Be concise and cite specific "
     "function names and file paths."),
    ("user", "Code context:\n\n{context}\n\nQuestion: {question}"),
])

def llm_code_qa(question: str, vs: Chroma, llm) -> dict:
    docs = vs.similarity_search(question, k=4)
    # Use source_code from metadata for rich context; fall back to page_content
    context = "\n\n---\n\n".join(
        d.metadata.get("source_code", d.page_content)[:600] for d in docs
    )
    chain   = _QA_PROMPT | llm | StrOutputParser()
    t0      = time.time()
    answer  = chain.invoke({"context": context, "question": question})
    return {
        "question":   question,
        "answer":     answer,
        "latency_ms": round((time.time() - t0) * 1000),
        "sources":    [{"file": d.metadata["file"], "name": d.metadata["name"]} for d in docs],
    }

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    llm = ChatOpenAI(
        base_url    = LLM_BASE_URL,
        api_key     = LLM_API_KEY,
        model       = LLM_MODEL,
        temperature = 0,
    )
    embeddings = OpenAIEmbeddings(
        base_url = EMB_BASE_URL,
        api_key  = EMB_API_KEY,
        model    = EMB_MODEL,
    )

    # ── 1. AST extraction ──────────────────────────────────────────────────
    print(f"[ast] Scanning {REPO_DIR} …")
    t0    = time.time()
    units = extract_repo(REPO_DIR)
    t_ast = time.time() - t0

    files     = sorted({u.file for u in units})
    functions = [u for u in units if u.kind == "function"]
    methods   = [u for u in units if u.kind == "method"]

    print(f"[ast] {len(files)} files · {len(functions)} functions · {len(methods)} methods"
          f"  ({t_ast:.2f}s)")

    # ── 2. Call graph ──────────────────────────────────────────────────────
    cg = CallGraph(units)
    total_edges = sum(len(v) for v in cg.callees.values())
    print(f"[graph] {len(cg.callees)} callers · {len(cg.callers)} callees · {total_edges} edges")

    # Find the top-5 most-called functions (core utilities)
    top_called = sorted(cg.callers.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    # Find functions with the deepest outgoing call fan
    top_callers = sorted(cg.callees.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    # Call chain examples: pick the top caller and ask for its downstream chain
    call_chain_examples = []
    for func_name, _ in top_callers[:3]:
        down  = cg.downstream(func_name, depth=3)
        up    = cg.upstream(func_name,   depth=3)
        entry = {
            "function":      func_name,
            "file":          next((u.file for u in units if u.name == func_name), ""),
            "calls_count":   len(cg.callees.get(func_name, [])),
            "downstream":    down[:8],
            "upstream":      up[:5],
        }
        call_chain_examples.append(entry)
        print(f"\n[graph] {func_name}")
        print(f"  directly calls : {sorted(cg.callees.get(func_name, []))}")
        print(f"  downstream     : {down[:6]}")
        print(f"  called by      : {up[:4]}")

    # Shortest-path demo between two real functions (if path exists)
    path_demos = []
    # Try every pair among top-callers until we find a connected pair
    found_path = None
    for src, _ in top_callers:
        for tgt, _ in top_called:
            if src != tgt:
                p = cg.shortest_path(src, tgt)
                if p and len(p) > 1:
                    found_path = {"from": src, "to": tgt, "path": p}
                    break
        if found_path:
            break

    if found_path:
        path_demos.append(found_path)
        print(f"\n[graph] shortest path: {' → '.join(found_path['path'])}")

    # ── 3. Vector store ────────────────────────────────────────────────────
    vs = build_vectorstore(units, embeddings)

    # ── 4. Semantic search ─────────────────────────────────────────────────
    search_queries = [
        "embedding caching to reduce API calls",
        "RAGAS evaluation metrics calculation",
        "rate limiting and access control in enterprise RAG",
        "incremental document indexing with record manager",
        "conversational history aware retriever",
    ]
    search_results = []
    for q in search_queries:
        hits = semantic_search(q, vs, k=3)
        search_results.append({"query": q, "results": hits})
        print(f"\n[search] '{q}'")
        for h in hits[:2]:
            print(f"  {h['score']:.3f}  {h['name']}  ({h['file']}:{h['start_line']})")

    # ── 5. LLM Q&A ─────────────────────────────────────────────────────────
    qa_questions = [
        "这个代码库中的 Embedding 缓存是如何实现的？用了哪些类和存储后端？",
        "RAGAS 评测框架是如何集成的？评测哪几个指标，分别在哪个文件？",
        "企业级 RAG 的多租户隔离机制是什么？如何实现按角色的访问控制？",
    ]
    qa_results = []
    for q in qa_questions:
        result = llm_code_qa(q, vs, llm)
        qa_results.append(result)
        print(f"\n[qa] {q}")
        print(f"  → {result['answer'][:200].strip()}")
        print(f"  latency: {result['latency_ms']}ms")

    # ── 6. Report ──────────────────────────────────────────────────────────
    # Per-article file stats
    article_stats = defaultdict(lambda: {"files": 0, "functions": 0, "methods": 0})
    for u in units:
        parts = Path(u.file).parts
        article = parts[0] if parts else "root"
        article_stats[article]["functions" if u.kind == "function" else "methods"] += 1
    for f in files:
        parts = Path(f).parts
        article = parts[0] if parts else "root"
        article_stats[article]["files"] += 1

    report = {
        "repo": {
            "url":             REPO_URL,
            "local_path":      str(REPO_DIR),
            "py_files":        len(files),
            "functions":       len(functions),
            "methods":         len(methods),
            "total_units":     len(units),
            "ast_extract_s":   round(t_ast, 2),
            "articles":        len(article_stats),
        },
        "call_graph": {
            "unique_callers":  len(cg.callees),
            "unique_callees":  len(cg.callers),
            "total_edges":     total_edges,
            "top_called":      [{"name": n, "caller_count": len(v)} for n, v in top_called],
            "top_callers":     [{"name": n, "callee_count": len(v)} for n, v in top_callers],
        },
        "call_chain_examples": call_chain_examples,
        "path_examples":       path_demos,
        "semantic_search":     search_results,
        "llm_qa":              qa_results,
        "per_article":         dict(article_stats),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Report → {REPORT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  代码 RAG — 分析汇总")
    print("=" * 65)
    r = report["repo"]
    print(f"  Python 文件   : {r['py_files']}")
    print(f"  函数          : {r['functions']}")
    print(f"  方法          : {r['methods']}")
    print(f"  文章目录      : {r['articles']}")
    cg_r = report["call_graph"]
    print(f"  调用图节点    : {cg_r['unique_callers']} 个调用者")
    print(f"  调用图边数    : {cg_r['total_edges']}")
    print(f"  向量化单元    : {r['total_units']}")
    print()
    print("  Top 被调用函数:")
    for item in report["call_graph"]["top_called"]:
        print(f"    {item['name']:30s} ← {item['caller_count']} 处")
    print("=" * 65)


if __name__ == "__main__":
    main()
