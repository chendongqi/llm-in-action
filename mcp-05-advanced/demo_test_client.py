"""
Test client for demo_advanced_server.py

Demonstrates:
  - Dynamic resource (sprint/current changes on each read)
  - Parameterized resource (project stats per project key)
  - Aggregate resource (dashboard)
  - Conditional prompt rendering (incident_report P0 vs P1)
  - Multi-turn prompt (pr_description)
  - No-args prompt (standup_update)

Run:
    conda activate llm_base
    python demo_test_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

WIDTH = 70


async def send(proc, req_id: int, method: str, params: dict) -> dict:
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    proc.stdin.write((json.dumps(msg) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]
    line = await proc.stdout.readline()  # type: ignore[union-attr]
    return json.loads(line)


async def run() -> None:
    server_path = Path(__file__).parent / "demo_advanced_server.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(server_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    req_id = 0
    def nid() -> int:
        nonlocal req_id; req_id += 1; return req_id

    # Initialize
    await send(proc, nid(), "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    })
    proc.stdin.write((json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"}) + "\n").encode())  # type: ignore[union-attr]
    await proc.stdin.drain()  # type: ignore[union-attr]

    print("=" * WIDTH)
    print("Advanced MCP Server — Resources and Prompts Test")
    print("=" * WIDTH)

    # ── List resources ────────────────────────────────────────────────────────
    print("\n[resources/list]")
    r = await send(proc, nid(), "resources/list", {})
    resources = r.get("result", {}).get("resources", [])
    for res in resources:
        print(f"  {res['uri']}")
        print(f"    {res['description'][:80]}")

    # ── Dynamic resource: read twice, show progress changes ───────────────────
    print("\n[Dynamic resource — read sprint/current twice to see progress change]")
    r1 = await send(proc, nid(), "resources/read", {"uri": "jira://sprint/current"})
    data1 = json.loads(r1["result"]["contents"][0]["text"])
    print(f"  Read 1: progress={data1['progress_pct']}%  last_updated={data1['last_updated'][-8:]}")

    await asyncio.sleep(0.1)  # small delay to ensure timestamp changes

    r2 = await send(proc, nid(), "resources/read", {"uri": "jira://sprint/current"})
    data2 = json.loads(r2["result"]["contents"][0]["text"])
    print(f"  Read 2: progress={data2['progress_pct']}%  last_updated={data2['last_updated'][-8:]}")
    changed = "✓ data changed between reads" if data1 != data2 else "= same data"
    print(f"  → {changed}")

    # ── Parameterized resource ────────────────────────────────────────────────
    print("\n[Parameterized resource — project stats per key]")
    for proj_key in ["PROJ", "MOBILE", "INFRA"]:
        r = await send(proc, nid(), "resources/read", {"uri": f"jira://project/{proj_key}/stats"})
        data = json.loads(r["result"]["contents"][0]["text"])
        print(f"  jira://project/{proj_key}/stats → total={data['total']}, "
              f"by_status={data['by_status']}")

    # ── Aggregate dashboard ───────────────────────────────────────────────────
    print("\n[Dashboard resource]")
    r = await send(proc, nid(), "resources/read", {"uri": "jira://dashboard"})
    dash = json.loads(r["result"]["contents"][0]["text"])
    print(f"  total_open={dash['total_open']}, "
          f"unassigned_critical={dash['unassigned_critical']}")
    for proj, stats in dash["by_project"].items():
        print(f"  {proj}: open={stats['open']}, in_progress={stats['in_progress']}, p0_open={stats['p0_open']}")

    # ── Conditional prompt: P0 vs P1 ─────────────────────────────────────────
    print("\n[Conditional prompt — incident_report P0 vs P1]")
    for severity in ["P0", "P1"]:
        r = await send(proc, nid(), "prompts/get", {
            "name": "incident_report",
            "arguments": {
                "issue_key": "PROJ-103",
                "severity": severity,
                "affected_users": "1500",
                "workaround": "Restart the WebSocket service to temporarily clear connections." if severity == "P1" else ""
            }
        })
        messages = r["result"]["messages"]
        text = messages[0]["content"]["text"]
        has_escalation = "Escalation" in text
        has_workaround = "Workaround" in text
        print(f"  {severity}: {len(messages)} turn(s), "
              f"escalation_section={'✓' if has_escalation else '✗'}, "
              f"workaround_section={'✓' if has_workaround else '✗'}")
        print(f"    preview: {text[:80].replace(chr(10),' ')}...")

    # ── Multi-turn prompt ─────────────────────────────────────────────────────
    print("\n[Multi-turn prompt — pr_description]")
    r = await send(proc, nid(), "prompts/get", {
        "name": "pr_description",
        "arguments": {
            "issue_key": "PROJ-102",
            "changed_files": "src/api/rate_limiter.py, tests/test_rate_limiter.py"
        }
    })
    messages = r["result"]["messages"]
    print(f"  Turn count: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        role = msg["role"]
        preview = msg["content"]["text"][:60].replace("\n", " ")
        print(f"  Turn {i} ({role}): {preview}...")

    # ── No-args prompt ────────────────────────────────────────────────────────
    print("\n[No-required-args prompt — standup_update]")
    r = await send(proc, nid(), "prompts/get", {
        "name": "standup_update",
        "arguments": {"team_member": "alice"}
    })
    messages = r["result"]["messages"]
    text = messages[0]["content"]["text"]
    references_resource = "jira://sprint/current" in text
    references_tool = "search_issues" in text
    print(f"  References resource: {'✓' if references_resource else '✗'}")
    print(f"  References tool:     {'✓' if references_tool else '✗'}")
    print(f"  Preview: {text[:100].replace(chr(10),' ')}...")

    print(f"\n{'=' * WIDTH}")
    print("Test complete")
    print(f"{'=' * WIDTH}\n")

    proc.terminate()
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(run())
