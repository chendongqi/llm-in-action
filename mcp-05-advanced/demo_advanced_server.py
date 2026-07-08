"""
MCP Article 05 — Resources and Prompts Deep Dive

Extends the Article 04 Jira Server with more sophisticated
Resources and Prompts patterns:

Resources (new patterns):
  jira://sprint/current           dynamic resource (changes on every read)
  jira://project/{key}/stats      parameterized URI — different data per project
  jira://dashboard                aggregate view across all projects

Prompts (new patterns):
  incident_report    3 required args, conditional sections based on severity
  pr_description     multi-turn prompt (system + user + assistant messages)
  standup_update     no required args, fetches context from tools at render time

Key concepts demonstrated:
  1. Dynamic Resources       return live-changing data, not static snapshots
  2. Parameterized URIs      one handler, many resources
  3. Conditional Prompts     render different sections based on argument values
  4. Multi-turn Prompts      system message + conversation history in one template

Run:
    conda activate llm_base
    python demo_advanced_server.py

Test all resources and prompts:
    python demo_test_client.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jira-mcp-advanced")

# ── Shared data (same as Article 04, condensed) ───────────────────────────────

PROJECTS = {
    "PROJ": {"name": "Core Platform", "lead": "alice"},
    "MOBILE": {"name": "Mobile App", "lead": "bob"},
    "INFRA": {"name": "Infrastructure", "lead": "carol"},
}

ISSUES = {
    "PROJ-101": {"key":"PROJ-101","project":"PROJ","summary":"NPE in parseInput()","status":"Open","priority":"P1","issue_type":"Bug","assignee":"alice","sprint":"Sprint 42"},
    "PROJ-102": {"key":"PROJ-102","project":"PROJ","summary":"Add rate limiting to API","status":"In Progress","priority":"P2","issue_type":"Story","assignee":"bob","sprint":"Sprint 42"},
    "PROJ-103": {"key":"PROJ-103","project":"PROJ","summary":"Memory leak in WebSocket pool","status":"Open","priority":"P0","issue_type":"Bug","assignee":None,"sprint":"Sprint 42"},
    "MOBILE-55": {"key":"MOBILE-55","project":"MOBILE","summary":"Crash on Android 14","status":"Open","priority":"P1","issue_type":"Bug","assignee":"bob","sprint":"Sprint 15"},
    "INFRA-12": {"key":"INFRA-12","project":"INFRA","summary":"Upgrade PostgreSQL to 16","status":"Done","priority":"P2","issue_type":"Task","assignee":"carol","sprint":"Sprint 8"},
}

# Simulated sprint state that changes over time (to demo dynamic resources)
_sprint_progress_pct = 65  # will increment each time resource is read

server = Server("jira-mcp-advanced")


# ── Tools (minimal, same as before) ──────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_issue",
            description="Get full details of a Jira issue by key.",
            inputSchema={
                "type": "object",
                "properties": {"issue_key": {"type": "string"}},
                "required": ["issue_key"]
            }
        ),
        Tool(
            name="search_issues",
            description="Search Jira issues by keyword. Returns matching issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "project": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_issue":
        key = arguments.get("issue_key", "").upper()
        if key not in ISSUES:
            return [TextContent(type="text", text=f"Issue '{key}' not found", isError=True)]  # type: ignore[call-arg]
        i = ISSUES[key]
        return [TextContent(type="text", text=(
            f"[{i['key']}] {i['summary']}\n"
            f"Status: {i['status']} | Priority: {i['priority']} | Sprint: {i.get('sprint','—')}"
        ))]

    if name == "search_issues":
        query = arguments.get("query", "").lower()
        proj = arguments.get("project", "").upper()
        results = [
            i for i in ISSUES.values()
            if (not proj or i["project"] == proj)
            and query in i["summary"].lower()
        ]
        if not results:
            return [TextContent(type="text", text="No issues found.")]
        return [TextContent(type="text", text="\n".join(
            f"[{i['key']}] {i['summary']} ({i['status']})" for i in results
        ))]

    return [TextContent(type="text", text=f"Unknown tool: {name}", isError=True)]  # type: ignore[call-arg]


# ── Resources ─────────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> list[Resource]:
    # Static list — but the content of each resource can be dynamic
    resources = [
        Resource(
            uri="jira://sprint/current",  # type: ignore[arg-type]
            name="Current Sprint Status",
            description=(
                "Live status of the active sprint: name, dates, issue counts, "
                "progress percentage. Read this when the user asks about sprint "
                "health, velocity, or what's in progress."
            ),
            mimeType="application/json"
        ),
        Resource(
            uri="jira://dashboard",  # type: ignore[arg-type]
            name="Cross-Project Dashboard",
            description=(
                "Aggregate view: open bugs by project, P0 issues, "
                "unassigned critical issues. Read for high-level status overview."
            ),
            mimeType="application/json"
        ),
    ]
    # Parameterized resources: one entry per project
    for key, proj in PROJECTS.items():
        resources.append(Resource(
            uri=f"jira://project/{key}/stats",  # type: ignore[arg-type]
            name=f"{proj['name']} Stats",
            description=f"Issue statistics for {proj['name']} ({key}): counts by status and priority.",
            mimeType="application/json"
        ))
    return resources


@server.read_resource()  # type: ignore[arg-type]
async def read_resource(uri: str) -> str:
    uri_str = str(uri)
    logger.info("resources/read: %s", uri_str)

    # ── Dynamic resource: changes on every read ───────────────────────────────
    if uri_str == "jira://sprint/current":
        global _sprint_progress_pct
        # Simulate progress advancing (dynamic data)
        _sprint_progress_pct = min(100, _sprint_progress_pct + random.randint(0, 3))

        sprint_issues = [i for i in ISSUES.values() if i.get("sprint") == "Sprint 42"]
        done = sum(1 for i in sprint_issues if i["status"] == "Done")
        in_progress = sum(1 for i in sprint_issues if i["status"] == "In Progress")
        open_count = sum(1 for i in sprint_issues if i["status"] == "Open")
        p0_open = sum(1 for i in sprint_issues if i["priority"] == "P0" and i["status"] != "Done")

        return json.dumps({
            "sprint_name": "Sprint 42",
            "start_date": "2026-06-30",
            "end_date": "2026-07-13",
            "days_remaining": 5,
            "progress_pct": _sprint_progress_pct,  # changes on each read
            "issues": {
                "total": len(sprint_issues),
                "done": done,
                "in_progress": in_progress,
                "open": open_count,
            },
            "p0_open": p0_open,
            "last_updated": datetime.now(timezone.utc).isoformat(),  # timestamp changes
        }, indent=2)

    # ── Static aggregate resource ─────────────────────────────────────────────
    if uri_str == "jira://dashboard":
        by_project = {}
        for proj_key in PROJECTS:
            proj_issues = [i for i in ISSUES.values() if i["project"] == proj_key]
            by_project[proj_key] = {
                "open": sum(1 for i in proj_issues if i["status"] == "Open"),
                "in_progress": sum(1 for i in proj_issues if i["status"] == "In Progress"),
                "done": sum(1 for i in proj_issues if i["status"] == "Done"),
                "p0_open": sum(1 for i in proj_issues if i["priority"] == "P0" and i["status"] != "Done"),
            }
        unassigned_critical = [
            i["key"] for i in ISSUES.values()
            if i["priority"] in ("P0", "P1") and i["assignee"] is None
        ]
        return json.dumps({
            "by_project": by_project,
            "unassigned_critical": unassigned_critical,
            "total_open": sum(v["open"] for v in by_project.values()),
        }, indent=2)

    # ── Parameterized resource: jira://project/{key}/stats ────────────────────
    if uri_str.startswith("jira://project/") and uri_str.endswith("/stats"):
        # Parse project key from URI: jira://project/PROJ/stats
        parts = uri_str.split("/")
        if len(parts) >= 4:
            proj_key = parts[3].upper()
        else:
            raise ValueError(f"Invalid project stats URI: {uri_str}")

        if proj_key not in PROJECTS:
            raise ValueError(f"Unknown project: {proj_key}. Available: {', '.join(PROJECTS)}")

        proj_issues = [i for i in ISSUES.values() if i["project"] == proj_key]
        by_status = {}
        by_priority = {}
        by_type = {}
        for i in proj_issues:
            by_status[i["status"]] = by_status.get(i["status"], 0) + 1
            by_priority[i["priority"]] = by_priority.get(i["priority"], 0) + 1
            by_type[i["issue_type"]] = by_type.get(i["issue_type"], 0) + 1

        return json.dumps({
            "project": proj_key,
            "name": PROJECTS[proj_key]["name"],
            "lead": PROJECTS[proj_key]["lead"],
            "total": len(proj_issues),
            "by_status": by_status,
            "by_priority": by_priority,
            "by_type": by_type,
        }, indent=2)

    raise ValueError(f"Unknown resource URI: {uri_str}")


# ── Prompts ───────────────────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="incident_report",
            description=(
                "Generate a structured incident report for a P0/P1 bug. "
                "Severity determines which sections are included: "
                "P0 adds escalation contacts and SLA breach risk."
            ),
            arguments=[
                PromptArgument(name="issue_key", description="Jira issue key of the incident", required=True),
                PromptArgument(name="severity", description="Incident severity: P0 or P1", required=True),
                PromptArgument(name="affected_users", description="Estimated number of affected users", required=True),
                PromptArgument(name="workaround", description="Temporary workaround if any", required=False),
            ]
        ),
        Prompt(
            name="pr_description",
            description=(
                "Generate a pull request description from a Jira issue. "
                "Returns a multi-turn prompt: system context + user request + "
                "assistant thinking step (to guide structured output)."
            ),
            arguments=[
                PromptArgument(name="issue_key", description="Jira issue key the PR addresses", required=True),
                PromptArgument(name="changed_files", description="Comma-separated list of changed files", required=False),
            ]
        ),
        Prompt(
            name="standup_update",
            description=(
                "Generate a standup update from recent Jira activity. "
                "No required arguments — uses current data automatically."
            ),
            arguments=[
                PromptArgument(name="team_member", description="Name of the team member (optional)", required=False),
            ]
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    args = arguments or {}

    # ── incident_report: conditional sections based on severity ───────────────
    if name == "incident_report":
        issue_key = args.get("issue_key", "")
        severity = args.get("severity", "P1").upper()
        affected_users = args.get("affected_users", "unknown")
        workaround = args.get("workaround", "")

        # Conditional section: only for P0
        p0_section = ""
        if severity == "P0":
            p0_section = (
                "\n## Escalation\n"
                "- Engineering VP: must be notified within 30 minutes\n"
                "- On-call engineer: page immediately\n"
                "- SLA breach risk: this incident may breach the 4-hour P0 SLA\n"
            )

        workaround_section = ""
        if workaround:
            workaround_section = f"\n## Workaround\n{workaround}\n"

        template = (
            f"Create a formal incident report for Jira issue {issue_key}.\n\n"
            f"Use the get_issue tool to fetch the issue details, then produce:\n\n"
            f"## Incident Report: {issue_key}\n"
            f"**Severity:** {severity}\n"
            f"**Affected Users:** {affected_users}\n\n"
            f"## Summary\n[2-3 sentence description of what is broken]\n\n"
            f"## Timeline\n[When was it first reported / detected]\n\n"
            f"## Impact\n[What functionality is affected, user-facing symptoms]\n\n"
            f"## Root Cause (Preliminary)\n[Initial hypothesis based on issue description]\n\n"
            f"## Next Steps\n[Immediate actions to mitigate or fix]"
            f"{p0_section}"
            f"{workaround_section}"
        )

        return GetPromptResult(
            description="Incident report prompt",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=template)
            )]
        )

    # ── pr_description: multi-turn prompt ─────────────────────────────────────
    if name == "pr_description":
        issue_key = args.get("issue_key", "")
        changed_files = args.get("changed_files", "")

        files_section = ""
        if changed_files:
            files_section = f"\nChanged files: {changed_files}"

        # Multi-turn: system + user + assistant thinking step
        return GetPromptResult(
            description="Pull request description prompt",
            messages=[
                # Turn 1: System context
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=(
                        f"You are a senior engineer writing a PR description.\n"
                        f"PR addresses: {issue_key}{files_section}\n\n"
                        f"First, use get_issue to read the Jira issue details."
                    ))
                ),
                # Turn 2: Assistant thinking step (guides structured output)
                PromptMessage(
                    role="assistant",
                    content=TextContent(type="text", text=(
                        "I'll fetch the issue details and then write a PR description "
                        "with: title, motivation, changes summary, test plan, and "
                        "links to the Jira issue."
                    ))
                ),
                # Turn 3: User final request
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=(
                        "Now write the complete PR description in Markdown."
                    ))
                ),
            ]
        )

    # ── standup_update: no required args, references live data ────────────────
    if name == "standup_update":
        team_member = args.get("team_member", "")
        member_line = f"for {team_member}" if team_member else "for the team"

        return GetPromptResult(
            description="Standup update prompt",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=(
                    f"Generate a daily standup update {member_line}.\n\n"
                    f"Steps:\n"
                    f"1. Read jira://sprint/current to see overall sprint health\n"
                    f"2. Use search_issues to find issues that changed status recently"
                    + (f" by {team_member}" if team_member else "") + "\n"
                    f"3. Write a standup in this format:\n"
                    f"   **Yesterday:** [what was completed]\n"
                    f"   **Today:** [what is planned]\n"
                    f"   **Blockers:** [any impediments, or 'None']\n"
                    f"   **Sprint health:** [summary from sprint resource]"
                ))
            )]
        )

    raise ValueError(f"Unknown prompt: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("Starting jira-mcp-advanced server")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
