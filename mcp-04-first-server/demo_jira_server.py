"""
MCP Article 04 — Building Your First MCP Server

A production-grade Jira MCP Server using mock data (no real Jira credentials needed).
Demonstrates 5 engineering patterns beyond the minimal hello-world:

  1. Multiple tools that compose   search + get + create + update
  2. Input validation              type checks, required fields, enum values
  3. Structured error handling     isError=true with actionable messages
  4. Logging                       structured logs to stderr (not stdout!)
  5. Resources                     project list as a readable data source

Tools:
  search_issues(query, project?, status?, max_results?)
  get_issue(issue_key)
  create_issue(project, summary, description?, priority?, issue_type?)
  update_issue(issue_key, status?, assignee?, priority?)

Resources:
  jira://projects  — list of available projects

Run:
    conda activate llm_base
    python demo_jira_server.py

Test all tools:
    python demo_test_client.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

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

# ── Logging setup ─────────────────────────────────────────────────────────────
# CRITICAL: log to stderr, NEVER stdout. stdout is the JSON-RPC channel.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jira-mcp")

# ── Mock data store ───────────────────────────────────────────────────────────

PROJECTS = {
    "PROJ": {"name": "Core Platform", "lead": "alice"},
    "MOBILE": {"name": "Mobile App", "lead": "bob"},
    "INFRA": {"name": "Infrastructure", "lead": "carol"},
}

ISSUES: dict[str, dict[str, Any]] = {
    "PROJ-101": {
        "key": "PROJ-101", "project": "PROJ",
        "summary": "NullPointerException in parseInput() when config is null",
        "description": "Stack trace shows NPE at line 247 of InputParser.java when config object is not initialized.",
        "status": "Open", "priority": "P1", "issue_type": "Bug",
        "assignee": "alice", "reporter": "user1",
        "created": "2026-06-01T10:00:00Z", "updated": "2026-06-15T14:30:00Z",
    },
    "PROJ-102": {
        "key": "PROJ-102", "project": "PROJ",
        "summary": "Add rate limiting to public API endpoints",
        "description": "We need to add token bucket rate limiting to prevent API abuse.",
        "status": "In Progress", "priority": "P2", "issue_type": "Story",
        "assignee": "bob", "reporter": "alice",
        "created": "2026-06-10T09:00:00Z", "updated": "2026-06-20T11:00:00Z",
    },
    "PROJ-103": {
        "key": "PROJ-103", "project": "PROJ",
        "summary": "Memory leak in WebSocket connection pool",
        "description": "Connections are not properly released after timeout, causing memory to grow unboundedly.",
        "status": "Open", "priority": "P0", "issue_type": "Bug",
        "assignee": None, "reporter": "carol",
        "created": "2026-06-25T08:00:00Z", "updated": "2026-06-25T08:00:00Z",
    },
    "MOBILE-55": {
        "key": "MOBILE-55", "project": "MOBILE",
        "summary": "Crash on Android 14 when opening notification settings",
        "description": "App crashes with IllegalStateException when user navigates to notification settings on Android 14.",
        "status": "Open", "priority": "P1", "issue_type": "Bug",
        "assignee": "bob", "reporter": "user2",
        "created": "2026-06-18T15:00:00Z", "updated": "2026-06-22T09:00:00Z",
    },
    "INFRA-12": {
        "key": "INFRA-12", "project": "INFRA",
        "summary": "Upgrade PostgreSQL from 14 to 16",
        "description": "Upgrade all database instances to PostgreSQL 16 for performance improvements and security patches.",
        "status": "Done", "priority": "P2", "issue_type": "Task",
        "assignee": "carol", "reporter": "carol",
        "created": "2026-05-01T10:00:00Z", "updated": "2026-06-01T16:00:00Z",
    },
}

VALID_STATUSES = {"Open", "In Progress", "In Review", "Done", "Closed"}
VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}
VALID_ISSUE_TYPES = {"Bug", "Story", "Task", "Epic"}

_issue_counter = 200  # next auto-increment ID


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_issue_key(key: str) -> str | None:
    """Return error message if key is invalid, None if valid."""
    if not key or "-" not in key:
        return f"Invalid issue key format: '{key}'. Expected format: PROJECT-123"
    project = key.split("-")[0]
    if project not in PROJECTS:
        return f"Unknown project: '{project}'. Available projects: {', '.join(PROJECTS)}"
    if key not in ISSUES:
        return f"Issue '{key}' not found"
    return None


def format_issue(issue: dict) -> str:
    assignee = issue["assignee"] or "Unassigned"
    return (
        f"[{issue['key']}] {issue['summary']}\n"
        f"  Status: {issue['status']} | Priority: {issue['priority']} "
        f"| Type: {issue['issue_type']}\n"
        f"  Assignee: {assignee} | Reporter: {issue['reporter']}\n"
        f"  Updated: {issue['updated']}"
    )


# ── Server setup ──────────────────────────────────────────────────────────────

server = Server("jira-mcp-demo")


# ── Tools ─────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("tools/list requested")
    return [
        Tool(
            name="search_issues",
            description=(
                "Search Jira issues by keyword, project, or status. "
                "Use when the user asks about bugs, tasks, tickets, or issues. "
                "Returns a list of matching issues with key, summary, status, priority."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (searches summary and description)"
                    },
                    "project": {
                        "type": "string",
                        "description": f"Filter by project key. Available: {', '.join(PROJECTS)}"
                    },
                    "status": {
                        "type": "string",
                        "description": f"Filter by status. One of: {', '.join(sorted(VALID_STATUSES))}"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, max: 50)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_issue",
            description=(
                "Get full details of a specific Jira issue by key. "
                "Use when the user asks about a specific issue like 'PROJ-101'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key, e.g. 'PROJ-101' or 'MOBILE-55'"
                    }
                },
                "required": ["issue_key"]
            }
        ),
        Tool(
            name="create_issue",
            description=(
                "Create a new Jira issue. "
                "Use when the user asks to create, file, or report a bug/task/story."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": f"Project key. One of: {', '.join(PROJECTS)}"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Issue title/summary (required, max 255 chars)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description (optional)"
                    },
                    "priority": {
                        "type": "string",
                        "description": f"Priority level. One of: {', '.join(sorted(VALID_PRIORITIES))}. Default: P2",
                        "default": "P2"
                    },
                    "issue_type": {
                        "type": "string",
                        "description": f"Issue type. One of: {', '.join(sorted(VALID_ISSUE_TYPES))}. Default: Bug",
                        "default": "Bug"
                    }
                },
                "required": ["project", "summary"]
            }
        ),
        Tool(
            name="update_issue",
            description=(
                "Update an existing Jira issue's status, assignee, or priority. "
                "Use when the user asks to change, update, or modify an issue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key to update"
                    },
                    "status": {
                        "type": "string",
                        "description": f"New status. One of: {', '.join(sorted(VALID_STATUSES))}"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Username to assign to (use null to unassign)"
                    },
                    "priority": {
                        "type": "string",
                        "description": f"New priority. One of: {', '.join(sorted(VALID_PRIORITIES))}"
                    }
                },
                "required": ["issue_key"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info("tools/call: %s args=%s", name, arguments)

    # ── search_issues ─────────────────────────────────────────────────────────
    if name == "search_issues":
        query = arguments.get("query", "").lower().strip()
        project_filter = arguments.get("project", "").upper()
        status_filter = arguments.get("status", "")
        max_results = min(int(arguments.get("max_results", 10)), 50)

        # Validate project filter
        if project_filter and project_filter not in PROJECTS:
            return [TextContent(type="text", text=(
                f"Unknown project: '{project_filter}'. "
                f"Available projects: {', '.join(PROJECTS)}"
            ))]  # Not isError — it's a valid response telling the LLM to correct

        # Validate status filter
        if status_filter and status_filter not in VALID_STATUSES:
            return [TextContent(
                type="text",
                text=f"Unknown status: '{status_filter}'. Valid values: {', '.join(sorted(VALID_STATUSES))}"
            )]

        results = []
        for issue in ISSUES.values():
            if project_filter and issue["project"] != project_filter:
                continue
            if status_filter and issue["status"] != status_filter:
                continue
            if query and query not in issue["summary"].lower() and query not in (issue["description"] or "").lower():
                continue
            results.append(issue)

        if not results:
            return [TextContent(type="text", text="No issues found matching your search criteria.")]

        results = results[:max_results]
        output = f"Found {len(results)} issue(s):\n\n"
        output += "\n\n".join(format_issue(i) for i in results)
        return [TextContent(type="text", text=output)]

    # ── get_issue ─────────────────────────────────────────────────────────────
    if name == "get_issue":
        key = arguments.get("issue_key", "").upper().strip()

        err = validate_issue_key(key)
        if err:
            return [TextContent(type="text", text=err, isError=True)]  # type: ignore[call-arg]

        issue = ISSUES[key]
        output = (
            f"Issue: {issue['key']}\n"
            f"Summary: {issue['summary']}\n"
            f"Type: {issue['issue_type']} | Status: {issue['status']} | Priority: {issue['priority']}\n"
            f"Assignee: {issue['assignee'] or 'Unassigned'} | Reporter: {issue['reporter']}\n"
            f"Created: {issue['created']} | Updated: {issue['updated']}\n\n"
            f"Description:\n{issue['description'] or '(no description)'}"
        )
        return [TextContent(type="text", text=output)]

    # ── create_issue ──────────────────────────────────────────────────────────
    if name == "create_issue":
        project = arguments.get("project", "").upper().strip()
        summary = arguments.get("summary", "").strip()
        description = arguments.get("description", "").strip()
        priority = arguments.get("priority", "P2").upper().strip()
        issue_type = arguments.get("issue_type", "Bug").strip()

        # Validate required fields
        if not project:
            return [TextContent(type="text", text="'project' is required.", isError=True)]  # type: ignore[call-arg]
        if not summary:
            return [TextContent(type="text", text="'summary' is required.", isError=True)]  # type: ignore[call-arg]

        # Validate against allowed values
        if project not in PROJECTS:
            return [TextContent(type="text", text=(
                f"Unknown project: '{project}'. Available: {', '.join(PROJECTS)}"
            ), isError=True)]  # type: ignore[call-arg]

        if priority not in VALID_PRIORITIES:
            return [TextContent(type="text", text=(
                f"Invalid priority: '{priority}'. Valid values: {', '.join(sorted(VALID_PRIORITIES))}"
            ), isError=True)]  # type: ignore[call-arg]

        if issue_type not in VALID_ISSUE_TYPES:
            return [TextContent(type="text", text=(
                f"Invalid issue_type: '{issue_type}'. Valid values: {', '.join(sorted(VALID_ISSUE_TYPES))}"
            ), isError=True)]  # type: ignore[call-arg]

        if len(summary) > 255:
            return [TextContent(type="text", text="Summary exceeds 255 characters.", isError=True)]  # type: ignore[call-arg]

        # Create issue
        global _issue_counter
        _issue_counter += 1
        key = f"{project}-{_issue_counter}"
        now = datetime.now(timezone.utc).isoformat()
        ISSUES[key] = {
            "key": key, "project": project,
            "summary": summary, "description": description or None,
            "status": "Open", "priority": priority, "issue_type": issue_type,
            "assignee": None, "reporter": "api",
            "created": now, "updated": now,
        }
        logger.info("Created issue %s", key)
        return [TextContent(type="text", text=f"Created issue {key}: {summary}")]

    # ── update_issue ──────────────────────────────────────────────────────────
    if name == "update_issue":
        key = arguments.get("issue_key", "").upper().strip()

        err = validate_issue_key(key)
        if err:
            return [TextContent(type="text", text=err, isError=True)]  # type: ignore[call-arg]

        # At least one field must be updated
        new_status = arguments.get("status", "").strip()
        new_assignee = arguments.get("assignee")
        new_priority = arguments.get("priority", "").strip().upper()

        if not any([new_status, new_assignee is not None, new_priority]):
            return [TextContent(type="text", text=(
                "No fields to update. Provide at least one of: status, assignee, priority."
            ), isError=True)]  # type: ignore[call-arg]

        if new_status and new_status not in VALID_STATUSES:
            return [TextContent(type="text", text=(
                f"Invalid status: '{new_status}'. Valid values: {', '.join(sorted(VALID_STATUSES))}"
            ), isError=True)]  # type: ignore[call-arg]

        if new_priority and new_priority not in VALID_PRIORITIES:
            return [TextContent(type="text", text=(
                f"Invalid priority: '{new_priority}'. Valid values: {', '.join(sorted(VALID_PRIORITIES))}"
            ), isError=True)]  # type: ignore[call-arg]

        issue = ISSUES[key]
        changes = []
        if new_status:
            old = issue["status"]
            issue["status"] = new_status
            changes.append(f"status: {old} → {new_status}")
        if new_assignee is not None:
            old = issue["assignee"] or "Unassigned"
            issue["assignee"] = new_assignee or None
            changes.append(f"assignee: {old} → {new_assignee or 'Unassigned'}")
        if new_priority:
            old = issue["priority"]
            issue["priority"] = new_priority
            changes.append(f"priority: {old} → {new_priority}")

        issue["updated"] = datetime.now(timezone.utc).isoformat()
        logger.info("Updated issue %s: %s", key, ", ".join(changes))
        return [TextContent(type="text", text=f"Updated {key}: {', '.join(changes)}")]

    # Unknown tool
    logger.warning("Unknown tool called: %s", name)
    return [TextContent(type="text", text=f"Unknown tool: {name}", isError=True)]  # type: ignore[call-arg]


# ── Resources ─────────────────────────────────────────────────────────────────

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="jira://projects",  # type: ignore[arg-type]
            name="Available Projects",
            description="List of Jira projects with name and lead. Read this to know valid project keys.",
            mimeType="application/json"
        ),
    ]


@server.read_resource()  # type: ignore[arg-type]
async def read_resource(uri: str) -> str:
    if str(uri) == "jira://projects":
        data = [
            {"key": k, "name": v["name"], "lead": v["lead"]}
            for k, v in PROJECTS.items()
        ]
        return json.dumps(data, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# ── Prompts ───────────────────────────────────────────────────────────────────

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="bug_analysis",
            description="Generate a structured bug analysis report from a Jira issue",
            arguments=[
                PromptArgument(name="issue_key", description="Jira issue key", required=True),
                PromptArgument(name="context", description="Additional context about the bug", required=False),
            ]
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    if name == "bug_analysis":
        key = (arguments or {}).get("issue_key", "")
        context = (arguments or {}).get("context", "")
        extra = f"\nAdditional context: {context}" if context else ""
        return GetPromptResult(
            description="Bug analysis report prompt",
            messages=[PromptMessage(
                role="user",
                content=TextContent(type="text", text=(
                    f"Analyze the Jira issue {key} and produce a structured bug report.{extra}\n\n"
                    f"Use the get_issue tool to fetch the issue details, then provide:\n"
                    f"1. Root cause hypothesis\n"
                    f"2. Affected components\n"
                    f"3. Severity assessment\n"
                    f"4. Recommended next steps"
                ))
            )]
        )
    raise ValueError(f"Unknown prompt: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("Starting jira-mcp-demo server")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
