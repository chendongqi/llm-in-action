"""Shared fixtures for all harness test suites."""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harness import (
    AgentHarness,
    BudgetExhaustedError,
    HumanApprovalRequired,
    PermissionError,
    PermissionLevel,
    RegisteredAction,
)

# ── minimal mock handlers ──────────────────────────────────────────────────

_store: dict[str, str] = {}
_sent_reports: list[str] = []
_deleted: list[str] = []


def mock_read(key: str) -> str:
    return _store.get(key, f"{key}: (empty)")


def mock_write(key: str, value: str) -> str:
    _store[key] = value
    return f"written {key}={value!r}"


def mock_send(to: str, body: str) -> str:
    _sent_reports.append(f"{to}: {body}")
    return f"sent to {to}"


def mock_delete(key: str) -> str:
    _deleted.append(key)
    _store.pop(key, None)
    return f"deleted {key}"


def make_harness(budget: int = 100, log_suffix: str = "") -> AgentHarness:
    h = AgentHarness(budget=budget,
                     log_path=f"/tmp/harness_test{log_suffix}.jsonl")
    h.registry.register(RegisteredAction(
        "read",   PermissionLevel.READ,        1,  "Read a value",         mock_read))
    h.registry.register(RegisteredAction(
        "write",  PermissionLevel.WRITE,       3,  "Write a value",        mock_write))
    h.registry.register(RegisteredAction(
        "send",   PermissionLevel.ADMIN,        5,  "Send a report",        mock_send))
    h.registry.register(RegisteredAction(
        "delete", PermissionLevel.IRREVERSIBLE, 10, "Delete forever",       mock_delete))
    return h


@pytest.fixture(autouse=True)
def reset_store():
    """Reset shared mock state before each test."""
    _store.clear()
    _sent_reports.clear()
    _deleted.clear()
    _store["k1"] = "value1"
    _store["k2"] = "value2"
    yield


@pytest.fixture
def harness():
    return make_harness()


@pytest.fixture
def tight_harness():
    return make_harness(budget=5, log_suffix="_tight")
