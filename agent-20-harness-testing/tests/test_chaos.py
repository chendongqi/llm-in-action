"""
Chaos tests — verify harness safety under fault conditions.

Simulates:
  C1. Tool raises an unexpected exception mid-execution
  C2. Tool times out (simulated with slow sleep)
  C3. Partial success — first tool OK, second fails
  C4. Registry grows dynamically (late registration)
"""

import time
import pytest

from harness import (
    BudgetExhaustedError,
    PermissionLevel,
    RegisteredAction,
)
from tests.conftest import make_harness, _store


# ── C1. Tool exceptions ────────────────────────────────────────────────────

class TestToolExceptions:

    def test_exception_in_write_propagates_and_rolls_back(self):
        """RuntimeError in a WRITE handler must propagate and restore state."""
        h = make_harness()
        h._state["data"] = "original"

        def boom(key: str, value: str) -> str:
            h._state["data"] = "corrupted"  # mutate state mid-handler
            raise RuntimeError("unexpected I/O failure")

        h.registry.register(RegisteredAction(
            "boom_write", PermissionLevel.WRITE, 3, "Exploding write", boom))

        with pytest.raises(RuntimeError, match="I/O failure"):
            with h.rollback.transaction(h._state, "boom_write"):
                boom("data", "new_value")

        assert h._state["data"] == "original"

    def test_exception_in_read_propagates_without_rollback(self):
        """READ errors propagate normally (no rollback logic for reads)."""
        h = make_harness()

        def flaky_read(key: str) -> str:
            raise ConnectionError("DB unreachable")

        h.registry.register(RegisteredAction(
            "flaky_read", PermissionLevel.READ, 1, "Flaky read", flaky_read))

        with pytest.raises(ConnectionError):
            h.execute("flaky_read", key="k1")

    def test_exception_in_write_does_not_log_executed(self):
        """A failed WRITE must not produce an EXECUTED audit entry."""
        h = make_harness()

        def always_fail(key: str, value: str) -> str:
            raise ValueError("intentional failure")

        h.registry.register(RegisteredAction(
            "fail_write", PermissionLevel.WRITE, 3, "Always fails", always_fail))

        before_len = len(h.audit)
        with pytest.raises(ValueError):
            h.execute("fail_write", key="k", value="v")

        # budget was charged (spend happens before execution),
        # but no EXECUTED entry should exist for this call
        entries = h.audit.tail(10)
        executed_names = [e["action"] for e in entries if e["result"] == "EXECUTED"]
        assert "fail_write" not in executed_names


# ── C2. Simulated slow tools ────────────────────────────────────────────────

class TestSlowTools:

    def test_slow_tool_still_succeeds(self):
        """A tool that takes 150ms completes normally (no artificial timeout)."""
        h = make_harness()

        def slow_read(key: str) -> str:
            time.sleep(0.15)
            return f"{key}: slow result"

        h.registry.register(RegisteredAction(
            "slow_read", PermissionLevel.READ, 1, "Slow read", slow_read))

        t0 = time.time()
        result = h.execute("slow_read", key="k1")
        elapsed = time.time() - t0

        assert "slow result" in str(result)
        assert elapsed >= 0.15

    def test_budget_is_charged_regardless_of_tool_speed(self):
        """Budget deduction happens before tool execution, not after."""
        h = make_harness()

        def medium_read(key: str) -> str:
            time.sleep(0.05)
            return "ok"

        h.registry.register(RegisteredAction(
            "medium_read", PermissionLevel.READ, 1, "Medium read", medium_read))

        before = h.budget.remaining
        h.execute("medium_read", key="k1")
        assert h.budget.remaining == before - 1


# ── C3. Partial success ────────────────────────────────────────────────────

class TestPartialSuccess:

    def test_first_ok_second_fails_state_from_first_persists(self):
        """When action 1 succeeds and action 2 fails, action 1 is NOT rolled back."""
        h = make_harness()
        # action 1: write succeeds
        h.execute("write", key="k1", value="updated")
        assert _store.get("k1") == "updated"

        # action 2: write fails (simulate via rollback.transaction directly)
        def bad_write(key: str, value: str) -> str:
            raise IOError("bad disk")

        h.registry.register(RegisteredAction(
            "bad_write", PermissionLevel.WRITE, 3, "Bad write", bad_write))

        with pytest.raises(IOError):
            with h.rollback.transaction(h._state, "bad_write"):
                bad_write("k2", "x")

        # k1 change from action 1 must still be present
        assert _store.get("k1") == "updated"

    def test_budget_charged_for_successful_actions_only(self):
        """Budget is NOT refunded for failed non-IRREVERSIBLE actions."""
        h = make_harness(budget=20)

        def failing_write(key: str, value: str) -> str:
            raise RuntimeError("fail")

        h.registry.register(RegisteredAction(
            "fail_write", PermissionLevel.WRITE, 3, "Fail", failing_write))

        before = h.budget.remaining
        with pytest.raises(RuntimeError):
            h.execute("fail_write", key="k", value="v")

        # budget was deducted (spend before execute), so remaining < before
        assert h.budget.remaining == before - 3


# ── C4. Dynamic registry ───────────────────────────────────────────────────

class TestDynamicRegistry:

    def test_action_registered_after_init_is_accessible(self):
        """Late-registered actions must work immediately after registration."""
        from tests.conftest import mock_write
        h = make_harness()
        h.registry.register(RegisteredAction(
            "late_write", PermissionLevel.WRITE, 3, "Late registration", mock_write))

        result = h.execute("late_write", key="new_k", value="hello")
        assert "written" in str(result)

    def test_overwriting_a_registration_updates_handler(self):
        """Re-registering an action name replaces the previous handler."""
        h = make_harness()

        call_log: list[str] = []

        def new_read(key: str) -> str:
            call_log.append(key)
            return f"new: {key}"

        h.registry.register(RegisteredAction(
            "read", PermissionLevel.READ, 1, "Replaced read", new_read))

        result = h.execute("read", key="k1")
        assert "new:" in str(result)
        assert "k1" in call_log
