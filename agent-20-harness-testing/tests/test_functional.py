"""
Functional tests — verify each harness layer behaves as specified.

Each test targets exactly one behaviour: if it fails, you know precisely
which layer broke.
"""

import pytest

from harness import (
    AgentHarness,
    BudgetExhaustedError,
    HumanApprovalRequired,
    PermissionError,
    PermissionLevel,
    RegisteredAction,
)
from tests.conftest import make_harness, mock_write


# ── Layer 2: Action Registry ───────────────────────────────────────────────

class TestActionRegistry:

    def test_unregistered_action_is_blocked(self, harness):
        """Any action not in the registry must raise PermissionError."""
        with pytest.raises(PermissionError, match="not in registry"):
            harness.execute("delete_all_data")

    def test_registered_read_action_executes(self, harness):
        """A registered READ action runs and returns output."""
        result = harness.execute("read", key="k1")
        assert "value1" in str(result)

    def test_registered_actions_are_listed(self, harness):
        """registry.names() returns all registered action names."""
        names = harness.registry.names()
        assert set(names) == {"read", "write", "send", "delete"}

    def test_unregistered_action_does_not_touch_budget(self, harness):
        """Blocked action must not consume budget."""
        before = harness.budget.remaining
        with pytest.raises(PermissionError):
            harness.execute("ghost_action")
        assert harness.budget.remaining == before


# ── Layer 3: Permission Budget ──────────────────────────────────────────────

class TestPermissionBudget:

    def test_budget_decreases_by_action_cost(self, harness):
        """Each execute() deducts exactly the registered budget_cost."""
        before = harness.budget.remaining
        harness.execute("read", key="k1")      # cost=1
        assert harness.budget.remaining == before - 1

        harness.execute("write", key="k1", value="v")  # cost=3
        assert harness.budget.remaining == before - 4

    def test_budget_exhaustion_blocks_execution(self, tight_harness):
        """BudgetExhaustedError is raised when remaining budget < cost."""
        # budget=5; write costs 3 → first OK, second fails (5-3=2 < 3)
        tight_harness.execute("write", key="k1", value="x")
        with pytest.raises(BudgetExhaustedError, match="Budget exhausted"):
            tight_harness.execute("write", key="k2", value="x")

    def test_read_actions_are_cheap(self, harness):
        """READ cost (1) must not exceed 2 budget units."""
        before = harness.budget.remaining
        harness.execute("read", key="k1")
        assert harness.budget.remaining >= before - 2

    def test_budget_summary_reflects_spending(self, harness):
        """budget.summary() shows correct remaining and spent count."""
        harness.execute("read", key="k1")
        harness.execute("write", key="k1", value="x")
        s = harness.budget.summary()
        assert "remaining" in s
        assert "2 entries" in s   # 2 spends logged


# ── Layer 5: Human Checkpoint ──────────────────────────────────────────────

class TestHumanCheckpoint:

    def test_irreversible_action_raises_approval_required(self, harness):
        """IRREVERSIBLE actions must not execute; raise HumanApprovalRequired."""
        with pytest.raises(HumanApprovalRequired) as exc_info:
            harness.execute("delete", key="k1")
        assert exc_info.value.action_name == "delete"

    def test_irreversible_action_does_not_execute_before_approval(self, harness):
        """The action handler must NOT run until approve_and_execute() is called."""
        from tests.conftest import _deleted
        try:
            harness.execute("delete", key="k1")
        except HumanApprovalRequired:
            pass
        assert "k1" not in _deleted   # handler never ran

    def test_budget_refunded_when_irreversible_intercepted(self, harness):
        """Budget spent before checkpoint must be refunded on interception."""
        before = harness.budget.remaining
        try:
            harness.execute("delete", key="k1")
        except HumanApprovalRequired:
            pass
        assert harness.budget.remaining == before   # net change = 0

    def test_approve_and_execute_runs_the_action(self, harness):
        """approve_and_execute() actually calls the handler."""
        from tests.conftest import _deleted
        try:
            harness.execute("delete", key="k1")
        except HumanApprovalRequired:
            harness.approve_and_execute("delete", key="k1")
        assert "k1" in _deleted


# ── Layer 7: Rollback ──────────────────────────────────────────────────────

class TestRollback:

    def test_failed_write_does_not_persist(self, harness):
        """State must be restored when a WRITE action raises mid-execution."""
        def _fail_write(key: str, value: str) -> str:
            harness._state["key"] = value   # modify state
            raise RuntimeError("disk full")

        harness.registry.register(RegisteredAction(
            "write_fail", PermissionLevel.WRITE, 3, "Failing write", _fail_write))

        harness._state["key"] = "original"
        with pytest.raises(RuntimeError):
            with harness.rollback.transaction(harness._state, "test_write"):
                _fail_write("key", "corrupted")

        assert harness._state.get("key") == "original"

    def test_successful_write_persists(self, harness):
        """State must persist normally when a WRITE action succeeds."""
        harness._state["x"] = "before"
        with harness.rollback.transaction(harness._state, "ok_write"):
            harness._state["x"] = "after"
        assert harness._state["x"] == "after"

    def test_rollback_depth_tracks_transactions(self, harness):
        """rollback.depth increases per started transaction, not on failure."""
        assert harness.rollback.depth == 0
        with harness.rollback.transaction(harness._state, "tx1"):
            assert harness.rollback.depth == 1
        assert harness.rollback.depth == 1   # committed, snapshot retained


# ── Layer 6: Audit Log ─────────────────────────────────────────────────────

class TestAuditLog:

    def test_executed_action_appears_in_audit(self, harness):
        """Every successful execute() must produce an audit entry."""
        before = len(harness.audit)
        harness.execute("read", key="k1")
        assert len(harness.audit) == before + 1

    def test_blocked_action_produces_no_audit_entry(self, harness):
        """Unregistered actions (blocked before registry) produce no log."""
        before = len(harness.audit)
        with pytest.raises(PermissionError):
            harness.execute("ghost")
        assert len(harness.audit) == before

    def test_audit_log_integrity_passes_after_normal_use(self, harness):
        """verify_integrity() must return True after normal operations."""
        harness.execute("read",  key="k1")
        harness.execute("write", key="k1", value="updated")
        assert harness.audit.verify_integrity() is True

    def test_audit_entry_contains_correct_result(self, harness):
        """Audit entry for a READ action must record result=EXECUTED."""
        harness.execute("read", key="k1")
        entry = harness.audit.tail(1)[0]
        assert entry["action"] == "read"
        assert entry["result"] == "EXECUTED"
