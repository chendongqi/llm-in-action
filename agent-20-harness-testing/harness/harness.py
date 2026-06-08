"""Unified Harness entry point — combines all layers behind a single execute() API."""

from typing import Any

from .audit import ImmutableAuditLog
from .budget import BudgetExhaustedError, PermissionBudget
from .registry import ActionRegistry, PermissionLevel, PermissionError
from .rollback import RollbackCoordinator
from .sandbox import sanitise_input


class HumanApprovalRequired(Exception):
    """Raised when an IRREVERSIBLE action needs human sign-off."""

    def __init__(self, action_name: str, action_args: dict):
        self.action_name = action_name
        self.action_args = action_args
        super().__init__(
            f"Action '{action_name}' requires human approval before execution."
        )


class AgentHarness:
    """
    8-layer safety harness for agent tool calls.

    Layers applied on each execute() call:
      4 — Sandbox: sanitise string arguments
      2 — Registry: block unregistered actions
      3 — Budget: deduct cost, raise if exhausted
      5 — Checkpoint: raise HumanApprovalRequired for IRREVERSIBLE actions
      7 — Rollback: wrap WRITE/ADMIN in a transaction; restore on failure
      6 — Audit: append result to immutable log
    """

    def __init__(
        self,
        budget: int = 100,
        log_path: str = "/tmp/harness_audit.jsonl",
    ) -> None:
        self.registry = ActionRegistry()
        self.budget = PermissionBudget(total=budget)
        self.audit = ImmutableAuditLog(log_path=log_path)
        self.rollback = RollbackCoordinator()
        self._state: dict[str, Any] = {}  # shared mutable state for rollback demos

    # ── public API ─────────────────────────────────────────────────────────

    def execute(self, action_name: str, actor: str = "agent", **kwargs) -> Any:
        """
        Safe single-action execution through the full harness stack.
        Raises:
            PermissionError         — action not in registry
            BudgetExhaustedError    — insufficient budget
            HumanApprovalRequired   — IRREVERSIBLE action, needs approval
        """
        # Layer 4: sanitise string arguments
        for k, v in kwargs.items():
            if isinstance(v, str):
                _, flagged = sanitise_input(v)
                if flagged:
                    self.audit.log(action_name, "sandbox", k, "INJECTION_FLAGGED",
                                   {"value": v[:80]})

        # Layer 2: registry check
        action = self.registry.get(action_name)  # raises PermissionError if absent

        # Layer 3: budget check
        self.budget.spend(action_name, action.budget_cost)

        # Layer 5: human checkpoint for IRREVERSIBLE
        if action.level == PermissionLevel.IRREVERSIBLE:
            self.budget.refund(action_name, action.budget_cost)  # refund on rejection
            self.audit.log(action_name, actor, str(kwargs), "PENDING_APPROVAL")
            raise HumanApprovalRequired(action_name, dict(kwargs))

        # Layer 7 + execute
        if action.level in (PermissionLevel.WRITE, PermissionLevel.ADMIN):
            with self.rollback.transaction(self._state, action_name):
                result = action.handler(**kwargs)
        else:
            result = action.handler(**kwargs)

        # Layer 6: audit
        self.audit.log(action_name, actor, str(kwargs), "EXECUTED",
                       {"level": action.level.name})
        return result

    def approve_and_execute(
        self, action_name: str, actor: str = "human", **kwargs
    ) -> Any:
        """
        Execute an IRREVERSIBLE action after explicit human approval.
        Call this after catching HumanApprovalRequired.
        """
        action = self.registry.get(action_name)
        self.budget.spend(action_name, action.budget_cost)
        result = action.handler(**kwargs)
        self.audit.log(action_name, actor, str(kwargs), "EXECUTED_AFTER_APPROVAL",
                       {"level": action.level.name})
        return result
