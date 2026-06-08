"""Layer 3 — Permission Budget: rate-limit agent actions by cost."""


class BudgetExhaustedError(Exception):
    pass


class PermissionBudget:
    def __init__(self, total: int = 100) -> None:
        self.total = total
        self.remaining = total
        self._ledger: list[dict] = []

    def spend(self, action_name: str, cost: int) -> None:
        if self.remaining < cost:
            raise BudgetExhaustedError(
                f"Budget exhausted: need {cost}, remaining {self.remaining} "
                f"(total {self.total}). Agent paused for human review."
            )
        self.remaining -= cost
        self._ledger.append({
            "action": action_name,
            "cost": cost,
            "remaining": self.remaining,
        })

    def refund(self, action_name: str, cost: int) -> None:
        self.remaining = min(self.total, self.remaining + cost)
        self._ledger.append({
            "action": f"REFUND:{action_name}",
            "cost": -cost,
            "remaining": self.remaining,
        })

    def reset(self, new_total: int | None = None) -> None:
        self.total = new_total or self.total
        self.remaining = self.total

    @property
    def usage_ratio(self) -> float:
        return (self.total - self.remaining) / self.total

    def summary(self) -> str:
        spent = self.total - self.remaining
        return (
            f"Budget: {self.remaining}/{self.total} remaining "
            f"(spent {spent} across {len(self._ledger)} entries)"
        )
