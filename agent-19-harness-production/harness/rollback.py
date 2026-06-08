"""Layer 7 — Rollback Coordinator: snapshot-based write transaction manager."""

import copy
from contextlib import contextmanager
from typing import Any


class RollbackCoordinator:
    def __init__(self) -> None:
        self._snapshots: list[dict] = []

    @contextmanager
    def transaction(self, state: dict[str, Any], op_name: str):
        """Snapshot state before a write. Restore automatically on exception."""
        snapshot = copy.deepcopy(state)
        self._snapshots.append({"op": op_name, "snapshot": snapshot})
        try:
            yield state
            # success: snapshot retained for manual rollback if needed
        except Exception:
            state.clear()
            state.update(snapshot)
            self._snapshots.pop()
            raise

    def rollback_last(self, state: dict[str, Any]) -> str | None:
        """Manually revert the last committed transaction."""
        if not self._snapshots:
            return None
        entry = self._snapshots.pop()
        state.clear()
        state.update(entry["snapshot"])
        return entry["op"]

    @property
    def depth(self) -> int:
        return len(self._snapshots)
