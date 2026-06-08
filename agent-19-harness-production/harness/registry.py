"""Layer 2 — Action Space Registry: whitelist-based permission system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PermissionLevel(Enum):
    READ        = 1
    WRITE       = 2
    ADMIN       = 3
    IRREVERSIBLE = 4


class PermissionError(Exception):
    pass


@dataclass
class RegisteredAction:
    name: str
    level: PermissionLevel
    budget_cost: int
    description: str
    handler: Any


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, RegisteredAction] = {}

    def register(self, action: RegisteredAction) -> None:
        self._actions[action.name] = action

    def get(self, name: str) -> RegisteredAction:
        if name not in self._actions:
            raise PermissionError(
                f"Action '{name}' not in registry. Execution blocked. "
                f"Registered: {list(self._actions)}"
            )
        return self._actions[name]

    def is_allowed(self, name: str) -> bool:
        return name in self._actions

    def names(self) -> list[str]:
        return list(self._actions)
