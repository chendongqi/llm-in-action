"""Layer 6 — Immutable Audit Log: append-only hash-chained JSONL."""

import hashlib
import json
import time
from pathlib import Path


class ImmutableAuditLog:
    def __init__(self, log_path: str = "/tmp/harness_audit.jsonl") -> None:
        self._path = Path(log_path)
        self._last_hash = "GENESIS"
        self._path.write_text("")  # wipe on init for demo isolation

    def _hash(self, payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def log(
        self,
        action: str,
        actor: str,
        target: str,
        result: str,
        metadata: dict | None = None,
    ) -> str:
        entry: dict = {
            "ts":        time.strftime("%H:%M:%S"),
            "action":    action,
            "actor":     actor,
            "target":    target,
            "result":    result,
            "metadata":  metadata or {},
            "prev_hash": self._last_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry["hash"] = self._hash(entry_str + self._last_hash)
        self._last_hash = entry["hash"]

        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry["hash"]

    def verify_integrity(self) -> bool:
        prev = "GENESIS"
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            check = {k: v for k, v in e.items() if k != "hash"}
            expected = self._hash(json.dumps(check, sort_keys=True) + prev)
            if expected != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def tail(self, n: int = 10) -> list[dict]:
        lines = [l for l in self._path.read_text().splitlines() if l.strip()]
        return [json.loads(l) for l in lines[-n:]]

    def __len__(self) -> int:
        return sum(1 for l in self._path.read_text().splitlines() if l.strip())
