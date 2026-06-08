"""Layer 4 — Execution Sandbox: input sanitisation + subprocess isolation."""

import re
import subprocess


INJECTION_PATTERN = re.compile(
    r"(ignore.*(previous|above|prior)|forget.*instruction|"
    r"you are now|act as|jailbreak|bypass|"
    r"override.*system|system.*override|"          # both orders
    r"</s>|\n\n###|###\s*system|<\|im_start\|>|system prompt)",  # real newline
    re.IGNORECASE,
)


def sanitise_input(text: str) -> tuple[str, bool]:
    """Return (text, was_flagged). Flags but does not block by default."""
    flagged = bool(INJECTION_PATTERN.search(text))
    return text, flagged


def sandboxed_eval(expression: str, timeout_s: int = 2) -> str:
    """Evaluate arithmetic in a subprocess. Rejects non-arithmetic characters."""
    allowed = set("0123456789 +-*/().")
    if not all(c in allowed for c in expression):
        return f"Rejected: illegal characters in expression"
    try:
        result = subprocess.run(
            ["python3", "-c", f"print(eval('{expression}'))"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as exc:
        return f"Error: {exc}"
