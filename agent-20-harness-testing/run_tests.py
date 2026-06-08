"""
Harness Test Suite Runner

Three test categories, 28 tests total:
  - Functional  (16 tests): each layer's basic behaviour
  - Adversarial  (8 tests): deliberate malicious inputs
  - Chaos        (9 tests): fault injection and edge cases

Run:
    conda activate dev_base
    pip install pytest -q   # if not already installed
    python run_tests.py
"""

import subprocess
import sys
import time


CATEGORIES = [
    ("Functional  (Layer 1–7 basic behaviour)", "tests/test_functional.py"),
    ("Adversarial (injection / escalation)",    "tests/test_adversarial.py"),
    ("Chaos       (fault injection / partial)", "tests/test_chaos.py"),
]

WIDTH = 70


def run_category(label: str, path: str) -> dict:
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    lines = result.stdout.splitlines()
    passed = sum(1 for l in lines if " PASSED" in l)
    failed = sum(1 for l in lines if " FAILED" in l or " ERROR" in l)
    total  = passed + failed

    return {
        "label":   label,
        "path":    path,
        "passed":  passed,
        "failed":  failed,
        "total":   total,
        "elapsed": elapsed,
        "output":  result.stdout,
        "rc":      result.returncode,
    }


def main() -> None:
    print("\n" + "=" * WIDTH)
    print("Agent Harness — Test Suite")
    print("=" * WIDTH)

    results = []
    for label, path in CATEGORIES:
        print(f"\nRunning: {label}")
        print("-" * WIDTH)
        r = run_category(label, path)
        results.append(r)

        # print individual test lines
        for line in r["output"].splitlines():
            if "PASSED" in line or "FAILED" in line or "ERROR" in line:
                status = "✓" if "PASSED" in line else "✗"
                # extract test name from pytest verbose output
                parts = line.split("::")
                name = parts[-1].split(" ")[0] if len(parts) > 1 else line[:60]
                mark = "  ✓" if "PASSED" in line else "  ✗"
                print(f"{mark} {name}")

        color = "PASS" if r["failed"] == 0 else "FAIL"
        print(f"\n  → {color}: {r['passed']}/{r['total']} passed  ({r['elapsed']:.2f}s)")

        # print failure details if any
        if r["failed"] > 0:
            print("\n  Failure details:")
            in_failure = False
            for line in r["output"].splitlines():
                if line.startswith("FAILED") or "_ FAILED _" in line:
                    in_failure = True
                if in_failure:
                    print("  " + line)
                if line.startswith("=") and in_failure:
                    in_failure = False

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * WIDTH)
    print("Summary")
    print("=" * WIDTH)

    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_tests  = sum(r["total"]  for r in results)
    total_time   = sum(r["elapsed"] for r in results)

    for r in results:
        bar_len = 30
        p = int(r["passed"] / r["total"] * bar_len) if r["total"] else 0
        bar = "█" * p + "░" * (bar_len - p)
        status = "PASS" if r["failed"] == 0 else "FAIL"
        print(f"  {r['label']:<42}  [{bar}]  {r['passed']:>2}/{r['total']:>2}  {status}")

    print(f"\n  {'Total':<42}  {total_passed:>3}/{total_tests:>3} tests passed"
          f"  ({total_time:.2f}s)")

    overall = "ALL TESTS PASS ✓" if total_failed == 0 else f"{total_failed} TEST(S) FAILED ✗"
    print(f"\n  {overall}")
    print("=" * WIDTH + "\n")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
