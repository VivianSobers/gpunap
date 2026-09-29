"""Compare a gpunap check report with the expectations table.

    python tests/gpu/compare.py REPORT.json [--expectations PATH] [--allow-skipped]

Prints one row per check and exits 1 if any verdict differs from the table, a check is missing or
skipped, the report is partial, or GPU memory in use did not return to within 100 MB of where it
started.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
LEFTOVER_MB = 100


def branch(driver: Optional[str]) -> Optional[str]:
    return driver.split(".")[0] if driver else None


def compare(report: Dict, expect: Dict, allow_skipped: bool = False) -> Tuple[List[Tuple], List[str]]:
    h = report.get("header", {})
    br = branch((h.get("system") or {}).get("driver"))
    checks = expect["checks"]
    if not any(br in e for e in checks.values()):
        return [], [f"no expectations for driver branch {br}"]
    got = {r["name"]: r["verdict"] for r in report.get("results", [])}
    rows, problems = [], []
    for name, e in checks.items():
        want, have = e.get(br), got.get(name)
        if have is None:
            rows.append((name, want, "-", "missing"))
            problems.append(f"{name}: not in the report")
        elif have == "skipped":
            rows.append((name, want, have, "skipped"))
            if not allow_skipped:
                problems.append(f"{name}: skipped")
        elif have == want:
            rows.append((name, want, have, "match"))
        else:
            rows.append((name, want, have, "MISMATCH"))
            problems.append(f"{name}: expected {want} on {br}, got {have}")
    before, after = h.get("gpu_memory_used_mb_before"), h.get("gpu_memory_used_mb_after")
    if isinstance(before, int) and isinstance(after, int) and after - before > LEFTOVER_MB:
        problems.append(f"GPU memory used went from {before} MB to {after} MB (more than {LEFTOVER_MB} MB)")
    if report.get("partial"):
        problems.append("the report is partial")
    return rows, problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--expectations", default=os.path.join(HERE, "expectations.json"))
    p.add_argument("--allow-skipped", action="store_true")
    a = p.parse_args(argv)
    rows, problems = compare(json.load(open(a.report)), json.load(open(a.expectations)), a.allow_skipped)
    for name, want, have, status in rows:
        print(f"{name:<18} expected {str(want):<8} got {have:<8} {status}")
    for x in problems:
        print("PROBLEM:", x)
    print("all checks match" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
