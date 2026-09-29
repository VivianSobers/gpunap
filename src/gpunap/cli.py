"""gpunap command line.

    gpunap check [--full] [--only A,B] [--label NAME] [--out DIR] [--list]
    gpunap sysinfo
    gpunap --version

Exit codes for check: 0 when every check ran to a verdict (hazards included), 1 when gpunap itself
failed in a check, 2 for a usage error or a refused run, 130 when interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

import gpunap
from gpunap import results, sysinfo
from gpunap.check import registry, runner
from gpunap.driver import Driver


def _line(r: results.CheckResult) -> str:
    return f"{r.name:<18} {r.verdict:<8} {r.seconds:6.1f} s  {r.summary}"


def cmd_list() -> int:
    for c in registry.CHECKS:
        print(f"{c.name:<18} {c.mode:<8} {c.doc}")
    return 0


def cmd_check(a) -> int:
    if a.list:
        return cmd_list()
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    try:
        checks = runner.select(registry.CHECKS, only)
    except ValueError as e:
        print(f"gpunap: {e}", file=sys.stderr)
        return 2
    ctx = runner.new_context(a.label, a.full)
    if a.full and ctx.module_kind == "open":
        print("gpunap: --full is refused on the open kernel module: killing a resumed process there can hang "
              "the machine (NVIDIA issue #53). Run without --full.", file=sys.stderr)
        return 2
    when = time.time()
    print(f"gpunap {gpunap.__version__}: {len(checks)} checks{' (--full)' if a.full else ''}", flush=True)
    report = runner.run(checks, ctx, on_result=lambda r: print(_line(r), flush=True))
    path = results.write(report, a.out, a.label, when)
    counts = report.to_dict()["counts"]
    print(", ".join(f"{n} {v}" for v, n in counts.items() if n) or "no results")
    if report.partial:
        print("interrupted: the report is partial")
    print(f"report: {path}")
    if report.partial:
        return 130
    return 1 if counts[results.ERROR] else 0


def cmd_sysinfo() -> int:
    print(json.dumps(sysinfo.collect(Driver()), indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="gpunap", description="Check how CUDA checkpoint and restore behave on "
                                                           "this machine's driver.")
    p.add_argument("--version", action="version", version=f"gpunap {gpunap.__version__}")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="run the checks and write a JSON report")
    c.add_argument("--full", action="store_true", help="also run the heavy checks (needs an idle GPU)")
    c.add_argument("--only", help="comma-separated check names")
    c.add_argument("--label", default="", help="machine label recorded in the report and file name")
    c.add_argument("--out", default=".", help="directory for the report (default: current directory)")
    c.add_argument("--list", action="store_true", help="list the checks and exit")
    sub.add_parser("sysinfo", help="print the machine facts a report records")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "sysinfo":
        return cmd_sysinfo()
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
