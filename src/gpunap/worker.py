"""Run driver calls in a separate process with a time limit.

Some checkpoint calls never return, for example a lock on a stopped process. Running each call in
its own process means a hung call can be killed without taking the caller with it (FINDINGS design
rule 6).

    python -m gpunap.worker <op> <pid> [--timeout-ms N]    one call, one JSON line on stdout
    python -m gpunap.worker <op> <pid> --announce           first a CALLING line, right before the call
    python -m gpunap.worker reserve <pid> <mb>             claim-before-restore helper
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

from gpunap.driver import Driver

OPS = ("lock", "checkpoint", "restore", "unlock", "state", "pause", "resume")
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def child_env() -> dict:
    env = dict(os.environ)
    paths = [PKG_PARENT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_json(argv: List[str], limit_s: float, env: Optional[dict] = None) -> dict:
    """Run argv, return its last JSON line as a dict. A run past limit_s is killed and reported as
    hung."""
    t = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=limit_s, env=env)
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {"result": "hung", "hung": True, "limit_s": limit_s, "wall_s": round(time.time() - t, 3),
                "stderr": err[-2000:]}
    rows = [line for line in p.stdout.splitlines() if line.startswith("{")]
    if not rows:
        return {"result": "no-output", "hung": False, "rc": p.returncode, "wall_s": round(time.time() - t, 3),
                "stderr": p.stderr[-2000:]}
    d = json.loads(rows[-1])
    d.update(hung=False, wall_s=round(time.time() - t, 3))
    return d


def call(op: str, pid: int, limit_s: float, timeout_ms: int = 0, python: Optional[str] = None) -> dict:
    """One driver call in a worker process. The result dict has op, result, code, start, end, hung."""
    argv = [python or sys.executable, "-m", "gpunap.worker", op, str(pid)]
    if timeout_ms:
        argv += ["--timeout-ms", str(timeout_ms)]
    d = run_json(argv, limit_s, env=child_env())
    d.setdefault("op", op)
    return d


def _reserve(pid: int, mb: int, driver: Driver) -> int:
    """Hold mb of GPU memory, then on GO free it and restore pid at once (FINDINGS design rule 1)."""
    lib = driver.lib()
    dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
    lib.cuDeviceGet(ctypes.byref(dev), 0)
    lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    lib.cuCtxSetCurrent(ctx)
    alloc = driver.error_name(lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(mb * 2 ** 20)))
    print(json.dumps({"event": "READY", "alloc": alloc}), flush=True)
    line = sys.stdin.readline().strip()
    if line != "GO":
        print(json.dumps({"result": "aborted", "alloc": alloc, "got": line}), flush=True)
        return 1
    t = time.time()
    free = driver.error_name(lib.cuMemFree_v2(ptr)) if alloc == "OK" else "not-allocated"
    calls = [driver.restore(pid)]
    gap_ms = round((calls[0].start - t) * 1000, 3)
    if calls[0].ok:
        calls.append(driver.unlock(pid))
    print(json.dumps({"op": "reserve", "alloc": alloc, "free": free, "free_to_restore_ms": gap_ms,
                      "calls": [c.to_dict() for c in calls],
                      "result": next((c.result for c in calls if not c.ok), "OK")}), flush=True)
    return 0


def main(argv: Optional[List[str]] = None, driver: Optional[Driver] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpunap.worker")
    p.add_argument("op", choices=OPS + ("reserve",))
    p.add_argument("pid", type=int)
    p.add_argument("mb", type=int, nargs="?", default=0)
    p.add_argument("--timeout-ms", type=int, default=0)
    p.add_argument("--announce", action="store_true")
    a = p.parse_args(argv)
    d = driver or Driver()
    if a.op == "reserve":
        return _reserve(a.pid, a.mb, d)
    if a.announce:  # load and initialise the library first, so the line marks the call itself
        d.lib()
        print(json.dumps({"event": "CALLING", "op": a.op, "time": time.time()}), flush=True)
    if a.op in ("pause", "resume"):
        calls = d.pause(a.pid, a.timeout_ms) if a.op == "pause" else d.resume(a.pid)
        failed = next((c for c in calls if not c.ok), None)
        out = {"op": a.op, "calls": [c.to_dict() for c in calls],
               "result": failed.result if failed else "OK", "code": failed.code if failed else 0,
               "start": calls[0].start, "end": calls[-1].end}
    else:
        c = d.lock(a.pid, a.timeout_ms) if a.op == "lock" else getattr(d, a.op)(a.pid)
        out = c.to_dict()
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
