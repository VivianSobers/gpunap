"""errors: calls in the wrong state, on a missing PID and on a process without CUDA must fail and
change nothing."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from gpunap import results
from gpunap.check import base

MB = 128
# (call, whether it should succeed in the state the previous calls leave)
SEQUENCE = [("checkpoint", False), ("restore", False), ("unlock", False), ("lock", True), ("lock", False),
            ("checkpoint", True), ("checkpoint", False), ("unlock", False), ("restore", True), ("unlock", True)]


def unused_pid(pid_max: int, proc: str = "/proc") -> int:
    pid = pid_max - 1
    while os.path.exists(os.path.join(proc, str(pid))):
        pid -= 1
    return pid


def verdict(seq: List[Tuple[str, bool, str]], final_state: str, check: str, rc, others: Dict[str, str],
            bystander_alive: bool) -> Tuple[str, Optional[str]]:
    for i, (op, want_ok, got) in enumerate(seq, 1):
        if want_ok and got != "OK":
            return results.INVALID, f"step {i}: {op} should have worked but returned {got}"
        if not want_ok and got == "OK":
            return results.HAZARD, f"step {i}: {op} in the wrong state succeeded"
    for name, got in others.items():
        if got == "OK":
            return results.HAZARD, f"{name} succeeded"
    if final_state != "running" or check != "ok" or rc != 0:
        return results.HAZARD, f"target left {final_state}, data {check}, exit {rc}"
    if not bystander_alive:
        return results.HAZARD, "the process without CUDA died"
    return results.OK, None


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        pid_max = int(open("/proc/sys/kernel/pid_max").read())
    except (OSError, ValueError):
        pid_max = 4194304
    missing = unused_pid(pid_max)
    others = {"missing_pid_state": ctx.call("state", missing, 30).get("result"),
              "missing_pid_lock": ctx.call("lock", missing, 30, 1000).get("result")}
    bystander = ctx.owned.start([ctx.python, "-c", "import time; time.sleep(600)", "--token", ctx.token], kind="no_cuda")
    for op in ("state", "lock", "checkpoint"):
        others[f"no_cuda_{op}"] = ctx.call(op, bystander.pid, 30, 1000 if op == "lock" else 0).get("result")
    bystander_alive = bystander.alive()
    try:
        t, s = base.start_target(ctx, "hold", "--mb", MB)
    except base.NotReady as e:
        return base.not_ready_result("errors", e, time.time() - t0)
    seq = []
    for op, want_ok in SEQUENCE:
        d = ctx.call(op, t.pid, 60, 1000 if op == "lock" else 0)
        seq.append((op, want_ok, d.get("result")))
        if want_ok and d.get("result") != "OK":
            break
        if op == "restore" and d.get("result") == "OK":
            t.resumed = True
    final_state = ctx.call("state", t.pid, 30).get("result")
    check = base.verify(s)
    rc = base.finish(s, t)
    v, why = verdict(seq, final_state, check, rc, others, bystander_alive)
    names = sorted({g for _, w, g in seq if not w} | set(others.values()))
    data = {"sequence": [{"op": op, "should_work": w, "result": g} for op, w, g in seq], "others": others,
            "final_state": final_state, "verify": check, "target_rc": rc, "missing_pid": missing}
    summary = why or "wrong-state, missing-PID and no-CUDA calls all failed without effect: " + ", ".join(names)
    return base.result("errors", v, summary, data, t0)


CHECK = base.Check("errors", base.DEFAULT, ("libcuda",), gpu_mb=MB + 600, host_mb=MB, run=run,
                   doc="Calls in the wrong state, on a missing PID and on a process without CUDA fail and change nothing.")
