"""child_context: a job whose child process has its own CUDA context.

FINDINGS: pausing the parent leaves the child running and holding its GPU memory; each process has
to be paused separately, and then both resume intact.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from gpunap import nvsmi, results
from gpunap.check import base

MB = 128
STEPS = ("pause parent", "pause child", "resume child", "resume parent")


def verdict(steps: List[Dict], check: str, rc, child_state: str) -> Tuple[str, str]:
    for name, d in zip(STEPS, steps):
        if not base.ok(d):
            return results.HAZARD, f"{name} returned {d.get('result')}"
    if check != "ok" or rc != 0:
        return results.HAZARD, f"data check {check}, exit {rc}"
    how = "left the child running" if child_state == "running" else f"left the child {child_state}"
    return results.OK, f"pausing the parent {how}; pausing each process separately worked, data intact"


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        t, s = base.start_target(ctx, "parent_child", "--mb", MB)
    except base.NotReady as e:
        return base.not_ready_result("child_context", e, time.time() - t0)
    line = s.first("CHILD")
    child = ctx.owned.adopt(int(line[1].split()[1]), kind="child")
    steps = [base.pause(ctx, t)]
    child_state = ctx.call("state", child.pid, 30).get("result")
    mem = {a["pid"]: a["used_mb"] for a in nvsmi.apps()}
    gpu_mb = {"parent": mem.get(t.pid, 0), "child": mem.get(child.pid, 0)}
    steps.append(base.pause(ctx, child))
    steps.append(base.resume(ctx, child))
    steps.append(base.resume(ctx, t))
    check = base.verify(s)
    rc = base.finish(s, t)
    v, summary = verdict(steps, check, rc, child_state)
    return base.result("child_context", v, summary, {"steps": dict(zip(STEPS, steps)), "child_state_after_parent_pause": child_state,
                                                     "gpu_mb_after_parent_pause": gpu_mb, "verify": check,
                                                     "target_rc": rc}, t0)


CHECK = base.Check("child_context", base.DEFAULT, ("libcuda",), gpu_mb=2 * MB + 1000, host_mb=2 * MB, run=run,
                   doc="A parent and a child process with separate CUDA contexts.")
