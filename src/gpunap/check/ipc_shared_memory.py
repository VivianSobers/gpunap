"""ipc_shared_memory: CUDA memory shared between processes (a CUDA tensor passed to a child with
torch.multiprocessing).

FINDINGS: on 595 the checkpoints failed after about 32 to 35 s and both processes died; on 580 the
child's restore returned INVALID_VALUE and it stayed stuck. NVIDIA lists this as unsupported before
driver 610 (external source).
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from gpunap import results
from gpunap.check import base

STEPS = ("pause child", "resume child", "pause parent", "resume parent")


def verdict(steps: Dict[str, Optional[Dict]], check: str, rc) -> Tuple[str, str]:
    refused = None
    for name in STEPS:
        d = steps.get(name)
        if d is None:
            continue
        if d.get("hung"):
            return results.HAZARD, f"{name} hung"
        if not base.ok(d):
            if name.startswith("resume"):
                return results.HAZARD, f"{name} returned {d.get('result')}"
            refused = refused or f"{name} returned {d.get('result')}"
    if check != "ok" or rc != 0:
        cause = f"{refused}; " if refused else ""
        return results.HAZARD, f"{cause}data check {check}, exit {rc}"
    if refused:
        return results.OK, f"{refused} (refused cleanly); both processes carried on"
    return results.OK, "shared CUDA memory survived pausing child and parent"


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        t, s = base.start_target(ctx, "torch_ipc", ready_timeout=180)
    except base.NotReady as e:
        return base.not_ready_result("ipc_shared_memory", e, time.time() - t0)
    child = ctx.owned.adopt(int(s.first("CHILD")[1].split()[1]), kind="ipc_child")
    steps: Dict[str, Optional[Dict]] = {}
    for who, target in (("child", child), ("parent", t)):
        steps[f"pause {who}"] = base.pause(ctx, target, limit_s=90)
        time.sleep(1.0)
        steps[f"resume {who}"] = base.resume(ctx, target, limit_s=90) if base.ok(steps[f"pause {who}"]) else None
        if steps[f"resume {who}"] is not None and not base.ok(steps[f"resume {who}"]):
            break
    check = base.verify(s, timeout=90) if t.alive() else "no reply"
    rc = base.finish(s, t, timeout=60) if t.alive() else t.popen.wait()
    v, summary = verdict(steps, check, rc)
    return base.result("ipc_shared_memory", v, summary, {"steps": steps, "verify": check, "target_rc": rc,
                                                         "child_alive_at_end": child.alive(),
                                                         "output": s.output(12)}, t0)


CHECK = base.Check("ipc_shared_memory", base.DEFAULT, ("libcuda", "torch"), gpu_mb=2000, host_mb=600, run=run,
                   doc="CUDA memory shared between a parent and a child process.")
