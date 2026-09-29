"""roundtrip: pause and resume a 256 MB process three times; its GPU data must survive."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from gpunap import results
from gpunap.check import base

MB, CYCLES = 256, 3


def verdict(cycles: List[Dict], rc) -> Tuple[str, Optional[str]]:
    problem = base.cycle_problem(cycles)
    if problem:
        return results.HAZARD, problem
    if rc != 0:
        return results.HAZARD, f"target exited with {rc}"
    return results.OK, None


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        t, s = base.start_target(ctx, "hold", "--mb", MB)
    except base.NotReady as e:
        return base.not_ready_result("roundtrip", e, time.time() - t0)
    cycles = []
    for _ in range(CYCLES):
        c = {"pause": base.pause(ctx, t)}
        time.sleep(0.5)
        c["resume"] = base.resume(ctx, t) if base.ok(c["pause"]) else {"result": "not attempted"}
        c["verify"] = base.verify(s)
        cycles.append(c)
        if c["verify"] != "ok":
            break
    rc = base.finish(s, t)
    v, why = verdict(cycles, rc)
    data = {"mb": MB, "cycles": cycles, "target_rc": rc}
    if v == results.OK:
        ck = [base.call_seconds(c["pause"], "checkpoint") for c in cycles]
        rs = [base.call_seconds(c["resume"], "restore") for c in cycles]
        summary = f"{len(cycles)} cycles on {MB} MB, data intact; checkpoint {min(ck):.2f}-{max(ck):.2f} s, restore {min(rs):.2f}-{max(rs):.2f} s"
    else:
        summary = why
    return base.result("roundtrip", v, summary, data, t0)


CHECK = base.Check("roundtrip", base.DEFAULT, ("libcuda",), gpu_mb=MB + 600, host_mb=MB, run=run,
                   doc="Pause and resume a 256 MB process three times; its GPU data must survive.")
