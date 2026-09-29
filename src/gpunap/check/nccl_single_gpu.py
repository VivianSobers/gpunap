"""nccl_single_gpu: pausing a single-GPU job with an NCCL process group (what torchrun and
Accelerate set up).

FINDINGS: on 595 the checkpoint fails and the job aborts; on 580 it pauses cleanly. Also records the
pt_nccl_* thread names, the detection signal from FINDINGS.
"""
from __future__ import annotations

import signal
import time
from typing import Dict, List, Optional, Tuple

from gpunap import procinfo, results
from gpunap.check import base

CYCLES = 3
NCCL_THREADS = ("pt_nccl_watchdg", "pt_nccl_heartbt")


def nccl_threads(names: Optional[List[str]]) -> List[str]:
    return sorted(n for n in (names or []) if n in NCCL_THREADS)


def verdict(cycles: List[Dict], rc) -> Tuple[str, str]:
    refused = None
    for i, c in enumerate(cycles, 1):
        if c["pause"].get("hung"):
            return results.HAZARD, f"cycle {i}: pause hung"
        if not base.ok(c["pause"]):
            refused = refused or c["pause"].get("result")
        elif not base.ok(c["resume"]):
            return results.HAZARD, f"cycle {i}: resume returned {(c['resume'] or {}).get('result')}"
    if rc != 0:
        name = f" ({signal.Signals(-rc).name})" if isinstance(rc, int) and rc < 0 else ""
        cause = f" after the checkpoint returned {refused}" if refused else ""
        return results.HAZARD, f"the job exited {rc}{name}{cause}"
    bad = next((c["verify"] for c in cycles if c["verify"] != "ok"), None)
    if bad:
        return results.HAZARD, f"collectives after resume: {bad}"
    if refused:
        return results.OK, f"pause refused ({refused}); the job carried on"
    return results.OK, f"{len(cycles)} pause and resume cycles, collectives kept working"


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        t, s = base.start_target(ctx, "torch_nccl", ready_timeout=180)
    except base.NotReady as e:
        return base.not_ready_result("nccl_single_gpu", e, time.time() - t0)
    threads = procinfo.thread_names(t.pid)
    cycles = []
    for _ in range(CYCLES):
        c = {"pause": base.pause(ctx, t, limit_s=90)}
        time.sleep(3.0)
        c["resume"] = base.resume(ctx, t) if base.ok(c["pause"]) and t.alive() else None
        c["verify"] = base.verify(s, timeout=60) if t.alive() else "no reply"
        cycles.append(c)
        if not t.alive() or c["verify"] != "ok":
            break
    rc = base.finish(s, t) if t.alive() else t.popen.wait()
    v, summary = verdict(cycles, rc)
    found = nccl_threads(threads)
    summary += f"; NCCL threads seen: {', '.join(found) or 'none'}"
    return base.result("nccl_single_gpu", v, summary, {"cycles": cycles, "target_rc": rc, "threads": threads,
                                                       "nccl_threads": found, "output": s.output(12)}, t0)


CHECK = base.Check("nccl_single_gpu", base.DEFAULT, ("libcuda", "torch"), gpu_mb=1500, host_mb=600, run=run,
                   doc="Pausing a single-GPU job with an NCCL process group; pt_nccl_* thread names.")
