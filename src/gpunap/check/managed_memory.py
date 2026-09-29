"""managed_memory: pausing a process that holds CUDA managed (unified) memory.

FINDINGS: the checkpoint is refused with NOT_SUPPORTED. On 595 the process was broken afterwards; on
580 it carried on. Also records /dev/nvidia-uvm mapping counts with and without managed memory, the
detection signal from FINDINGS.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from gpunap import procinfo, results
from gpunap.check import base

MB, MANAGED_MB = 128, 64


def verdict(pause: Dict, resume: Optional[Dict], check: str, rc) -> Tuple[str, str]:
    if base.ok(pause):
        if not base.ok(resume):
            return results.HAZARD, f"checkpointed, but resume returned {(resume or {}).get('result')}"
        if check != "ok" or rc != 0:
            return results.HAZARD, f"checkpointed and resumed, but data check {check}, exit {rc}"
        return results.OK, "checkpoint and restore worked with managed memory, data intact"
    if pause.get("hung"):
        return results.HAZARD, "the pause hung"
    if check != "ok" or rc != 0:
        return results.HAZARD, f"pause refused ({pause.get('result')}); afterwards data check {check}, exit {rc}"
    return results.OK, f"pause refused ({pause.get('result')}); the process kept running with its data intact"


def _uvm_count(ctx: base.Context, kind: str) -> Optional[int]:
    try:
        t, s = base.start_target(ctx, kind, "--mb", MB, "--managed-mb", MANAGED_MB)
    except base.NotReady:
        return None
    time.sleep(0.5)
    maps = procinfo.uvm_maps(t.pid)
    base.finish(s, t)
    return None if maps is None else len(maps)


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    plain = _uvm_count(ctx, "hold")
    try:
        t, s = base.start_target(ctx, "managed", "--mb", MB, "--managed-mb", MANAGED_MB)
    except base.NotReady as e:
        return base.not_ready_result("managed_memory", e, time.time() - t0)
    maps = procinfo.uvm_maps(t.pid)
    managed = None if maps is None else len(maps)
    p = base.pause(ctx, t, limit_s=60)
    time.sleep(1.0)
    r = base.resume(ctx, t) if base.ok(p) else None
    check = base.verify(s)
    rc = base.finish(s, t)
    v, summary = verdict(p, r, check, rc)
    data = {"pause": p, "resume": r, "verify": check, "target_rc": rc, "mb": MB, "managed_mb": MANAGED_MB,
            "uvm_maps_plain": plain, "uvm_maps_managed": managed,
            "uvm_signal_separates": None if plain is None or managed is None else plain < managed}
    summary += f"; /dev/nvidia-uvm mappings: {plain} without managed memory, {managed} with"
    return base.result("managed_memory", v, summary, data, t0)


CHECK = base.Check("managed_memory", base.DEFAULT, ("libcuda",), gpu_mb=MB + MANAGED_MB + 700, host_mb=MB + MANAGED_MB,
                   run=run, doc="Pausing a process with CUDA managed memory; /dev/nvidia-uvm mapping counts.")
