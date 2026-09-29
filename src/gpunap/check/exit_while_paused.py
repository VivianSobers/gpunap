"""exit_while_paused: a process that exits, with no further CUDA call, while it is checkpointed.

FINDINGS: it exits with code 0, so a scheduler has to treat a vanished PID as normal.
"""
from __future__ import annotations

import signal
import subprocess
import time
from typing import Optional, Tuple

from gpunap import results
from gpunap.check import base
from gpunap.check.sync_first_call import stamp

SLEEP_S, PAUSE_AT_S, WAIT_S = 3.0, 0.5, 10.0


def verdict(pause_ok: bool, ckpt_end: Optional[float], exiting_t: Optional[float], exited_while_paused: bool,
            rc) -> Tuple[str, str]:
    if not pause_ok:
        return results.INVALID, "pause failed"
    if exiting_t is not None and ckpt_end is not None and exiting_t < ckpt_end:
        return results.INVALID, "the exit started before the checkpoint finished"
    if rc != 0:
        name = f" ({signal.Signals(-rc).name})" if isinstance(rc, int) and rc < 0 else ""
        return results.HAZARD, f"exit code {rc}{name}"
    if exiting_t is None:
        return results.INVALID, "the target never reached its exit"
    if exited_while_paused:
        return results.OK, "exited with code 0 while checkpointed"
    return results.OK, "the exit waited for the resume, then code 0"


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        t, s = base.start_target(ctx, "exit_after", "--sleep", SLEEP_S)
    except base.NotReady as e:
        return base.not_ready_result("exit_while_paused", e, time.time() - t0)
    time.sleep(PAUSE_AT_S)
    p = base.pause(ctx, t, limit_s=60)
    ckpt_end = next((c["end"] for c in p.get("calls", []) if c["op"] == "checkpoint" and c["result"] == "OK"), None)
    r = None
    try:
        rc = t.popen.wait(timeout=WAIT_S)
        exited = True
    except subprocess.TimeoutExpired:
        exited = False
        r = base.resume(ctx, t)
        try:
            rc = t.popen.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = "timeout"
    s.wait_for("NEVER", 0.3)
    exiting_t, _ = stamp(s.output(), "EXITING")
    v, summary = verdict(base.ok(p), ckpt_end, exiting_t, exited, rc)
    return base.result("exit_while_paused", v, summary, {"pause": p, "resume": r, "exited_while_paused": exited,
                                                         "target_rc": rc, "output": s.output(8)}, t0)


CHECK = base.Check("exit_while_paused", base.DEFAULT, ("libcuda",), gpu_mb=700, host_mb=64, run=run,
                   doc="A process that exits, with no further CUDA call, while checkpointed.")
