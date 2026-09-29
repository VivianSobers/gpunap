"""sigstop: checkpoint calls on a process stopped with SIGSTOP (Ctrl-Z).

FINDINGS: lock (even with a 5 s timeout), state query and restore all hang until the process gets
SIGCONT; afterwards it recovers with its data intact.
"""
from __future__ import annotations

import os
import signal
import time
from typing import Dict, Tuple

from gpunap import procinfo, results
from gpunap.check import base

MB = 128
LOCK_TIMEOUT_MS, LOCK_LIMIT_S, STATE_LIMIT_S, RESTORE_LIMIT_S = 5000, 15.0, 10.0, 15.0


def verdict(a: Dict, b: Dict, b_paused: bool) -> Tuple[str, str]:
    if not b_paused:
        return results.INVALID, "could not pause the second target before stopping it"
    for name, c in (("stopped while running", a), ("stopped while checkpointed", b)):
        if c["verify"] != "ok" or c["rc"] != 0:
            return results.HAZARD, f"{name}: data check {c['verify']}, exit {c['rc']}"
    hung = [f"{op} ({name})" for name, c in (("running", a), ("checkpointed", b))
            for op, d in c["calls"].items() if d.get("hung")]
    if hung:
        return results.HAZARD, "hung while the process was stopped: " + ", ".join(hung) + "; recovered after SIGCONT with data intact"
    return results.OK, "every call returned while the process was stopped; data intact"


def _stop(pid: int) -> None:
    os.kill(pid, signal.SIGSTOP)
    end = time.time() + 5
    while time.time() < end and procinfo.stat_state(pid) != "T":
        time.sleep(0.02)


def _recover(ctx: base.Context, t) -> Dict:
    st = ctx.call("state", t.pid, STATE_LIMIT_S)
    out = {"state_after_cont": st.get("result")}
    if st.get("result") == "checkpointed":
        out["resume"] = base.resume(ctx, t)
    elif st.get("result") == "locked":
        out["unlock"] = ctx.call("unlock", t.pid, 30)
    return out


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        ta, sa = base.start_target(ctx, "hold", "--mb", MB)
        tb, sb = base.start_target(ctx, "hold", "--mb", MB, "--seed", 1)
    except base.NotReady as e:
        return base.not_ready_result("sigstop", e, time.time() - t0)
    a: Dict = {"calls": {}}
    try:
        _stop(ta.pid)
        a["calls"]["lock"] = ctx.call("lock", ta.pid, LOCK_LIMIT_S, LOCK_TIMEOUT_MS)
        a["calls"]["state"] = ctx.call("state", ta.pid, STATE_LIMIT_S)
    finally:
        os.kill(ta.pid, signal.SIGCONT)
    time.sleep(2)
    a.update(_recover(ctx, ta))
    a["verify"] = base.verify(sa)
    a["rc"] = base.finish(sa, ta)

    b: Dict = {"calls": {}}
    b["pause"] = base.pause(ctx, tb, limit_s=60)
    b_paused = base.ok(b["pause"])
    if b_paused:
        try:
            _stop(tb.pid)
            b["calls"]["restore"] = ctx.call("restore", tb.pid, RESTORE_LIMIT_S)
            b["calls"]["state"] = ctx.call("state", tb.pid, STATE_LIMIT_S)
        finally:
            os.kill(tb.pid, signal.SIGCONT)
        time.sleep(2)
        b.update(_recover(ctx, tb))
    b["verify"] = base.verify(sb)
    b["rc"] = base.finish(sb, tb)
    v, summary = verdict(a, b, b_paused)
    return base.result("sigstop", v, summary, {"stopped_running": a, "stopped_checkpointed": b,
                                               "limits_s": {"lock": LOCK_LIMIT_S, "state": STATE_LIMIT_S,
                                                            "restore": RESTORE_LIMIT_S},
                                               "lock_timeout_ms": LOCK_TIMEOUT_MS}, t0)


CHECK = base.Check("sigstop", base.DEFAULT, ("libcuda",), gpu_mb=2 * MB + 1000, host_mb=MB, run=run,
                   doc="Lock, state query and restore on a process stopped with SIGSTOP.")
