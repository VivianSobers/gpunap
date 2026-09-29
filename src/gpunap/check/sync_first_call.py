"""sync_first_call: a synchronize that is the first CUDA call after a checkpoint.

For each call, a target prints READY, sleeps 3 s and makes exactly that call. The check pauses it
0.5 s after READY, so the call lands while it is checkpointed, then resumes it if it is still alive.
FINDINGS: cuCtxSynchronize and cuStreamSynchronize (and their PyTorch equivalents) segfault; a copy,
an event synchronize, and a synchronize after a small launch wait for the resume.
"""
from __future__ import annotations

import signal
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from gpunap import results
from gpunap.check import base

SYNC_OPS = ("ctx_sync", "stream_sync")
CONTROL_OPS = ("event_sync", "memcpy", "guarded_ctx_sync")
OPS = SYNC_OPS + CONTROL_OPS
SLEEP_S, PAUSE_AT_S, AFTER_RESUME_S = 3.0, 0.5, 30.0
WAIT_S = SLEEP_S + 3.0


def stamp(lines: List[str], tag: str) -> Tuple[Optional[float], Optional[str]]:
    """(time, rc) from the first '<tag> <time> [rc=<rc>]' line."""
    for line in lines:
        f = line.split()
        if f and f[0] == tag:
            rc = next((x[3:] for x in f[2:] if x.startswith("rc=")), None)
            return float(f[1]), rc
    return None, None


def label(pause_ok: bool, ckpt_end, call_t, done_t, done_rc, resume_start, rc) -> str:
    """What happened in one run. Times are unix seconds; rc is the target's exit code or 'timeout'."""
    if not pause_ok:
        return "invalid: pause failed"
    if call_t is None:
        return "invalid: never reached the call"
    if ckpt_end is None or call_t < ckpt_end:
        return "invalid: call came before the checkpoint"
    if rc == "timeout":
        return "hung after resume"
    if rc < 0:
        return f"crash: {signal.Signals(-rc).name}"
    if done_t is None:
        return f"error: exit {rc}"
    outcome = "returned while checkpointed" if resume_start is None or done_t < resume_start else "waited for resume"
    return outcome + ("" if done_rc == "0" else f" (rc={done_rc})")


def verdict(runs: List[Dict]) -> Tuple[str, str]:
    counts: Dict[Tuple[str, str, str], int] = {}
    totals: Dict[Tuple[str, str], int] = {}
    for r in runs:
        if r["label"].startswith("invalid"):
            continue
        counts[(r["api"], r["op"], r["label"])] = counts.get((r["api"], r["op"], r["label"]), 0) + 1
        totals[(r["api"], r["op"])] = totals.get((r["api"], r["op"]), 0) + 1
    if not totals:
        return results.INVALID, "no run landed its call after the checkpoint"
    parts = [f"{api} {op}: {lab} {n}/{totals[(api, op)]}" for (api, op, lab), n in sorted(counts.items())]
    harmful = any(lab.startswith(("crash", "hung", "error")) or "(rc=" in lab for (_, _, lab) in counts)
    return (results.HAZARD if harmful else results.OK), "; ".join(parts)


def run_one(ctx: base.Context, api: str, op: str) -> Dict:
    kind = "sync_first" if api == "driver" else "torch_sync_first"
    r: Dict = {"api": api, "op": op}
    try:
        t, s = base.start_target(ctx, kind, "--op", op, "--sleep", SLEEP_S)
    except base.NotReady as e:
        r.update(label="invalid: never ready", output=e.output, rc=e.rc)
        return r
    time.sleep(PAUSE_AT_S)
    p = base.pause(ctx, t, limit_s=60)
    r["pause"] = p
    ckpt_end = next((c["end"] for c in p.get("calls", []) if c["op"] == "checkpoint" and c["result"] == "OK"), None)
    resume_start = None
    try:
        rc = t.popen.wait(timeout=WAIT_S)
        r["exited_while_paused"] = True
    except subprocess.TimeoutExpired:
        r["exited_while_paused"] = False
        rs = base.resume(ctx, t, limit_s=60)
        r["resume"] = rs
        resume_start = rs.get("start")
        try:
            rc = t.popen.wait(timeout=AFTER_RESUME_S)
        except subprocess.TimeoutExpired:
            rc = "timeout"
    s.settle()
    out = s.output()
    call_t, _ = stamp(out, "CALL")
    done_t, done_rc = stamp(out, "DONE")
    r.update(rc=rc, output=out[-8:], call_after_checkpoint_s=round(call_t - ckpt_end, 3) if call_t and ckpt_end else None,
             label=label(base.ok(p), ckpt_end, call_t, done_t, done_rc, resume_start, rc))
    return r


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    apis = ["driver"] + (["torch"] if base.has_module("torch") else [])
    repeats = 3 if ctx.full else 2
    runs = [run_one(ctx, api, op) for _ in range(repeats) for api in apis for op in OPS]
    v, summary = verdict(runs)
    notes = [] if "torch" in apis else ["PyTorch not installed: driver API only"]
    return base.result("sync_first_call", v, summary, {"repeats": repeats, "apis": apis, "notes": notes,
                                                        "sleep_s": SLEEP_S, "pause_at_s": PAUSE_AT_S, "runs": runs}, t0)


CHECK = base.Check("sync_first_call", base.DEFAULT, ("libcuda",), gpu_mb=1200, host_mb=600, run=run,
                   doc="A synchronize as the first CUDA call after a checkpoint; copies, event synchronize and a "
                       "guarded synchronize as controls.")
