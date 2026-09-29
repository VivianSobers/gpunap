"""controller_death (--full): the controller killed partway through a checkpoint and partway through
a restore.

For each phase, a worker makes the one driver call (checkpoint on a locked job, or restore on a
checkpointed one) and is killed with SIGKILL 0.25 s after it announces the call. The check then
polls the job's state for 10 s, brings it back the way the state calls for, and verifies its data.
FINDINGS: the checkpoint completed without its controller (595); the restore completed but the job
stayed locked until another controller unlocked it (both drivers).
"""
from __future__ import annotations

import json
import select
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from gpunap import nvsmi, procinfo, results
from gpunap.check import base
from gpunap.check.size_sweep import fits
from gpunap.worker import child_env

SIZES_MB = (16384, 8192, 4096, 2048)
KILL_AFTER_S, POLLS, POLL_EVERY_S = 0.25, 20, 0.5
PHASES = ("checkpoint", "restore")


def pick_mb(free_mb: Optional[int], host_mb: Optional[int]) -> Optional[int]:
    return next((mb for mb in SIZES_MB if fits(mb, free_mb, host_mb) is None), None)


def outcome(phase: str, states: List[str]) -> str:
    last = states[-1] if states else "unknown"
    if last == "checkpointed":
        return "the checkpoint completed without the controller" if phase == "checkpoint" else \
            "the restore was abandoned, the job stayed checkpointed"
    if last == "locked":
        return "the restore completed, the job stayed locked" if phase == "restore" else \
            "the checkpoint was abandoned, the job stayed locked"
    if last == "running":
        return "the job is running"
    return f"the job is in the {last} state"


def recovery(state: str) -> Optional[str]:
    return {"checkpointed": "resume", "locked": "unlock"}.get(state)


def verdict(phases: List[Dict]) -> Tuple[str, str]:
    if all("skipped" in p for p in phases):
        return results.SKIPPED, phases[0]["skipped"]
    parts, bad, invalid = [], [], []
    for p in phases:
        name = p["phase"]
        if "not_ready" in p or "prepare_failed" in p:
            invalid.append(f"{name}: {'target never became ready' if 'not_ready' in p else p['prepare_failed']}")
            continue
        if not p["killed_during_call"]:
            invalid.append(f"{name}: the call finished before the kill")
            continue
        last = p["states"][-1] if p["states"] else "unknown"
        if last not in ("checkpointed", "locked", "running"):
            bad.append(f"{name}: {outcome(name, p['states'])}")
        elif p["recover"] is not None and not base.ok(p["recover"]):
            bad.append(f"{name}: recovery returned {p['recover'].get('result')}")
        elif p["verify"] != "ok":
            bad.append(f"{name}: data check {p['verify']}")
        elif p["rc"] != 0:
            bad.append(f"{name}: target exited with {p['rc']}")
        parts.append(f"{name}: {outcome(name, p['states'])}")
    if bad:
        return results.HAZARD, "; ".join(bad)
    if invalid:
        return results.INVALID, "; ".join(invalid)
    return results.OK, "; ".join(parts) + "; a new controller brought the job back with its data intact"


def _announced(c: subprocess.Popen, timeout: float) -> Optional[float]:
    """The time from the worker's CALLING line, or None if it never came."""
    if not select.select([c.stdout], [], [], timeout)[0]:
        return None
    line = c.stdout.readline()
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d.get("time") if d.get("event") == "CALLING" else None


def run_phase(ctx: base.Context, phase: str, mb: int) -> Dict:
    d: Dict = {"phase": phase, "mb": mb}
    try:
        t, s = base.start_target(ctx, "hold", "--mb", mb, "--seed", 7, ready_timeout=300)
    except base.NotReady as e:
        d["not_ready"] = {"output": e.output, "rc": e.rc}
        return d
    prep = ctx.call("lock", t.pid, 30) if phase == "checkpoint" else base.pause(ctx, t, limit_s=300)
    d["prepare"] = prep
    if not base.ok(prep):
        d["prepare_failed"] = f"preparing the job returned {prep.get('result')}"
        d["rc"] = base.finish(s, t)
        return d
    c = subprocess.Popen([ctx.python, "-m", "gpunap.worker", phase, str(t.pid), "--announce"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=child_env())
    called = _announced(c, 60)
    if called is not None:
        time.sleep(max(0.0, KILL_AFTER_S - (time.time() - called)))
    d["killed_during_call"] = called is not None and c.poll() is None
    kill_t = time.time()
    c.kill()
    c.wait()
    d["kill_after_s"] = round(kill_t - called, 3) if called is not None else None
    d["worker_output"] = c.stdout.read()[-500:]
    c.stdout.close()
    d["states"], d["polls"] = [], []
    for _ in range(POLLS):
        st = ctx.call("state", t.pid, 30)
        d["states"].append(st.get("result"))
        d["polls"].append({"at_s": round(st.get("start", kill_t) - kill_t, 2), "state": st.get("result"),
                           "seconds": st.get("seconds")})
        time.sleep(POLL_EVERY_S)
    step = recovery(d["states"][-1])
    if step == "resume":
        d["recover"] = base.resume(ctx, t, limit_s=300)
    elif step == "unlock":
        d["recover"] = ctx.call("unlock", t.pid, 30)
        if base.ok(d["recover"]):
            t.resumed = True
    else:
        d["recover"] = None
    if d["recover"] is None or base.ok(d["recover"]):
        d["verify"] = base.verify(s, timeout=300)
    else:
        d["verify"] = "not checked"
    d["rc"] = base.finish(s, t, timeout=60)
    return d


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    phases = []
    for phase in PHASES:
        time.sleep(1)
        free, host = (nvsmi.gpu() or {}).get("memory.free"), procinfo.mem_available_mb()
        mb = pick_mb(free, host)
        if mb is None:
            phases.append({"phase": phase, "skipped": fits(SIZES_MB[-1], free, host)})
            continue
        phases.append(run_phase(ctx, phase, mb))
    v, summary = verdict(phases)
    return base.result("controller_death", v, summary, {"kill_after_s": KILL_AFTER_S, "phases": phases}, t0)


CHECK = base.Check("controller_death", base.FULL, ("libcuda", "idle_gpu"), gpu_mb=SIZES_MB[-1] + 2560,
                   host_mb=SIZES_MB[-1] + 2048, run=run,
                   doc="The controller killed 0.25 s into a checkpoint and into a restore of a job of up to 16 GB.")
