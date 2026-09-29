"""cgroup: pausing a process in a memory-limited cgroup, as under Slurm, Docker or systemd.

The checkpoint copies GPU memory into the process's own memory, which counts against its cgroup.
FINDINGS: with a limit above the running size but below the paused size, the process is killed
during the checkpoint.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from gpunap import procinfo, results
from gpunap.check import base

MB = 512


def limit_mb(rss_mb: int, gpu_mb: int) -> int:
    """Above the running size, well below running size plus the copied GPU memory."""
    return rss_mb + gpu_mb // 2


def redact(cg: Optional[Dict]) -> Optional[Dict]:
    """Keep the limit, drop the path: it names the user's UID and session."""
    if not cg:
        return cg
    return {"scope": cg["path"].rsplit("/", 1)[-1], "max_bytes": cg["max_bytes"],
            "limit_on_own_cgroup": cg["limit_at"] == cg["path"], "current_bytes": cg["current_bytes"]}


def verdict(cg: Optional[Dict], alive: bool, check: str, rc) -> Tuple[str, str]:
    if not cg or cg.get("max_bytes") is None:
        return results.INVALID, "the cgroup memory limit did not apply"
    limit = cg["max_bytes"] // 2 ** 20
    if not alive or rc != 0:
        how = " (SIGKILL)" if rc == -9 else ""
        return results.HAZARD, f"killed during the checkpoint with a {limit} MB cgroup limit: exit {rc}{how}"
    if check != "ok":
        return results.HAZARD, f"survived a {limit} MB limit, but data check {check}"
    return results.OK, f"survived the checkpoint under a {limit} MB cgroup limit, data intact"


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        probe, ps = base.start_target(ctx, "hold", "--mb", MB)
    except base.NotReady as e:
        return base.not_ready_result("cgroup", e, time.time() - t0)
    rss = procinfo.vm_rss_mb(probe.pid) or 0
    base.finish(ps, probe)
    limit = limit_mb(rss, MB)
    prefix = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}M", "-p", "MemorySwapMax=0"]
    try:
        t, s = base.start_target(ctx, "hold", "--mb", MB, prefix=prefix)
    except base.NotReady as e:
        return base.not_ready_result("cgroup", e, time.time() - t0)
    cg = redact(procinfo.cgroup_memory(t.pid))
    p = base.pause(ctx, t, limit_s=60)
    time.sleep(1.0)
    alive = t.alive()
    r, check = None, "not checked"
    if alive:
        if ctx.call("state", t.pid, 30).get("result") == "checkpointed":
            r = base.resume(ctx, t)
        check = base.verify(s)
    rc = base.finish(s, t)
    v, summary = verdict(cg, alive, check, rc)
    return base.result("cgroup", v, summary, {"probe_rss_mb": rss, "gpu_mb": MB, "memory_max_mb": limit, "cgroup": cg,
                                              "pause": p, "resume": r, "alive_after_pause": alive, "verify": check,
                                              "target_rc": rc}, t0)


CHECK = base.Check("cgroup", base.DEFAULT, ("libcuda", "systemd-run"), gpu_mb=2 * MB + 700, host_mb=MB, run=run,
                   doc="Pausing a process in a memory-limited cgroup.")
