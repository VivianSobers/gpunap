"""restore_oom (--full): restoring into too little free GPU memory, and the claim-before-restore
workaround.

FINDINGS: the failed restore returns OUT_OF_MEMORY and every later restore fails, so the job can
never be restored (NVIDIA issue #44, external source). Holding a placeholder of the job's size,
freeing it and restoring at once avoided it (design rule 1).
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from gpunap import nvsmi, results
from gpunap.check import base
from gpunap.worker import child_env

JOB_MB, HOG_CONTEXT_MB, RESERVE_MARGIN_MB = 1024, 500, 256


def hog_mb(free_mb: int, job_mb: int) -> int:
    """Leave 40 % of the job's footprint free, after the hog's own context."""
    return free_mb - int(job_mb * 0.4) - HOG_CONTEXT_MB


def verdict(first: Dict, attempts: List[Dict], recovered: bool, workaround: Optional[Dict]) -> Tuple[str, str]:
    if base.ok(first):
        return results.INVALID, "the restore succeeded next to the process filling the GPU"
    if workaround is None:
        how = "not run"
    elif workaround.get("result") == "OK" and workaround.get("verify") == "ok":
        how = "workaround worked"
    else:
        how = f"workaround failed ({workaround.get('result')}, data {workaround.get('verify')})"
    if recovered:
        return results.OK, f"restore returned {first.get('result')}, then succeeded once memory was free; {how}"
    later = ", ".join(sorted({a.get("result") for a in attempts})) or "nothing"
    return results.HAZARD, (f"restore returned {first.get('result')}; later restores returned {later}: "
                            f"unrestorable; {how}")


def _job_mb(pid: int) -> int:
    return next((a["used_mb"] for a in nvsmi.apps() if a["pid"] == pid and isinstance(a["used_mb"], int)), JOB_MB)


def _scenario_oom(ctx: base.Context) -> Dict:
    t, s = base.start_target(ctx, "hold", "--mb", JOB_MB)
    need = _job_mb(t.pid)
    d: Dict = {"job_gpu_mb": need, "pause": base.pause(ctx, t)}
    free = (nvsmi.gpu() or {}).get("memory.free", 0)
    d["hog_mb"] = hog_mb(free, need)
    h, hs = base.start_target(ctx, "hog", "--mb", d["hog_mb"])
    d["free_with_hog_mb"] = (nvsmi.gpu() or {}).get("memory.free")
    d["first_restore"] = ctx.call("restore", t.pid, 60)
    base.finish(hs, h)
    time.sleep(2)
    d["attempts"] = []
    for wait in (0, 10):
        time.sleep(wait)
        st = ctx.call("state", t.pid, 30).get("result")
        if st != "checkpointed":
            d["attempts"].append({"state": st})
            break
        d["attempts"].append(ctx.call("restore", t.pid, 60))
        if base.ok(d["attempts"][-1]):
            break
    d["recovered"] = bool(d["attempts"]) and base.ok(d["attempts"][-1])
    if d["recovered"]:
        ctx.call("unlock", t.pid, 30)
        t.resumed = True
        d["verify"] = base.verify(s)
    return d


def _scenario_workaround(ctx: base.Context) -> Dict:
    t, s = base.start_target(ctx, "hold", "--mb", JOB_MB, "--seed", 1)
    need = _job_mb(t.pid) + RESERVE_MARGIN_MB
    d: Dict = {"placeholder_mb": need, "pause": base.pause(ctx, t)}
    helper = subprocess.Popen([ctx.python, "-m", "gpunap.worker", "reserve", str(t.pid), str(need)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env())
    try:
        ready = json.loads(helper.stdout.readline() or "{}")
        d["placeholder_alloc"] = ready.get("alloc")
        free = (nvsmi.gpu() or {}).get("memory.free", 0)
        h, hs = base.start_target(ctx, "hog", "--mb", max(free - 200 - HOG_CONTEXT_MB, 1))
        d["free_with_hog_mb"] = (nvsmi.gpu() or {}).get("memory.free")
        helper.stdin.write("GO\n")
        helper.stdin.flush()
        out = json.loads(helper.stdout.readline() or "{}")
        d.update(result=out.get("result"), free_to_restore_ms=out.get("free_to_restore_ms"), calls=out.get("calls"))
        base.finish(hs, h)
    finally:
        if helper.poll() is None:
            helper.kill()
        helper.wait()
    if d.get("result") == "OK":
        t.resumed = True
        d["verify"] = base.verify(s)
    else:
        d["verify"] = "not checked"
    return d


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    try:
        a = _scenario_oom(ctx)
        ctx.owned.cleanup(ctx.call, ctx.module_kind)
        b = _scenario_workaround(ctx)
    except base.NotReady as e:
        return base.not_ready_result("restore_oom", e, time.time() - t0)
    v, summary = verdict(a["first_restore"], a["attempts"], a["recovered"], b)
    return base.result("restore_oom", v, summary, {"oom": a, "workaround": b}, t0)


CHECK = base.Check("restore_oom", base.FULL, ("libcuda", "idle_gpu"), gpu_mb=0, host_mb=2 * JOB_MB, run=run,
                   doc="Restoring into too little free GPU memory, and the claim-before-restore workaround.")
