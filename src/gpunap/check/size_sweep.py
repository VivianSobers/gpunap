"""size_sweep (--full): pause and resume time against the size of the GPU state, data intact.

Each size that fits gets its own target and three pause, resume and verify cycles. A size fits when
the GPU has the size plus GPU_MARGIN_MB free (the target's own context and headroom for others) and
the host has the size plus HOST_MARGIN_MB available, since a checkpoint moves the GPU state into host
memory. FINDINGS: checkpoint and restore time grow with size, and the data survives at every size.
"""
from __future__ import annotations

import statistics
import time
from typing import Dict, List, Optional, Tuple

from gpunap import nvsmi, procinfo, results
from gpunap.check import base

SIZES_MB = (512, 1024, 2048, 4096, 8192, 16384, 20480)
CYCLES = 3
GPU_MARGIN_MB, HOST_MARGIN_MB = 2560, 2048


def fits(mb: int, free_mb: Optional[int], host_mb: Optional[int]) -> Optional[str]:
    """None if a target of mb fits, else why not."""
    if free_mb is None or free_mb < mb + GPU_MARGIN_MB:
        return f"not enough free GPU memory ({free_mb} MB free)"
    if host_mb is None or host_mb < mb + HOST_MARGIN_MB:
        return f"not enough host memory ({host_mb} MB available)"
    return None


def gb(mb: int) -> str:
    return f"{mb / 1024:g} GB"


def rate(mb: int, seconds: Optional[float]) -> Optional[float]:
    return round(mb / 1024 / seconds, 2) if seconds else None


def timings(mb: int, cycles: List[Dict]) -> Dict:
    """Median checkpoint and restore time over the cycles, and the rates they give."""
    ck = [x for x in (base.call_seconds(c["pause"], "checkpoint") for c in cycles) if x is not None]
    rs = [x for x in (base.call_seconds(c["resume"], "restore") for c in cycles) if x is not None]
    out: Dict = {"checkpoint_s": statistics.median(ck) if ck else None,
                 "restore_s": statistics.median(rs) if rs else None}
    out["checkpoint_gb_s"] = rate(mb, out["checkpoint_s"])
    out["restore_gb_s"] = rate(mb, out["restore_s"])
    return out


def _problem(size: Dict) -> Optional[str]:
    problem = base.cycle_problem(size["cycles"])
    if problem:
        return f"{gb(size['mb'])} {problem}"
    if size.get("rc") != 0:
        return f"{gb(size['mb'])}: target exited with {size.get('rc')}"
    return None


def verdict(sizes: List[Dict]) -> Tuple[str, str]:
    problems = [p for p in (_problem(s) for s in sizes if "cycles" in s) if p]
    if problems:
        return results.HAZARD, "; ".join(problems)
    stalled = [gb(s["mb"]) for s in sizes if "not_ready" in s]
    if stalled:
        return results.INVALID, f"target never became ready at {', '.join(stalled)}"
    ran = [s for s in sizes if "cycles" in s]
    skipped = [gb(s["mb"]) for s in sizes if "skipped" in s]
    if not ran:
        return results.SKIPPED, f"no size fits: {sizes[0]['skipped']}" if sizes else "no sizes"
    parts = []
    for s in ran:
        t = timings(s["mb"], s["cycles"])
        parts.append(f"{gb(s['mb'])}: checkpoint {t['checkpoint_s']:.2f} s, restore {t['restore_s']:.2f} s")
    tail = f"; {', '.join(skipped)} skipped" if skipped else ""
    return results.OK, f"data intact at every size; {'; '.join(parts)}{tail}"


def run_size(ctx: base.Context, mb: int) -> Dict:
    d: Dict = {"mb": mb}
    try:
        t, s = base.start_target(ctx, "hold", "--mb", mb, "--seed", mb, ready_timeout=300)
    except base.NotReady as e:
        d["not_ready"] = {"output": e.output, "rc": e.rc}
        return d
    d["cycles"] = []
    for _ in range(CYCLES):
        c = {"pause": base.pause(ctx, t, limit_s=300)}
        time.sleep(0.5)
        c["resume"] = base.resume(ctx, t, limit_s=300) if base.ok(c["pause"]) else {"result": "not attempted"}
        c["verify"] = base.verify(s, timeout=300)
        d["cycles"].append(c)
        if c["verify"] != "ok":
            break
    d["rc"] = base.finish(s, t, timeout=60)
    d.update(timings(mb, d["cycles"]))
    return d


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    sizes = []
    for mb in SIZES_MB:
        time.sleep(1)  # let the previous target's memory return before measuring
        free = (nvsmi.gpu() or {}).get("memory.free")
        why = fits(mb, free, procinfo.mem_available_mb())
        if why:
            sizes.append({"mb": mb, "skipped": why})
            continue
        sizes.append(run_size(ctx, mb))
        if _problem(sizes[-1]) or "not_ready" in sizes[-1]:
            break
    v, summary = verdict(sizes)
    return base.result("size_sweep", v, summary, {"cycles": CYCLES, "gpu_margin_mb": GPU_MARGIN_MB,
                                                  "host_margin_mb": HOST_MARGIN_MB, "sizes": sizes}, t0)


CHECK = base.Check("size_sweep", base.FULL, ("libcuda", "idle_gpu"), gpu_mb=SIZES_MB[0] + GPU_MARGIN_MB,
                   host_mb=SIZES_MB[0] + HOST_MARGIN_MB, run=run,
                   doc="Pause and resume time against size, up to 20 GB, with the data checked after every cycle.")
