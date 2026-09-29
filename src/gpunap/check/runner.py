"""Runs checks one at a time: pre-flight, the check, cleanup, the report.

Cleanup runs after every check, including one that raised or was interrupted. Ctrl-C stops the run
and returns the results gathered so far, marked partial.
"""
from __future__ import annotations

import shutil
import sys
import time
import traceback
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from gpunap import nvsmi, procinfo, results, sysinfo, worker
from gpunap.check import base, owned
from gpunap.driver import Driver

GPU_HEADROOM_MB = HOST_HEADROOM_MB = 1024
# Other processes may hold this much GPU memory and the GPU still counts as idle. A desktop session
# keeps a few hundred MB on the GPU for good; the report records what was there.
IDLE_OTHER_MB = 1024

MISSING = {"libcuda": "no CUDA driver with the checkpoint API (libcuda.so.1, driver 550 or newer)",
           "torch": "PyTorch is not installed",
           "systemd-run": "systemd-run is not available"}


def _others(apps: Sequence[Dict]) -> Tuple[int, int]:
    return len(apps), sum(a["used_mb"] for a in apps if isinstance(a.get("used_mb"), int))


def preflight(check: base.Check, gpu: Optional[Dict], apps: Sequence[Dict], mem_avail: Optional[int],
              module_kind: Optional[str], full: bool, have: Set[str]) -> Tuple[bool, Optional[str]]:
    """Whether check may run now, and why not. Pure: every input is passed in."""
    if full and module_kind == "open":
        return False, ("--full is refused on the open kernel module: killing a resumed process there can "
                       "hang the machine (NVIDIA issue #53)")
    for need in check.needs:
        if need in MISSING and need not in have:
            return False, MISSING[need]
    if gpu is None:
        return False, "nvidia-smi reports no GPU"
    free = gpu.get("memory.free")
    if not isinstance(free, int) or free < check.gpu_mb + GPU_HEADROOM_MB:
        return False, f"needs {check.gpu_mb + GPU_HEADROOM_MB} MB of free GPU memory, {free} MB free"
    if mem_avail is None or mem_avail < check.host_mb + HOST_HEADROOM_MB:
        return False, f"needs {check.host_mb + HOST_HEADROOM_MB} MB of available host memory, {mem_avail} MB available"
    if "idle_gpu" in check.needs:
        n, mb = _others(apps)
        if mb > IDLE_OTHER_MB:
            return False, f"needs an idle GPU: {n} other processes use {mb} MB (at most {IDLE_OTHER_MB} MB allowed)"
    return True, None


def select(checks: Sequence[base.Check], only: Optional[Sequence[str]]) -> List[base.Check]:
    if not only:
        return list(checks)
    names = {c.name for c in checks}
    unknown = [n for n in only if n not in names]
    if unknown:
        raise ValueError(f"unknown check: {', '.join(unknown)}")
    return [c for c in checks if c.name in only]


class Probe:
    """The machine as pre-flight sees it. Tests pass a fake."""

    def __init__(self, driver: Optional[Driver] = None):
        self.driver = driver or Driver()

    def gpu(self) -> Optional[Dict]:
        return nvsmi.gpu()

    def apps(self) -> List[Dict]:
        return nvsmi.apps()

    def mem_available_mb(self) -> Optional[int]:
        return procinfo.mem_available_mb()

    def have(self) -> Set[str]:
        out = set()
        if self.driver.available():
            out.add("libcuda")
        if base.has_module("torch"):
            out.add("torch")
        if shutil.which("systemd-run"):
            out.add("systemd-run")
        return out

    def sysinfo(self) -> Dict:
        return sysinfo.collect(self.driver)


def module_kind() -> Optional[str]:
    try:
        with open("/proc/driver/nvidia/version") as f:
            return sysinfo.module_kind(f.readline())
    except OSError:
        return None


def new_context(label: str, full: bool) -> base.Context:
    token = "gpunap-" + uuid.uuid4().hex[:12]
    return base.Context(owned=owned.OwnedProcesses(token), call=worker.call, token=token,
                        python=sys.executable, label=label, module_kind=module_kind(), full=full)


def _used(gpu: Optional[Dict]):
    return gpu.get("memory.used") if gpu else None


def run(checks: Sequence[base.Check], ctx: base.Context, probe=None,
        on_result: Optional[Callable[[results.CheckResult], None]] = None) -> results.Report:
    probe = probe or Probe()
    have = probe.have()
    gpu, apps = probe.gpu(), probe.apps()
    n, mb = _others(apps)
    header = {"label": ctx.label, "token": ctx.token, "full": ctx.full, "started": time.time(),
              "checks": [c.name for c in checks], "system": probe.sysinfo(), "available": sorted(have),
              "other_gpu_processes": {"count": n, "mb": mb}, "idle_other_mb_allowed": IDLE_OTHER_MB,
              "gpu_memory_used_mb_before": _used(gpu),
              "reconciled": owned.reconcile(ctx.owned.run_dir, ctx.call, ctx.module_kind)}
    report = results.Report(header)

    def add(r: results.CheckResult) -> None:
        report.results.append(r)
        if on_result:
            on_result(r)

    try:
        for check in checks:
            if check.mode == base.FULL and not ctx.full:
                add(results.CheckResult(check.name, results.SKIPPED, "needs --full"))
                continue
            ok, why = preflight(check, probe.gpu(), probe.apps(), probe.mem_available_mb(), ctx.module_kind,
                                ctx.full, have)
            if not ok:
                add(results.CheckResult(check.name, results.SKIPPED, why))
                continue
            t0 = time.time()
            try:
                r = check.run(ctx)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                r = results.CheckResult(check.name, results.ERROR, f"gpunap raised {type(e).__name__}: {e}",
                                        {"traceback": traceback.format_exc()}, time.time() - t0)
            finally:
                cleaned = ctx.owned.cleanup(ctx.call, ctx.module_kind)
            if cleaned:
                r.data["cleanup"] = cleaned
            add(r)
    except KeyboardInterrupt:
        report.partial = True
        ctx.owned.cleanup(ctx.call, ctx.module_kind)
    finally:
        ctx.owned.close()
    header["ended"] = time.time()
    after = probe.apps()
    header["gpu_memory_used_mb_after"] = _used(probe.gpu())
    header["other_gpu_processes_after"] = dict(zip(("count", "mb"), _others(after)))
    return report
