"""api: does the driver export the checkpoint API, and which driver and kernel module is this?"""
from __future__ import annotations

import time
from typing import List, Optional

from gpunap import nvsmi, results, sysinfo
from gpunap.check import base
from gpunap.driver import FUNCS, Driver


def verdict(missing: List[str], init_code: Optional[int]) -> str:
    return results.OK if not missing and init_code == 0 else results.HAZARD


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    d = Driver()
    missing, init_code = d.missing(), d.init_code()
    gpu = nvsmi.gpu()
    data = {"functions": list(FUNCS), "missing": missing, "cuInit": d.error_name(init_code) if init_code is not None else None,
            "cuda_driver_api_version": d.version() if init_code == 0 else None,
            "driver": gpu["driver_version"] if gpu else None, "gpu": gpu["name"] if gpu else None,
            "kernel_module": ctx.module_kind}
    v = verdict(missing, init_code)
    summary = (f"all 5 checkpoint functions present, driver {data['driver']}, {ctx.module_kind} kernel module"
               if v == results.OK else f"missing {missing or 'nothing'}, cuInit {data['cuInit']}")
    return base.result("api", v, summary, data, t0)


CHECK = base.Check("api", base.DEFAULT, ("libcuda",), gpu_mb=0, host_mb=0, run=run,
                   doc="The driver exports the five checkpoint functions and cuInit succeeds.")
