"""Machine facts a result depends on. Never records the hostname."""
from __future__ import annotations

import importlib.metadata
import platform
from typing import Dict, Optional

from gpunap import nvsmi

PACKAGES = ("torch", "nvidia-nccl-cu12", "nvidia-nccl-cu13")


def module_kind(version_line: Optional[str]) -> Optional[str]:
    """'open' or 'proprietary' from the first line of /proc/driver/nvidia/version."""
    if not version_line:
        return None
    return "open" if "Open Kernel Module" in version_line else "proprietary"


def meminfo_mb(text: str) -> Dict[str, int]:
    out = {}
    for line in text.splitlines():
        key = line.split(":")[0]
        if key in ("MemTotal", "MemAvailable", "SwapTotal"):
            out[key] = int(line.split()[1]) // 1024
    return out


def _first_line(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.readline().strip()
    except OSError:
        return None


def _package(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(driver=None) -> Dict:
    """driver: a gpunap.driver.Driver, or None to skip the CUDA driver API version."""
    gpu = nvsmi.gpu()
    try:
        meminfo = open("/proc/meminfo").read()
    except OSError:
        meminfo = ""
    api_version = None
    if driver is not None and driver.available():
        api_version = driver.version()
    return {
        "gpu": gpu["name"] if gpu else None,
        "gpu_count": nvsmi.gpu_count(),
        "driver": gpu["driver_version"] if gpu else None,
        "cuda_driver_api_version": api_version,
        "kernel_module": module_kind(_first_line("/proc/driver/nvidia/version")),
        "gpu_memory_mb": {k: gpu[f"memory.{k}"] for k in ("used", "free", "total")} if gpu else None,
        "pcie_idle": nvsmi.pcie(),
        "kernel_release": platform.release(),
        "memory_mb": meminfo_mb(meminfo),
        "python": platform.python_version(),
        "packages": {n: _package(n) for n in PACKAGES},
    }
