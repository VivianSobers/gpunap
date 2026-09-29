import socket

from gpunap import sysinfo


def test_proprietary_module():
    assert sysinfo.module_kind("NVRM version: NVIDIA UNIX x86_64 Kernel Module  595.84  Release Build") == "proprietary"


def test_open_module():
    assert sysinfo.module_kind("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.71.05") == "open"


def test_missing_module_version():
    assert sysinfo.module_kind(None) is None


def test_meminfo_reads_total_available_and_swap():
    text = "MemTotal:       131581952 kB\nMemAvailable:   2097152 kB\nSwapTotal:             0 kB\n"
    assert sysinfo.meminfo_mb(text) == {"MemTotal": 128498, "MemAvailable": 2048, "SwapTotal": 0}


def test_collect_works_without_a_gpu_and_never_records_the_hostname(monkeypatch):
    monkeypatch.setattr(sysinfo.nvsmi, "EXE", "nvidia-smi-does-not-exist")
    info = sysinfo.collect(driver=None)
    assert info["gpu"] is None and "kernel_release" in info and "python" in info
    assert socket.gethostname() not in repr(info)
