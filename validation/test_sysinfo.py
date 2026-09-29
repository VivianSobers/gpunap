from sysinfo import meminfo_mb, module_kind


def test_proprietary_module():
    line = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  595.84  Thu Jan  1 00:00:00 UTC 2026"
    assert module_kind(line) == "proprietary"


def test_open_module():
    line = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.71.05  Release Build"
    assert module_kind(line) == "open"


def test_meminfo_reads_total_and_swap():
    text = "MemTotal:       131581952 kB\nMemFree:  1 kB\nSwapTotal:             0 kB\n"
    assert meminfo_mb(text) == {"MemTotal": 128498, "SwapTotal": 0}
