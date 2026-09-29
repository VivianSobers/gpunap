"""Read-only probes of /proc and cgroup v2. Every function takes the root as an argument so it can be
tested against a fake tree, and returns None when the process is gone."""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _stat_fields(pid: int, proc: str) -> Optional[List[str]]:
    """Fields after the command name: [state, ppid, ...]. The name may contain spaces and parens."""
    text = _read(f"{proc}/{pid}/stat")
    if text is None or ")" not in text:
        return None
    return text.rsplit(")", 1)[1].split()


def stat_state(pid: int, proc: str = "/proc") -> Optional[str]:
    f = _stat_fields(pid, proc)
    return f[0] if f else None


def start_time(pid: int, proc: str = "/proc") -> Optional[int]:
    """Start time in clock ticks since boot (field 22 of stat)."""
    f = _stat_fields(pid, proc)
    return int(f[19]) if f and len(f) > 19 else None


def cmdline(pid: int, proc: str = "/proc") -> Optional[List[str]]:
    text = _read(f"{proc}/{pid}/cmdline")
    return None if text is None else [x for x in text.split("\0") if x]


def children(pid: int, proc: str = "/proc") -> List[int]:
    out = []
    for name in os.listdir(proc):
        if name.isdigit():
            f = _stat_fields(int(name), proc)
            if f and int(f[1]) == pid:
                out.append(int(name))
    return sorted(out)


def uvm_maps(pid: int, proc: str = "/proc") -> Optional[List[str]]:
    """Mappings of /dev/nvidia-uvm, the managed-memory signal in FINDINGS."""
    text = _read(f"{proc}/{pid}/maps")
    if text is None:
        return None
    return [line.split()[0] + " " + line.split()[-1] for line in text.splitlines() if "nvidia-uvm" in line]


def thread_names(pid: int, proc: str = "/proc") -> Optional[List[str]]:
    try:
        tids = os.listdir(f"{proc}/{pid}/task")
    except OSError:
        return None
    names = [(_read(f"{proc}/{pid}/task/{t}/comm") or "").strip() for t in tids]
    return sorted(n for n in names if n)


def mem_available_mb(meminfo: Optional[str] = None) -> Optional[int]:
    text = meminfo if meminfo is not None else _read("/proc/meminfo") or ""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def cgroup_path(pid: int, proc: str = "/proc") -> Optional[str]:
    text = _read(f"{proc}/{pid}/cgroup")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip()
    return None


def _int_or_none(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None


def cgroup_memory(pid: int, proc: str = "/proc", sys_fs: str = "/sys/fs/cgroup") -> Optional[Dict]:
    """The tightest memory.max on the process's cgroup path, and the current usage of its own cgroup."""
    path = cgroup_path(pid, proc)
    if path is None:
        return None
    parts = [p for p in path.split("/") if p]
    best, at = None, None
    for i in range(len(parts) + 1):
        sub = "/" + "/".join(parts[:i])
        m = _int_or_none(_read(os.path.join(sys_fs, *parts[:i], "memory.max")))
        if m is not None and (best is None or m < best):
            best, at = m, sub
    current = _int_or_none(_read(os.path.join(sys_fs, *parts, "memory.current")))
    return {"path": path, "max_bytes": best, "limit_at": at, "current_bytes": current}
