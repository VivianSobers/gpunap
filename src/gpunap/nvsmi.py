"""nvidia-smi queries. They return None or an empty list when nvidia-smi is missing, so every caller
works on a machine without an NVIDIA driver."""
from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Sequence

EXE = "nvidia-smi"
GPU_FIELDS = ("name", "driver_version", "memory.used", "memory.free", "memory.total")
PCIE_FIELDS = ("pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max")


def parse_csv(text: str, fields: Sequence[str]) -> List[Dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        vals = [v.strip() for v in line.split(",")]
        rows.append({k: int(v) if v.isdigit() else v for k, v in zip(fields, vals)})
    return rows


def query(args: Sequence[str]) -> str:
    try:
        p = subprocess.run([EXE, *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return p.stdout if p.returncode == 0 else ""


def gpu(index: int = 0) -> Optional[Dict]:
    rows = parse_csv(query([f"--query-gpu={','.join(GPU_FIELDS)}"]), GPU_FIELDS)
    return rows[index] if len(rows) > index else None


def apps() -> List[Dict]:
    """Every compute process on the GPUs, with its memory in MiB."""
    return parse_csv(query(["--query-compute-apps=pid,used_memory"]), ("pid", "used_mb"))


def apps_mb() -> int:
    return sum(a["used_mb"] for a in apps() if isinstance(a["used_mb"], int))


def pcie(index: int = 0) -> Optional[Dict]:
    rows = parse_csv(query([f"--query-gpu={','.join(PCIE_FIELDS)}"]), PCIE_FIELDS)
    return rows[index] if len(rows) > index else None


def gpu_count() -> int:
    return len(parse_csv(query(["--query-gpu=index"]), ("index",)))
