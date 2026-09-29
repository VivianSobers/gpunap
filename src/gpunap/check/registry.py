"""Every check, in the order a run executes them."""
from __future__ import annotations

from typing import Dict, List

from gpunap.check import api, cgroup, child_context, errors, exit_while_paused, ipc_shared_memory, managed_memory, nccl_single_gpu, roundtrip, sigstop, sync_first_call
from gpunap.check.base import Check

CHECKS: List[Check] = [api.CHECK, roundtrip.CHECK, errors.CHECK, sync_first_call.CHECK,
                     managed_memory.CHECK, sigstop.CHECK,
                     exit_while_paused.CHECK, child_context.CHECK, cgroup.CHECK,
                     nccl_single_gpu.CHECK, ipc_shared_memory.CHECK]


def by_name() -> Dict[str, Check]:
    return {c.name: c for c in CHECKS}
