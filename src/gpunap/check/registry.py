"""Every check, in the order a run executes them."""
from __future__ import annotations

from typing import Dict, List

from gpunap.check import api, errors, managed_memory, roundtrip, sigstop, sync_first_call
from gpunap.check.base import Check

CHECKS: List[Check] = [api.CHECK, roundtrip.CHECK, errors.CHECK, sync_first_call.CHECK,
                     managed_memory.CHECK, sigstop.CHECK]


def by_name() -> Dict[str, Check]:
    return {c.name: c for c in CHECKS}
